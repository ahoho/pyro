import json
import shutil
from pathlib import Path
from pyro.primitives import get_param_store

from tqdm import trange

import configargparse
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from utils.topic_metrics import compute_npmi_at_n_during_training, compute_tu

def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix(
        (npy['data'], (npy['row'], npy['col'])), shape=npy['shape']
    )
    return coo_matrix.tocsc()


def load_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)


def save_json(obj, fpath):
    with open(fpath, "w") as outfile:
        return json.dump(obj, outfile)


def load_embeddings(fpath, vocab):
    """
    Load word embeddings and align with vocabulary
    """
    pretrained = np.load(Path(fpath, "vectors.npy"))
    pretrained_vocab = load_json(Path(fpath, "vocab.json"))
    embeddings = np.random.rand(pretrained.shape[1], len(vocab)) * 0.25 - 5

    for word, idx in vocab.items():
        if word in pretrained_vocab:
            embeddings[:, idx] = pretrained[pretrained_vocab[word], :]

    return embeddings
    

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # setup linear transformations
        encoder_input_dim = args.vocab_size * 2 if args.encode_doc_reps else args.vocab_size
        self.embedding_layer = nn.Linear(encoder_input_dim, args.embeddings_dim)

        # map embeddings
        if args.pretrained_embeddings is not None:
            embeddings = torch.tensor(args.pretrained_embeddings)
            self.embedding_layer.weight.data.copy_(embeddings)
            self.embedding_layer.weight.requires_grad = args.update_embeddings

        self.fc = nn.Linear(args.embeddings_dim, args.encoder_hidden_dim)
        self.fc_drop = nn.Dropout(args.encoder_dropout)

        if args.second_hidden_layer:
            self.second_hidden_layer = True
            self.fc2 = nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim)
            self.fc2_drop = nn.Dropout(args.encoder_dropout)
        
        self.alpha_layer = nn.Linear(args.encoder_hidden_dim, args.num_topics)
        self.alpha_bn_layer = nn.BatchNorm1d(
            args.num_topics, eps=0.001, momentum=0.001, affine=True
        )

        # Do not use BN scale params (seems to help)
        self.alpha_bn_layer.weight.data.copy_(torch.ones(args.num_topics))
        self.alpha_bn_layer.weight.requires_grad = args.learn_bn_scale

    def forward(self, x, annealing_factor=1.0):
        embedded = F.relu(self.embedding_layer(x))

        hidden = F.relu(self.fc(embedded))
        hidden_do = self.fc_drop(hidden)

        if args.second_hidden_layer:
            hidden = F.relu(self.fc2(hidden)) # don't dropout the first layer
            hidden_do = self.fc2_drop(hidden)

        alpha = self.alpha_layer(hidden_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.00001, device=alpha_bn.device)
        )

        return alpha_pos


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.z_drop = nn.Dropout(args.decoder_dropout) # TODO: re-softmax?

        self.eta_layer = nn.Linear(args.num_topics, args.vocab_size)
        self.eta_bn_layer = nn.BatchNorm1d(
            args.vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        
        # Do not use BN scale parameters
        self.eta_bn_layer.weight.data.copy_(torch.ones(args.vocab_size))
        self.eta_bn_layer.weight.requires_grad = args.learn_bn_scale

    def forward(self, z, annealing_factor=0.0):
        z_do = self.z_drop(z)
        eta = self.eta_layer(z_do)
        eta_bn = self.eta_bn_layer(eta)

        x_recon = (
            (annealing_factor) * F.softmax(eta, dim=-1)
            + (1 - annealing_factor) * F.softmax(eta_bn, dim=-1)
        )
        return x_recon
    
    @property
    def beta(self):
        return self.eta_layer.weight.T.cpu().detach().numpy()


class CollapsedMultinomial(dist.Multinomial):
    """
    Equivalent to n separate Multinomial(1, probs), where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of Multinomial(n, probs))
    """
    def log_prob(self, value):
        return ((self.probs.log() + 1e-10) * value).sum(-1)


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        if args.cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = args.cuda
        self.num_topics = args.num_topics
        self.alpha_prior = args.alpha_prior

        self.encode_doc_reps = args.encode_doc_reps
        self.distillation_weight = args.distillation_weight

    # define the model p(x|z)p(z)
    def model(self, x, doc_reps, annealing_factor=1.0):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            alpha_0 = torch.ones(
                x.shape[0], self.num_topics, device=x.device
            ) * self.alpha_prior
            
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("doc_topics", dist.Dirichlet(alpha_0))
            # decode the latent code z
            x_recon = self.decoder(z, annealing_factor)
            # score against actual data
            if self.distillation_weight > 0:
                # TODO: align with our KD
                alpha = self.distillation_weight
                doc_reps = doc_reps * (doc_reps > 1e-5).float()
                pseudo_x = doc_reps * x.sum(1, keepdims=True)
                x = (1 - alpha) * x + alpha * pseudo_x

            pyro.sample("obs", CollapsedMultinomial(1., x_recon), obs=x)

            return x_recon

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, doc_reps, annealing_factor=0.0):
        # register PyTorch module `encoder` with Pyro
        input = torch.cat([x, doc_reps], dim=1) if self.encode_doc_reps else x
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", input.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(input)
            # sample the latent code z
            pyro.sample("doc_topics", dist.Dirichlet(z))


def calculate_annealing_factor(args, epoch, minibatch, batches_per_epoch):
    """
    Calculate annealing factor. Taken from /examples/dmm.py
    """
    if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
        # taken from examples/dmm.py
        min_af = args.minimum_annealing_factor
        annealing_factor = (
            min_af + (1.0 - min_af) *
            (
                (minibatch + epoch * batches_per_epoch + 1) /
                (args.annealing_epochs * batches_per_epoch)
            )
        )
    elif args.annealing_epochs == 0:
        annealing_factor = 0.0
    else:
        annealing_factor = 1.0
    

    return annealing_factor

def main(args):
    # clear param store
    pyro.clear_param_store()
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    model_dir = args.temp_model_dir or args.output_dir
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # load the data
    x_train = load_sparse(Path(args.data_dir, args.counts_fpath))
    x_train = torch.tensor(x_train.astype(np.float32).todense())
    n_train = x_train.shape[0]

    args.vocab_size = x_train.shape[1]
    args.max_doc_length = int(x_train.max())

    if not args.dev_counts_fpath and args.dev_split > 0:
        split_idx = np.random.choice(
            (True, False),
            size=x_train.shape[0],
            p=(1-args.dev_split, args.dev_split),
        )
        x_train, x_dev = x_train[split_idx], x_train[~split_idx]
        n_train, n_dev = x_train.shape[0], x_dev.shape[0]
        args.max_doc_length = max(args.max_doc_length, int(x_dev.max()))

    if args.dev_counts_fpath:
        x_dev = load_sparse(Path(args.data_dir, args.dev_counts_fpath))
        x_dev = torch.tensor(x_dev.astype(np.float32).todense())
        n_dev = x_dev.shape[0]
        args.max_doc_length = max(args.max_doc_length, int(x_dev.max()))

    if args.encode_doc_reps or args.distillation_weight > 0:
        if args.doc_reps_source == "tf-idf":
            tfidf = TfidfTransformer()
            tfidf.fit(x_train)

            doc_reps_train = torch.tensor(
                tfidf.transform(x_train).todense(), dtype=torch.float
            )
            doc_reps_dev = torch.tensor(
                tfidf.transform(x_dev).todense(), dtype=torch.float
            )
        else:
            doc_reps_train = torch.tensor(
                np.load(Path(args.doc_reps_source, "train.npy"))
            )
            doc_reps_dev = torch.tensor(
                np.load(Path(args.doc_reps_source, "dev.npy"))
            )
            assert(doc_reps_train.shape[0] == x_train.shape[0])
            assert(doc_reps_dev.shape[0] == x_dev.shape[0])
    else:
        doc_reps_train, doc_reps_dev = torch.tensor([]), torch.tensor([])

    # load the vocabulary
    if args.vocab_fpath is not None:
        vocab = load_json(Path(args.data_dir, args.vocab_fpath))

    # load the embeddings
    if args.pretrained_embeddings is not None:
        args.pretrained_embeddings = load_embeddings(args.pretrained_embeddings, vocab)
        args.embeddings_dim = args.pretrained_embeddings.shape[0]
    else:
        args.embeddings_dim = args.encoder_hidden_dim
    
    # setup the VAE
    vae = VAE(args)

    # setup the optimizer
    adam_args = {
        "lr": args.learning_rate,
        "betas": (0.99, 0.999), # from ProdLDA
    }
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = []
    results = []
    dev_metrics = {
        "loss": np.inf,
        "npmi": 0,
        "tu": 0,
    }
    target = args.dev_metric_target

    # training loop
    t = trange(args.num_epochs, leave=True)
    for epoch in t:
        # initialize loss accumulator
        random_idx = np.random.choice(n_train, size=n_train, replace=False)
        x_train = x_train[random_idx] #shuffle
        doc_reps_train = doc_reps_train[random_idx] if len(doc_reps_train) > 0 else doc_reps_train
        train_batches = n_train // args.batch_size

        epoch_loss = 0.
        for i in range(train_batches):
            annealing_factor = calculate_annealing_factor(args, epoch, i, train_batches)
            # if on GPU put mini-batch into CUDA memory
            x_batch = x_train[i * args.batch_size:(i + 1) * args.batch_size]
            doc_reps_batch = doc_reps_train[i * args.batch_size:(i + 1) * args.batch_size]

            if args.cuda:
                x_batch = x_batch.cuda()
                doc_reps_batch = doc_reps_batch.cuda()
            # do ELBO gradient and accumulate loss
            annealing_factor = torch.tensor(annealing_factor, device=x_batch.device)
            epoch_loss += svi.step(x_batch, doc_reps_batch, annealing_factor)

        # report training diagnostics
        epoch_loss /= n_train
        train_elbo.append(epoch_loss)


        # evaluate on the dev set
        result_message = {"tr loss": f"{epoch_loss:0.1f}"}
        if (args.dev_counts_fpath or args.dev_split > 0) and epoch % args.eval_step == 0:
            eval_batches = n_dev // args.batch_size
            
            # get loss
            dev_loss = 0.
            for i in range(eval_batches):
                x_batch = x_dev[i * args.batch_size:(i + 1) * args.batch_size]
                doc_reps_batch = doc_reps_dev[i * args.batch_size:(i + 1) * args.batch_size]
                if args.cuda:
                    x_batch = x_batch.cuda()
                    doc_reps_batch = doc_reps_batch.cuda()
                dev_loss += svi.evaluate_loss(x_batch, doc_reps_batch, annealing_factor)

            dev_loss /= n_dev

            # get npmi
            beta = vae.decoder.beta
            npmi = compute_npmi_at_n_during_training(beta, x_dev.numpy(), n=args.npmi_words, smoothing=0.)

            # finally, topic-uniqueness
            topic_terms = [word_probs.argsort()[::-1] for word_probs in beta]
            tu = np.mean(compute_tu(topic_terms, l=args.tu_words))

            dev_metrics['loss'] = min(dev_loss, dev_metrics['loss'])
            dev_metrics['npmi'] = max(npmi, dev_metrics['npmi'])
            dev_metrics['tu'] = max(tu, dev_metrics['tu'])

            if dev_metrics[target] == {"loss": dev_loss, "npmi": npmi, "tu": tu}[target]:
                pyro.get_param_store().save(Path(model_dir, "model.pt"))

            result_message.update({
                "dev loss": f"{dev_loss:0.1f}",
                "npmi": f"{npmi:0.4f}",
                "tu": f"{tu:0.4f}"
            })
            results.append((epoch, epoch_loss, dev_loss, npmi, tu))
        t.set_postfix(result_message)

    if results:
        results_df = pd.DataFrame(
            results, columns=["epoch", "train_loss", "dev_loss", "dev_npmi", "dev_tu"]
        )
        results_df.to_csv(Path(args.output_dir, "results.csv"))
        t.write(
            f"Best NPMI: {results_df.dev_npmi.max():0.4f} @ {np.argmax(results_df.dev_npmi)}\n"
            f"Best TU @ this NPMI: {results_df.dev_tu[np.argmax(results_df.dev_npmi)]:0.4f}"
        )
    if args.temp_model_dir:
        shutil.copyfile(Path(model_dir, "model.pt"), Path(args.output_dir, "model.pt"))

    return vae, dev_metrics


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    # parse command line arguments
    parser = configargparse.ArgParser(description="parse args")

    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--output_dir", required=True, default=None)
    parser.add("--temp_model_dir", default=None, help="Temporary model storage during run, when I/O bound")


    parser.add("--data_dir", default=None)
    parser.add("-i", "--counts_fpath", default="train.npz")
    parser.add("-v", "--vocab_fpath", default="train.vocab.json")
    parser.add("-d", "--dev_counts_fpath", default="dev.npz")
    parser.add("--dev_split", default=0.2, type=float)
    
    parser.add("-k", "--num_topics", default=50, type=int)
    
    parser.add("--encoder_hidden_dim", default=100, type=int)
    parser.add("--encoder_dropout", default=0.2, type=float)
    parser.add("--decoder_dropout", default=0.2, type=float)
    parser.add("--learn_bn_scale", default=False, action="store_true")
    parser.add("--alpha_prior", default=0.02, type=float)
    parser.add("--pretrained_embeddings_dir", dest="pretrained_embeddings", default=None, help="directory containing vocab.json and vectors.npy")
    parser.add("--update_embeddings", action="store_true", default=False)
    parser.add("--second_hidden_layer", action="store_true", default=False)
    
    parser.add("--encode_doc_reps", action="store_true", default=False)
    parser.add("--distillation_weight", default=0.0, type=float)
    parser.add("--doc_reps_source", default="tf_idf")

    parser.add('-lr', '--learning_rate', default=0.002, type=float)
    parser.add("-b", "--batch_size", default=200, type=int)
    parser.add("-n", '--num_epochs', default=101, type=int)
    parser.add("--annealing_epochs", default=50, type=int)
    parser.add("--minimum_annealing_factor", default=0.01, type=float)
    
    parser.add("--eval_step", default=1, type=int)
    parser.add("--dev_metric_target", default="npmi", choices=["npmi", "loss", "tu"])
    parser.add("--npmi_words", default=10, type=int)
    parser.add("--tu_words", default=10, type=int)

    parser.add("--seed", default=42, type=int)
    parser.add('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    model, metrics = main(args)
