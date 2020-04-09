import argparse
import json
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import pyro
import pyro.distributions as dist
from pyro.infer import JitTrace_ELBO, Trace_ELBO

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
        self.embedding_layer = nn.Linear(args.vocab_size, args.embeddings_dim)

        # map embeddings to pretrained
        if args.pretrained_embeddings is not None:
            embeddings = torch.tensor(args.pretrained_embeddings)
            self.embedding_layer.weight.data.copy_(embeddings)
            self.embedding_layer.weight.requires_grad = args.update_pretrained_embeddings
        
        # TODO: allow creation of additional hidden layers

        self.hidden_drop = nn.Dropout(args.encoder_dropout)

        self.alpha_layer = nn.Linear(args.embeddings_dim, args.num_topics)
        self.alpha_bn_layer = nn.BatchNorm1d(
            args.num_topics, eps=0.001, momentum=0.001, affine=True
        )

        # Do not use BN scale params (we are porting from TF)
        self.alpha_bn_layer.weight.data.copy_(torch.ones(args.num_topics))
        self.alpha_bn_layer.weight.requires_grad = False

    def forward(self, x):
        embedded = F.relu(self.embedding_layer(x))
        embedded_do = self.hidden_drop(embedded)

        alpha = self.alpha_layer(embedded_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.001, device=alpha_bn.device)
        )

        return alpha_pos


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.z_drop = nn.Dropout(args.decoder_dropout)
        self.eta_layer = nn.Linear(args.num_topics, args.vocab_size, bias=False)
        self.eta_bn_layer = nn.BatchNorm1d(
            args.vocab_size, eps=0.001, momentum=0.001, affine=True
        )

        # Do not use BN scale parameters
        self.eta_bn_layer.weight.data.copy_(torch.ones(args.vocab_size))
        self.eta_bn_layer.weight.requires_grad = False

    def forward(self, z, annealing_factor=1.0):
        z_do = self.z_drop(z)
        eta = self.eta_layer(z_do)
        eta_bn = self.eta_bn_layer(eta)

        x_recon = (
            (annealing_factor) * F.softmax(eta, dim=-1) +
            (1 - annealing_factor) * F.softmax(eta_bn, dim=-1)
        )
        return x_recon

    def masked_beta_sim(self, args):
        """
        Compute pairwise topic similarities, with the goal of minimizing it.
        """
        # must normalize else optimization is unbounded
        # beta_sim = F.normalize(self.eta_layer.weight, p=2, dim=0)
        beta_sim = F.softmax(self.eta_layer.weight, dim=0)
        topic_cov_masked = torch.tril((beta_sim.T @ beta_sim), diagonal=-1)
        topic_cov_norm = torch.norm(topic_cov_masked, p=args.beta_sim_power)
        return topic_cov_norm * args.beta_sim_weight
    
    @property
    def beta(self):
        return self.eta_layer.weight.T.cpu().detach().numpy()


class CollapsedMultinomial(dist.Multinomial):
    """
    Equivalent to n separate Multinomial(1, probs), where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of Multinomial(n, probs))

    This aligns with the loss in the various neural topic modeling papers
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

    # define the model p(x|z)p(z)
    def model(self, x, annealing_factor=1.0):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            alpha_0 = torch.ones(
                x.shape[0], self.num_topics, device=x.device
            ) * self.alpha_prior
            
            # sample from prior (value will be sampled by guide when computing the ELBO)
            # with pyro.poutine.scale(None, annealing_factor):
            z = pyro.sample("doc_topics", dist.Dirichlet(alpha_0))
            # decode the latent code z
            x_recon = self.decoder(z, annealing_factor=annealing_factor)
            # score against actual data
            pyro.sample("obs", CollapsedMultinomial(1., x_recon), obs=x)
            
            return x_recon

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, annealing_factor=1.0):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(x)
            # sample the latent code z
            #with pyro.poutine.scale(None, annealing_factor):
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
    else:
        annealing_factor = 1.0
    

    return annealing_factor

def main(args):
    # clear param store
    pyro.clear_param_store()
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    # load the data
    x_train = load_sparse(args.counts_fpath)
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
        x_dev = load_sparse(args.dev_counts_fpath)
        x_dev = torch.tensor(x_dev.astype(np.float32).todense())
        n_dev = x_dev.shape[0]
        args.max_doc_length = max(args.max_doc_length, int(x_dev.max()))

    # load the vocabulary
    if args.vocab_fpath is not None:
        vocab = load_json(args.vocab_fpath)
        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda kv: kv[1])}

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
        "betas": (0.9, 0.999), # from ProdLDA
    }
    optimizer = Adam(vae.parameters(), **adam_args) #dev loss: 671.8198, npmi: 0.1316, tr: 0.0103

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    loss_fn = elbo.differentiable_loss

    train_elbo = []
    results = []
    dev_metrics = {
        "loss": np.inf,
        "npmi": 0,
        "tu": np.inf,
    }
    from scipy.spatial.distance import pdist

    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        x_train = x_train[np.random.choice(n_train, size=n_train, replace=False)] #shuffle
        train_batches = n_train // args.batch_size

        epoch_loss = 0.
        for i in range(train_batches):
            annealing_factor = calculate_annealing_factor(args, epoch, i, train_batches)
            # if on GPU put mini-batch into CUDA memory
            x_batch = x_train[i * args.batch_size:(i + 1) * args.batch_size]
            if args.cuda:
                x_batch = x_batch.cuda()
            # do ELBO gradient and accumulate loss
            annealing_factor = torch.tensor(annealing_factor, device=x_batch.device)
            loss = (
                loss_fn(vae.model, vae.guide, x_batch, annealing_factor)
                + vae.decoder.masked_beta_sim(args)
            )
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # report training diagnostics
        epoch_loss /= n_train
        train_elbo.append(epoch_loss)
        # print(f"{epoch}  average training loss: {epoch_loss:0.4f}")

        # evaluate on the dev set
        if (args.dev_counts_fpath or args.dev_split > 0) and epoch % args.eval_step == 0:
            eval_batches = n_dev // args.batch_size
            
            # get loss
            dev_loss = 0.
            for i in range(eval_batches):
                x_batch = x_dev[i * args.batch_size:(i + 1) * args.batch_size]
                if args.cuda:
                    x_batch = x_batch.cuda()
                dev_loss += (
                    loss_fn(vae.model, vae.guide, x_batch, annealing_factor)
                    + vae.decoder.masked_beta_sim(args)
                ).item()
            dev_loss /= n_dev

            # get npmi
            beta = vae.decoder.beta
            npmi = compute_npmi_at_n_during_training(beta, x_dev.numpy(), n=args.npmi_words, smoothing=0.)

            # finally, topic-uniqueness
            topic_terms = [word_probs.argsort()[::-1] for word_probs in beta]
            tu = np.mean(compute_tu(topic_terms, args.tu_words))

            dev_metrics['loss'] = min(dev_loss, dev_metrics['loss'])
            dev_metrics['npmi'] = max(npmi, dev_metrics['npmi'])
            dev_metrics['tu'] = min(tu, dev_metrics['tu'])

            beta_dists = pdist(F.softmax(torch.tensor(beta), dim=-1).detach()).sum()
            result = (
                f"{epoch}: tr loss: {epoch_loss:0.4f}, beta_dists: {beta_dists:0.4f}, beta sim: {vae.decoder.masked_beta_sim(args).item():0.4f} "
                f"dev loss: {dev_loss:0.4f}, npmi: {npmi:0.4f}, tu: {tu:0.4f}"
            )
            print(result)
            results.append((epoch, epoch_loss, dev_loss, npmi, tu, beta_dists))

    if args.save_all:
        import shutil
        from datetime import datetime

        import pandas as pd
        
        now = datetime.now().strftime("%Y-%m-%d_%H%M")
        outpath = Path(f"results/{now}")
        outpath.mkdir()

        shutil.copy("vae.py", Path(outpath, "vae.py"))
        save_json(args.__dict__, Path(outpath, "args.json"))
        results_df = pd.DataFrame(
            results, columns=["epoch", "train_loss", "dev_loss", "dev_npmi", "dev_tu", "beta_dists"]
        )
        results_df.to_csv(Path(outpath, "results.csv"))

    import ipdb; ipdb.set_trace()
    return vae, dev_metrics


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")

    parser.add_argument("-i", "--counts-fpath", default=None)
    parser.add_argument("--vocab-fpath", default=None)
    parser.add_argument("-d", "--dev-counts-fpath", default=None)
    parser.add_argument("--dev-split", default=0.1, type=float)
    parser.add_argument("--save-all", action="store_true", default=False)
    
    parser.add_argument("-k", "--num-topics", default=50, type=int)
    
    parser.add_argument("--encoder-hidden-dim", default=100, type=int)
    parser.add_argument("--encoder-dropout", default=0.6, type=float)
    parser.add_argument("--decoder-dropout", default=0.0, type=float)
    parser.add_argument("--alpha-prior", default=0.01, type=float)
    parser.add_argument("--pretrained-embeddings-dir", dest="pretrained_embeddings", default=None, help="directory containing vocab.json and vectors.npy")
    parser.add_argument("--update-pretrained-embeddings", action="store_true", default=False)

    parser.add_argument("--beta-sim-weight", default=0., type=float)
    parser.add_argument("--beta-sim-power", default=1, type=float, choices=[1, 2])

    parser.add_argument('-lr', '--learning-rate', default=0.002, type=float)
    parser.add_argument("-b", "--batch-size", default=200, type=int)
    parser.add_argument("-n", '--num-epochs', default=300, type=int)
    parser.add_argument("--annealing-epochs", default=100, type=int)
    parser.add_argument("--minimum-annealing-factor", default=0.0, type=float)
    
    parser.add_argument("--eval-step", default=1, type=int)
    parser.add_argument("--npmi-words", default=10, type=int)
    parser.add_argument("--tu-words", default=10, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    model, metrics = main(args)
