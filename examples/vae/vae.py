# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

from tqdm import tqdm

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from utils.npmi import compute_npmi_at_n_during_training

def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix(
        (npy['data'], (npy['row'], npy['col'])), shape=npy['shape']
    )
    return coo_matrix.tocsc()


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # setup linear transformations
        self.en_fc1 = nn.Linear(args.vocab_size, args.encoder_hidden_dim)
        
        # second layer (potentially optional)
        self.en_fc2 = nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim)
        self.en_fc_drop = nn.Dropout(args.encoder_dropout)

        self.alpha_layer = nn.Linear(args.encoder_hidden_dim, args.num_topics)
        self.alpha_bn_layer = nn.BatchNorm1d(
            args.num_topics#, eps=0.001, momentum=0.001, affine=True
        )

        # Do not use BN scale params (seems to help)
        self.alpha_bn_layer.weight.data.copy_(torch.ones(args.num_topics))
        self.alpha_bn_layer.weight.requires_grad = False

    def forward(self, x):
        en1 = F.relu(self.en_fc1(x))
        en2 = F.relu(self.en_fc2(en1))
        en2_do = self.en_fc_drop(en2)

        alpha = self.alpha_layer(en2_do)
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
        self.beta_layer = nn.Linear(args.num_topics, args.vocab_size)
        self.beta_bn_layer = nn.BatchNorm1d(args.vocab_size)
        
        # Do not use BN scale parameters
        self.beta_bn_layer.weight.data.copy_(torch.ones(args.vocab_size))
        self.beta_bn_layer.weight.requires_grad = False

    def forward(self, z):
        eta = self.beta_layer(z)
        eta_bn = self.beta_bn_layer(eta)
        x_recon = F.softmax(eta_bn, dim=-1)#.log()
        return x_recon

    @property
    def topics(self):
        return self.beta_layer.weight.T.cpu().detach().numpy()


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
    def model(self, x):
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
            x_recon = self.decoder(z)
            # score against actual data (TODO: is using multinomial like this ok?)
            pyro.sample("obs", dist.Multinomial(args.max_doc_length, x_recon), obs=x)
            
            return x_recon

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(x)
            # sample the latent code z
            pyro.sample("doc_topics", dist.Dirichlet(z))


def main(args):
    # clear param store
    pyro.clear_param_store()
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    # load the data
    data = load_sparse(args.counts_fpath)
    data = torch.tensor(data.todense(), dtype=torch.float32)
    args.vocab_size = data.shape[1]
    args.max_doc_length = int(data.max())
    if args.dev_split > 0:
        split_idx = np.random.choice(
            (True, False),
            size=data.shape[0],
            p=(1-args.dev_split, args.dev_split),
        )
        x_train, x_dev = data[split_idx], data[~split_idx]
        n_train, n_dev = x_train.shape[0], x_dev.shape[0]
    else:
        n_train = data.shape[0]
        x_train = data

    # load the vocabulary
    if args.vocab_fpath is not None:
        import json
        with open(args.vocab_fpath) as infile:
            vocab = json.load(infile)
            inv_vocab = dict(zip(vocab.values(), vocab.keys()))

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
    dev_metrics = {
        "loss": np.inf,
        "npmi": 0,
    }

    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        x_train = x_train[np.random.choice(n_train, size=n_train, replace=False)] #shuffle
        train_batches = n_train // args.batch_size

        epoch_loss = 0.
        for i in range(train_batches):
            # if on GPU put mini-batch into CUDA memory
            x_batch = x_train[i * args.batch_size:(i + 1) * args.batch_size]
            if args.cuda:
                x_batch = x_batch.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x_batch)

        # report training diagnostics
        epoch_loss /= n_train
        train_elbo.append(epoch_loss)
        print(f"{epoch}  average training loss: {epoch_loss:0.4f}")

        # evaluate on the dev set
        if args.dev_split > 0 and epoch % args.eval_step == 0:
            eval_batches = n_dev // args.batch_size
            
            # get loss
            dev_loss = 0.
            for i in range(eval_batches):
                x_batch = x_dev[i * args.batch_size:(i + 1) * args.batch_size]
                if args.cuda:
                    x_batch = x_batch.cuda()
                dev_loss += svi.evaluate_loss(x_batch)

            dev_loss /= n_dev

            # get npmi
            beta = vae.decoder.topics
            npmi = compute_npmi_at_n_during_training(beta, x_dev.numpy(), n=args.npmi_words)

            dev_metrics['loss'] = min(dev_loss, dev_metrics['loss'])
            dev_metrics['npmi'] = max(npmi, dev_metrics['npmi'])

            print(f"dev loss: {dev_loss:0.4f}, npmi: {npmi:0.4f}")
    
    return vae, dev_metrics


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-i", "--counts-fpath", default=None)
    parser.add_argument("--vocab-fpath", default=None)
    parser.add_argument("--dev-split", default=0.2, type=float)
    
    parser.add_argument("-k", "--num-topics", default=50, type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.002, type=float)
    parser.add_argument("--encoder-hidden-dim", default=100, type=int)
    parser.add_argument("--encoder-dropout", default=0.2, type=float)
    parser.add_argument("--alpha-prior", default=0.02, type=float)

    parser.add_argument("-b", "--batch-size", default=200, type=int)
    parser.add_argument("-n", '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument("--eval-step", default=1, type=int)
    parser.add_argument("--npmi-words", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    model = main(args)
    import ipdb; ipdb.set_trace()
