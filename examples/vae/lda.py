import logging
import json
import shutil
from pathlib import Path

import configargparse
from gensim.models import ldamodel
import numpy as np
import pandas as pd
from scipy import sparse

from gensim.matutils import Sparse2Corpus
from gensim.models.wrappers import LdaMallet
from gensim.models.ldamulticore import LdaMulticore, LdaModel
from gensim.utils import check_output

from utils.topic_metrics import compute_npmi_at_n_during_training, compute_tu, compute_topic_overlap

logger = logging.getLogger(__name__)

PATH_TO_MALLET_BINARY = "/workspace/kd-topic-modeling/Mallet/bin/mallet"

def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix(
        (npy['data'], (npy['row'], npy['col'])), shape=npy['shape']
    )
    return coo_matrix.tocsc().astype(np.int32)


def load_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)


def save_json(obj, fpath):
    with open(fpath, "w") as outfile:
        return json.dump(obj, outfile, indent=2)


class LdaMalletWithBeta(LdaMallet):
    def __init__(self, beta=None, *args, **kwargs):
        self.beta = beta
        super().__init__(*args, **kwargs)
    
    def train(self, corpus):
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --beta %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s'

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.beta, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed)
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

def main(args):

    np.random.seed(args.seed)
    x_train = load_sparse(Path(args.data_dir, args.counts_fpath))

    model_dir = args.temp_model_dir or args.output_dir
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if not args.dev_counts_fpath and args.dev_split > 0:
        split_idx = np.random.choice(
            (True, False),
            size=x_train.shape[0],
            p=(1-args.dev_split, args.dev_split),
        )
        x_train, x_dev = x_train[split_idx], x_train[~split_idx]

    if args.dev_counts_fpath:
        x_dev = load_sparse(Path(args.data_dir, args.dev_counts_fpath))

    x_train = Sparse2Corpus(x_train, documents_columns=False)
    x_dev = np.array(x_dev.todense())

    # load the vocabulary
    vocab = None
    if args.vocab_fpath is not None:
        vocab = load_json(Path(args.data_dir, args.vocab_fpath))
        vocab = {i: v for i, v in enumerate(vocab)}

    if args.model == "gensim":
        extra_kwargs = {}
        lda_class = LdaModel
        if not args.alpha == "auto" and not args.eta == "auto":
            lda_class = LdaMulticore
            extra_kwargs["workers"] = args.workers
        lda = lda_class(
            corpus=x_train,
            num_topics=args.num_topics,
            id2word=vocab,
            alpha=args.alpha,
            eta=args.eta,
            minimum_probability=0.,
            iterations=args.iterations,
            random_state=args.seed,
            **extra_kwargs,
        )

    if args.model == "mallet":
        lda = LdaMalletWithBeta(
            mallet_path=PATH_TO_MALLET_BINARY,
            corpus=x_train,
            num_topics=args.num_topics,
            id2word=vocab,
            alpha=args.alpha,
            beta=args.beta,
            optimize_interval=args.optimize_interval,
            topic_threshold=0.,
            iterations=args.iterations,
            prefix=model_dir,
            workers=args.workers,
            random_seed=args.seed,
        )

    topics = lda.get_topics()
    topic_terms = [word_probs.argsort()[::-1] for word_probs in topics]
    npmi = compute_npmi_at_n_during_training(topics, x_dev, n=args.n_eval_words, smoothing=0.)
    tu = compute_tu(topic_terms, l=args.n_eval_words)
    metrics = {
        'dev_npmi': npmi,
        'dev_npmi_mean': np.mean(npmi),
        'tu': tu,
        'tu_mean': np.mean(tu),
    }
    for th in args.topic_overlap_thresholds:
        metrics[f'topic_overlap_at_{th}'] = compute_topic_overlap(
            topic_terms, word_overlap_threshold=th, n=args.n_eval_words
        )
    np.save(Path(model_dir, "beta.npy"), topics)
    save_json(metrics, Path(model_dir, "metrics.json"))

    if args.temp_model_dir:
        shutil.copyfile(Path(model_dir, "metrics.json"), Path(args.output_dir, "metrics.json"))
        shutil.copyfile(Path(model_dir, "beta.npy"), Path(args.output_dir, "beta.npy"))

    return lda, metrics

if __name__ == "__main__":
    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    # Data
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--output_dir", required=True, default=None)
    parser.add("--temp_model_dir", default=None, help="Temporary model storage during run, when I/O bound")

    parser.add("--data_dir", default=None)
    parser.add("-i", "--counts_fpath", default="train.npz")
    parser.add("-v", "--vocab_fpath", default="train.vocab.json")
    parser.add("-d", "--dev_counts_fpath", default="dev.npz")
    parser.add("--dev_split", default=0.2, type=float)

    # Model-specific hyperparams
    parser.add("-k", "--num_topics", default=50, type=int)
    parser.add("--model", required=True, choices=["mallet", "gensim"])
    
    parser.add("--alpha", default=None)
    parser.add("--iterations", default=None, type=int)

    ## Gensim-only
    parser.add("--eta", default=None)

    ## Mallet-only
    parser.add("--beta", default=0.01, type=float)
    parser.add("--optimize_interval", type=int, default=0)
    
    # Evaluation
    parser.add("--n_eval_words", default=10, type=int)
    parser.add("--topic_overlap_thresholds", default=[7, 10], nargs="+", type=int)

    # Run settings
    parser.add("--run_seeds", default=[42], type=int, nargs="+", help="Seeds to use for each run")
    parser.add("--workers", default=4, type=int)
    args = parser.parse_args()

    data_dir_map = {
        "20ng": "/workspace/kd-topic-modeling/data/20ng-prodlda/replicated/dev/",
        "imdb": "/workspace/kd-topic-modeling/data/imdb/processed-dev/",
        "wiki": "/workspace/kd-topic-modeling/data/wikitext/processed/new-dev/",
    }
    args.data_dir = data_dir_map[args.data_dir]

    if args.model == "gensim":
        args.alpha = "symmetric" if args.alpha is None else args.alpha
        args.eta = "symmetric" if args.eta is None else args.eta
        args.iterations = 50 if args.iterations is None else args.iterations

        try:
            args.alpha = float(args.alpha)
        except ValueError:
            args.alpha = args.alpha

        try:
            args.eta = float(args.eta)
        except ValueError:
            args.eta = args.eta

        
    if args.model == "mallet":
        args.alpha = 5.0 if args.alpha is None else args.alpha
        args.iterations = 1000 if args.iterations is None else args.iterations

    # Run for each seed
    base_output_dir = args.output_dir
    Path(base_output_dir).mkdir(exist_ok=True, parents=True)

    for i, seed in enumerate(args.run_seeds):
        # make subdirectories for each run
        args.seed = seed
        output_dir = Path(base_output_dir, str(seed))
        output_dir.mkdir(exist_ok=True, parents=True)
        args.output_dir = str(output_dir)
    
        # train
        print(f"\nOn run {i} of {len(args.run_seeds)}")
        model, metrics = main(args)
    
    # Aggregate results
    if len(args.run_seeds) > 1:
        agg_run_results = []
        for seed in args.run_seeds:
            output_dir = Path(base_output_dir, str(seed))
            try:
                metrics = load_json(Path(output_dir, "metrics.json"))
            except FileNotFoundError:
                continue
            metrics.pop("dev_npmi")
            metrics.pop("tu")
            agg_run_results.append(metrics)

        agg_run_results_df = pd.DataFrame.from_records(agg_run_results)
        agg_run_results_df.to_csv(Path(base_output_dir, "run_results.csv"))
        print(
            f"\n=== Results over {len(args.run_seeds)} runs ===\n"
            f"Mean NPMI: "
            f"{agg_run_results_df.dev_npmi_mean.mean():0.4f} ({agg_run_results_df.dev_npmi_mean.std():0.4f}) "
        )