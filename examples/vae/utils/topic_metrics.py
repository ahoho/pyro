import numpy as np

def compute_npmi_at_n(
    topics, ref_vocab, ref_counts, n=10, cols_to_skip=0, silent=False, return_mean=True
):

    vocab_index = dict(zip(ref_vocab, range(len(ref_vocab))))
    n_docs, _ = ref_counts.shape

    npmi_means = []
    for topic in topics:
        words = topic.strip().split()[cols_to_skip:]
        npmi_vals = []
        for word_i, word1 in enumerate(words[:n]):
            if word1 in vocab_index:
                index1 = vocab_index[word1]
            else:
                index1 = None
            for word2 in words[word_i+1:n]:
                if word2 in vocab_index:
                    index2 = vocab_index[word2]
                else:
                    index2 = None
                if index1 is None or index2 is None:
                    npmi = 0.0
                else:
                    col1 = np.array(ref_counts[:, index1] > 0, dtype=int)
                    col2 = np.array(ref_counts[:, index2] > 0, dtype=int)
                    c1 = col1.sum()
                    c2 = col2.sum()
                    c12 = np.sum(col1 * col2)
                    if c12 == 0:
                        npmi = 0.0
                    else:
                        npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        if not silent:
            print(str(np.mean(npmi_vals)) + ': ' + ' '.join(words[:n]))
        npmi_means.append(np.mean(npmi_vals))
    if not silent:
        print(np.mean(npmi_means))
    if return_mean:
        return np.mean(npmi_means)
    else:
        return np.array(npmi_means)

def compute_npmi_at_n_during_training(beta, ref_counts, n=10, smoothing=0.01):

    n_docs, _ = ref_counts.shape

    n_topics, vocab_size = beta.shape

    npmi_means = []
    for k in range(n_topics):
        order = np.argsort(beta[k, :])[::-1]
        indices = order[:n]
        npmi_vals = []
        for index1 in indices:
            for index2 in indices:
                col1 = np.array((ref_counts[:, index1] > 0), dtype=int) + smoothing
                col2 = np.array((ref_counts[:, index2] > 0), dtype=int) + smoothing
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = np.sum(col1 * col2)
                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        npmi_means.append(np.mean(npmi_vals))
    return np.mean(npmi_means)

def compute_tu(topics, l=10):
    """
    Topic uniqueness measure from https://www.aclweb.org/anthology/P19-1640.pdf
    """
    tu_results = []
    for topics_i in topics:
        w_counts = 0
        for w in topics_i[:l]:
            w_counts += 1 / np.sum([w in topics_j[:l] for topics_j in topics]) # count(k, l)
        tu_results.append((1 / l) * w_counts)
    return tu_results