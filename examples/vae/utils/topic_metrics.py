import numpy as np

def compute_npmi_at_n_during_training(beta, ref_counts, n=10, smoothing=0.01, return_mean=False):

    n_docs, _ = ref_counts.shape

    n_topics, vocab_size = beta.shape

    npmi_means = []
    for k in range(n_topics):
        order = np.argsort(beta[k, :])[::-1]
        indices = order[:n]
        npmi_vals = []
        for i, index1 in enumerate(indices):
            for index2 in indices[i+1:n]:
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
    return np.mean(npmi_means) if return_mean else npmi_means

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

def compute_tr(topics, l=10):
    """
    Compute topic redundancy score from 
    https://pdfs.semanticscholar.org/2898/149eb423782ad2211c35eca98fdc4e08780a.pdf
    """
    tr_results = []
    k = len(topics)
    for i, topics_i in enumerate(topics):
        w_counts = 0
        for w in topics_i[:l]:
            w_counts += np.sum([w in topics_j[:l] for j, topics_j in enumerate(topics) if j != i]) # count(k, l)
        tr_results.append((1 / (k - 1)) * w_counts)
    return tr_results


def compute_topic_overlap(topics, word_overlap_threshold=10, n=10):
    """
    Calculate topic overlap
    """
    overlapping_topics = set()
    for i, t_i in enumerate(topics):
        for j, t_j in enumerate(topics[i+1:], start=i+1):
            if len(set(t_i[:n]) & set(t_j[:n])) >= word_overlap_threshold:
                overlapping_topics |= set([i, j])
    return len(overlapping_topics)