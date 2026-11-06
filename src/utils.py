"""Build, store, load, and parse inverted indexes.

Author: Josh Borthick
Date Completed: 9/12/2026
Assignment: CSC734 HW3 

_get_cpu_count
Returns the number of physical cores on the device.

_chunks
Creates parallelizable chunks of files for processing.

_save_index
Saves the postings to a pickle file.

_load_index
Loads the postings from a pickle file.

greeting
Prints a greeting to the screen.
"""

from collections import Counter
from math import log10, sqrt
from psutil import cpu_count
import heapq
import pickle

from src.normalizer import Normalizer


def chunks(files, n_batches):
    batch_size = (len(files) + n_batches - 1) // n_batches

    return [files[start:start + batch_size]
            for start in range(0, len(files), batch_size)]


def save_index(pkl_file, postings):
    """Save inverted index as pickle file."""
    with open(pkl_file, "wb") as pf:
        pickle.dump(postings, pf, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(pkl_file):
    """Load inverted index from pickle file."""
    with open(pkl_file, "rb") as pf:
        postings = pickle.load(pf)
    return postings


def get_cpu_count():
    """Return the number of physical CPUs."""
    return cpu_count(logical=False)


def tf(counts, ifds):
    """Return the """
    return dict(counts)


def tfidf(counts, idfs):
    return {t: c * idfs.get(t, 0.0) for t, c in counts.items()}


def wfidf(counts, idfs):
    return {t: (1 + log10(c)) * idfs.get(t, 0.0) for t, c in counts.items() if c > 0}


def norm(vec):
    return sqrt(sum(w * w for w in vec.values()))


def cosine(a, na, b, nb):
    if not na or not nb:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items()) / (na * nb)


def count_terms(path):
    return Counter(normalizer.get_tokens(path, from_file=True))


SCHEMES = {"tf": tf, "tf_idf": tfidf, "wf_idf": wfidf}
normalizer = Normalizer("files/stopwords.txt")
_VECS = None   # {scheme: [(vector, norm), ...]} shared with workers via initializer


def init_worker(vecs):
    global _VECS
    _VECS = vecs


def top_pairs(rows, k):
    """Compare each doc i in rows with every doc j > i; keep top k per 
    scheme.
    """
    best = {}
    for scheme, vecs in _VECS.items():
        n = len(vecs)
        sims = ((cosine(*vecs[i], *vecs[j]), i, j)
                for i in rows for j in range(i + 1, n))
        best[scheme] = heapq.nlargest(k, sims)
    return best


def greeting():
    print("\n=================== CSC734-IR Homework 03 ==============\n"
          + "First Name: Josh\n"
          + "Last Name : Borthick\n"
          + "========================================================\n")
