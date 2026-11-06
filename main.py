from concurrent.futures import ProcessPoolExecutor
from math import log10
from pathlib import Path

import heapq


def term_inc_matrix():
    pass


def main():
    docs = sorted(str(p) for p in Path("documents/").iterdir() if p.is_file())
    # print(docs)

    ti_matrix = term_inc_matrix(docs)
    print(ti_matrix)


if __name__ == "__main__":
    main()