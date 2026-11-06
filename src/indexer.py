"""Build, store, load, and parse inverted indexes.

Author: Josh Borthick
Date Completed: 9/12/2026
Assignment: CSC734 HW3 
"""
import heapq
from itertools import groupby
from src.normalizer import Normalizer


class Index:
    """An inverted index stored as a DataFrame of terms and linked lists.
    
    Each row contains one term and its linked list. Each document's ID and
    that term's frequency in the document are stored in a linked-list node.
    The linked list stores both the number of documents that mention a term
    and the total frequency of a term across all documents.

    The ``postings`` DataFrame has two columns: ``term`` and ``ids``.
    The ``ids`` column contains ``LinkedList`` objects.

    ### Args:
        **stopfile: str**!
        The name of the file containing the stopwords to remove from 
        terms.  This is the only positional argument.<br>

        **normalizer: Normalizer | class**
        The normalizer class to use for normalizing the terms.
        Defaults to built-in Normalizer.<br>
        Both indexed and query terms must be normalized for querying
        to match on terms.<br>

        **files_to_index: str**
        The name of the directory to search for documents.
        Set this when building an index.<br>

        **loaded_postings: pandas.DataFrame**
        An empty or supplied DataFrame with ``term`` and ``ids`` columns.
        Set this when loading a pickled index.<br>

        **postings_list: LinkedList**
        The list to store document IDs for a term.
        This is not really an argument, but a class field.

    ### Methods:
        **build_index(): Index**
        Generate the postings and lists and return the postings.<br>

        **get_all_postings(): None**
        Display all postings and their lists.<br>

        **get_posting(term): None**
        Display a single term's postings list.<br>

        **get_postings(terms): None**
        Display a selection of term's postings lists.<br>

        **get_top_n(n): list[tuple[str, int]]**
        Return the terms with the top n largest postings lists.
    
    ### Examples:
        `index = Index("dir", "stopwords.txt")`
        `index.build_index()`<br>
        
        `index.get_all_postings()`
        *{term: [(1, 3), (2, 1), (7, 2)]}*<br>

        `print(len(index.get_all_postings()))`
        *1*<br>

        `index.build_index().get_all_postings()`
        *{term: [(1, 3), (2, 1), (7, 2)]}*<br>

        `self.postings[["term", "ids"]]`
    """
    def __init__(self, stopfile, files_to_index=None, loaded_postings=None, normalizer=Normalizer):
        self.stopfile = stopfile
        self.files = files_to_index
        self.postings = loaded_postings or {}
        self.normalizer = normalizer(stopfile)

    def build_index(self):
        """Create the postings DataFrame and its linked lists."""
        def _map_it(filename, terms):
            """Map file text to (term, (docID, 1))."""
            doc_id = int(filename.name.split("_")[1].split(".")[0])
            return list(map(lambda term: (term, (doc_id, 1)), terms))

        def _group_it(sorted_terms):
            """Group sorted, mapped terms by (term, (docID, count in text))."""
            grouped_terms = groupby(sorted_terms, key=lambda item: item[0])
            terms = []
            for _, group in grouped_terms:
                group = list(group)
                freq = len(group)
                term = group[0][0]
                doc_id = group[0][1][0]
                terms.append((term, (doc_id, freq)))
            return terms

        def _reduce_it(term, llist):
            """Add term's docID and count to linked list.
            
            Args:
                **term**: tuple[str, tuple[int, int]]
                The term with its document ID and frequency count.

            Returns:
                LinkedList
                The postings list for the dictionary term.
            """
            doc_freq = term[1]
            llist.add_node(doc_freq)
            return llist

        for file in self.files:
            tokens = self.normalizer.get_tokens(file)
            mapped_terms = _map_it(file, tokens)
            grouped_terms = _group_it(mapped_terms)

            for term in grouped_terms:
                if self.postings.get(term[0], None) is None:
                    self.postings[term[0]] = self.postings_list()
                _reduce_it(term, self.postings[term[0]])

        return self

    def get_top_n(self, n):
        """Return the ``n`` terms with the highest total frequencies."""
        top_postings = heapq.nlargest(n, 
                                      self.postings.items(), 
                                      key=lambda posting: posting[1].total_term_freq)
        return [p[0] for p in top_postings]

    
