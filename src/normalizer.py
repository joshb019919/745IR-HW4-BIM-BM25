"""Build, store, load, and parse inverted indexes.

Author: Josh Borthick
Date Completed: 9/12/2026
Assignment: CSC734 HW3 
"""
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import string

class Normalizer:
    """Normalize all words by removing punctuation and stop words, then
    tokenize and stem them.  Uses PorterStemmer by default.

    ### Args:
        **stemmer: class**
        The stemmer object to stem terms.<br>

        **stopfile: str**
        The text file containing words to remove from the terms list.

    ### Methods:
        **_read_file(fname): list[str]**
        Read file fname and return each word in a list of strings.<br>

        **_read_stopwords(fname): list[str]**
        Read the stopwords file and return them in a list of strings.<br>

        **_normalize_terms(file, stops): list[str]**
        Normalize all terms and return them in a list of strings.<br>

        **get_tokens(file): list[str]**
        Read in tokens and return them normalized and in ascending order.
    """

    def __init__(self, stopfile, stemmer=PorterStemmer()):
        self.stemmer = stemmer
        self._punctuation_table = str.maketrans(
            {char: "" for char in string.punctuation + "’‘“”" + "—–…«»" + "‹›"})

        with open(stopfile, encoding="utf-8") as f:
            self.stopwords = {line.strip().lower()
                              for line in f
                              if line.strip()}

    def _read_file(self, fname):
        """Take text, lowercase, strip extra whitespace, make list.
        
        Tries different text encodings (zipped, utf-8, then latin-1).
        
        ### Args:
            **fname: str**
            The dynamically assigned file path to open.

        ### Returns:
            **list[str]**
            Lower-cased, stripped list of words in document.    
        """
        import gzip

        # Try gzip decompress
        try:
            with gzip.open(fname, 'rt', encoding='utf-8') as f:
                return f.read()
        except (OSError, gzip.BadGzipFile):        
            pass

        # Try UTF-8
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # Try Latin-1 (ISO-8859-1) — fallback that accepts all byte values    
        try:
            with open(fname, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Cannot read file {fname}: {e}")

    def _normalize_terms(self, file):
        """Tokenize with NLTK, remove punctuation, and stem.

        Optimized by using translate method to remove punctuation.
        
        Args:
            **file**: """
        stem = self.stemmer.stem        
        stopwords = self.stopwords        
        translate = self._punctuation_table
        tokens = [stem(cleaned, to_lowercase=False)
                for word in word_tokenize(file)
                if (cleaned := word.lower().translate(translate))
                and cleaned not in stopwords]
        return tokens


    def get_tokens(self, file, sort=True, from_file=False):
        """Return the normalized terms in ascending order.
        
        ### Args:
            **file: os.scandir DirEntry[str]**
            The file to read.

        ### Returns:
            **list[str]**
            The sorted, normalized tokens from the document.    
        """
        if from_file:
            tokens = self._normalize_terms(self._read_file(file))
        else:
            tokens = self._normalize_terms(file)
        
        if sort:            
            tokens.sort()
        return tokens

    def has_next(self):
        pass
    