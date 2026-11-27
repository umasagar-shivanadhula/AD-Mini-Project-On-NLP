"""Keyword extraction: most frequent lemmas after preprocessing."""
from collections import Counter

from ..preprocessing import TextPreprocessor
from .base import NLPModule


class KeywordModule(NLPModule):
    name = "keywords"
    title = "Keyword Extraction"
    description = "The terms that appear most, after removing stop words."
    order = 30

    TOP_N = 10

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or TextPreprocessor()

    def run(self, text):
        terms = [t for t in self.preprocessor.process(text) if len(t) > 2]
        if not terms:
            raise ValueError("No keywords found. Try a longer piece of text.")
        return {
            "keywords": [
                {"term": term, "count": count}
                for term, count in Counter(terms).most_common(self.TOP_N)
            ]
        }
