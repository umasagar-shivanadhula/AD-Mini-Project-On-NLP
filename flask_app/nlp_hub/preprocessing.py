"""Text preprocessing: tokenization, stop-word removal and lemmatization."""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NLTK_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}


def ensure_nltk_data():
    """Download any NLTK data that is missing."""
    for path, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Could not download NLTK data: {package}")


class TextPreprocessor:
    """A pipeline of small steps, each usable on its own."""

    # Penn Treebank tag prefix -> WordNet part of speech
    POS_MAP = {"J": "a", "V": "v", "N": "n", "R": "r"}

    def __init__(self, language="english"):
        ensure_nltk_data()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
        """Lower-case word tokens, with punctuation and numbers dropped."""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.replace("-", "").isalpha()]

    def remove_stop_words(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tagged_tokens):
        """Lemmatize (token, tag) pairs, using the tag as a hint."""
        return [
            self.lemmatizer.lemmatize(token, self.POS_MAP.get(tag[0], "n"))
            for token, tag in tagged_tokens
        ]

    def process(self, text):
        """Full pipeline. Tags are assigned before stop words are removed,
        because the tagger needs the whole sentence for context."""
        tagged = nltk.pos_tag(self.tokenize(text))
        kept = [(t, tag) for t, tag in tagged if t not in self.stop_words]
        return self.lemmatize(kept)
