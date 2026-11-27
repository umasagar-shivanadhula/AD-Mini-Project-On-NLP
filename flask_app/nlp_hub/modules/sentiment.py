"""Sentiment analysis with TextBlob."""
from textblob import TextBlob

from .base import NLPModule


class SentimentModule(NLPModule):
    name = "sentiment"
    title = "Sentiment Analysis"
    description = "Is the tone positive, negative or neutral? (English)"
    order = 10

    THRESHOLD = 0.1  # polarity within +/- this counts as neutral

    def run(self, text):
        score = TextBlob(text).sentiment
        if score.polarity > self.THRESHOLD:
            label = "Positive"
        elif score.polarity < -self.THRESHOLD:
            label = "Negative"
        else:
            label = "Neutral"
        return {
            "label": label,
            "polarity": round(score.polarity, 3),
            "subjectivity": round(score.subjectivity, 3),
        }
