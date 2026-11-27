"""Zero-shot text classification with a Transformers model."""
from functools import cached_property

from .base import NLPModule


class ClassificationModule(NLPModule):
    name = "classify"
    title = "Text Classification"
    description = "Which topic fits the text best?"
    order = 40

    MODEL = "typeform/distilbert-base-uncased-mnli"
    LABELS = ["technology", "business", "sports", "politics",
              "health", "science", "entertainment", "education"]

    @cached_property
    def _classifier(self):
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=self.MODEL)

    def run(self, text):
        result = self._classifier(text, candidate_labels=self.LABELS)
        scores = [
            {"label": label, "score": round(float(score), 4)}
            for label, score in zip(result["labels"], result["scores"])
        ]
        return {"category": scores[0]["label"], "scores": scores}
