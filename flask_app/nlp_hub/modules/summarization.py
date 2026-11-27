"""Abstractive summarization with a Transformers model."""
from functools import cached_property

from .base import NLPModule


class SummarizationModule(NLPModule):
    name = "summarize"
    title = "Summarization"
    description = "A short summary of a longer passage (at least 40 words)."
    order = 20

    MODEL = "sshleifer/distilbart-cnn-12-6"
    MIN_WORDS = 40
    MAX_INPUT_TOKENS = 1024

    @cached_property
    def _model(self):
        """Loaded on first use so the server starts quickly."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        return (AutoTokenizer.from_pretrained(self.MODEL),
                AutoModelForSeq2SeqLM.from_pretrained(self.MODEL))

    def run(self, text):
        words = len(text.split())
        if words < self.MIN_WORDS:
            raise ValueError(f"Enter at least {self.MIN_WORDS} words to summarize.")

        tokenizer, model = self._model
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=self.MAX_INPUT_TOKENS)
        input_length = inputs["input_ids"].shape[1]
        max_length = min(150, max(30, input_length // 2))
        output = model.generate(**inputs, max_length=max_length,
                                min_length=min(20, max_length // 2),
                                num_beams=4, do_sample=False)
        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return {
            "summary": summary,
            "original_words": words,
            "summary_words": len(summary.split()),
            "truncated": input_length >= self.MAX_INPUT_TOKENS,
        }
