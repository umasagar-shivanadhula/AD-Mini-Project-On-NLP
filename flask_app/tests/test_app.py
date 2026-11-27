import pytest

from nlp_hub import create_app
from nlp_hub.modules import load_modules
from nlp_hub.modules.base import NLPModule
from nlp_hub.preprocessing import TextPreprocessor


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_all_four_modules_are_discovered():
    assert list(load_modules()) == ["sentiment", "summarize", "keywords", "classify"]


def test_preprocessing_pipeline():
    result = TextPreprocessor().process("The cats are running in the gardens.")
    assert result == ["cat", "run", "garden"]


@pytest.mark.parametrize("text,label", [
    ("I love this, it is wonderful.", "Positive"),
    ("This is terrible and awful.", "Negative"),
    ("The box is on the table.", "Neutral"),
])
def test_sentiment(client, text, label):
    response = client.post("/api/sentiment", json={"text": text})
    assert response.get_json()["label"] == label


def test_keywords(client):
    response = client.post("/api/keywords", json={"text": "Python code. Python tests. Code."})
    terms = [k["term"] for k in response.get_json()["keywords"]]
    assert terms[:2] == ["python", "code"]


@pytest.mark.parametrize("body", [None, [], {}, {"text": "   "}, {"text": 5}])
def test_bad_input_is_rejected(client, body):
    assert client.post("/api/sentiment", json=body).status_code == 400


def test_limits_and_unknown_feature(client):
    assert client.post("/api/sentiment", json={"text": "a" * 20_001}).status_code == 413
    assert client.post("/api/nothing", json={"text": "hi"}).status_code == 404
    assert client.post("/api/summarize", json={"text": "too short"}).status_code == 400


def test_new_module_needs_no_route_changes():
    class WordCount(NLPModule):
        name, title, description = "count", "Word Count", "Counts words."

        def run(self, text):
            return {"words": len(text.split())}

    app = create_app({"count": WordCount()})
    assert app.test_client().post("/api/count", json={"text": "a b c"}).get_json() == {"words": 3}
    assert b"Word Count" in app.test_client().get("/").data


# The two Transformers features are tested with stand-in models, so no download is needed.
def test_summarization_response():
    torch = pytest.importorskip("torch")
    module = load_modules()["summarize"]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.ones((1, 80), dtype=torch.long)}

        def decode(self, ids, **kwargs):
            return "A short summary."

    class Model:
        def generate(self, **kwargs):
            assert kwargs["min_length"] < kwargs["max_length"] <= 150
            return [[1, 2, 3]]

    module.__dict__["_model"] = (Tokenizer(), Model())
    result = module.run("word " * 60)
    assert result == {"summary": "A short summary.", "original_words": 60,
                      "summary_words": 3, "truncated": False}


def test_classification_response():
    module = load_modules()["classify"]
    module.__dict__["_classifier"] = lambda text, candidate_labels: {
        "labels": ["science", "sports"], "scores": [0.9, 0.1]}
    result = module.run("A new telescope.")
    assert result["category"] == "science"
    assert result["scores"][1] == {"label": "sports", "score": 0.1}
