# Next-Gen NLP Hub (Flask)

A Flask web app with an HTML, CSS and JavaScript frontend.

| Feature | How it works |
|---|---|
| Sentiment analysis | TextBlob polarity and subjectivity |
| Summarization | Transformers, `sshleifer/distilbart-cnn-12-6` |
| Keyword extraction | NLTK preprocessing, then term frequency |
| Text classification | Transformers zero-shot, `typeform/distilbert-base-uncased-mnli`, 8 topics |

## Run it (Windows PowerShell)

```powershell
cd flask_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The first run needs internet: NLTK data is
downloaded at startup, and each Transformers model on its first use.

On macOS or Linux, activate with `source .venv/bin/activate`.

## Test

```
python -m pytest tests
```

## Structure

```
app.py                    entry point
nlp_hub/routes.py         one generic route: POST /api/<name>
nlp_hub/preprocessing.py  tokenize, remove stop words, lemmatize
nlp_hub/modules/          one class per feature, discovered automatically
templates/, static/       frontend
```

## Add a feature

Create one file in `nlp_hub/modules/`; the route and tab appear automatically.

```python
from .base import NLPModule

class WordCountModule(NLPModule):
    name = "wordcount"
    title = "Word Count"
    description = "Counts the words."

    def run(self, text):
        return {"words": len(text.split())}
```
