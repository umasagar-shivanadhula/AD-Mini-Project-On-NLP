"""Entry point: python app.py"""
from nlp_hub import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
