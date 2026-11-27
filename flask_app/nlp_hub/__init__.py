"""Next-Gen NLP Hub: Flask application factory."""
from flask import Flask

from .modules import load_modules
from .routes import create_blueprint


def create_app(modules=None):
    """Build the app. Tests can pass their own modules."""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request limit
    app.json.sort_keys = False
    if modules is None:
        modules = load_modules()
    app.register_blueprint(create_blueprint(modules))
    return app
