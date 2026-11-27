"""HTTP routes. One generic route serves every registered module."""
from flask import Blueprint, current_app, jsonify, render_template, request

MAX_CHARS = 20_000


def create_blueprint(modules):
    bp = Blueprint("nlp_hub", __name__)

    def error(message, status):
        return jsonify({"error": message}), status

    @bp.get("/")
    def index():
        return render_template("index.html", modules=modules.values())

    @bp.get("/api/modules")
    def list_modules():
        return jsonify([
            {"name": m.name, "title": m.title, "description": m.description}
            for m in modules.values()
        ])

    @bp.post("/api/<name>")
    def analyse(name):
        module = modules.get(name)
        if module is None:
            return error(f"Unknown feature: {name}", 404)

        body = request.get_json(silent=True)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return error("Please enter some text.", 400)
        if len(text) > MAX_CHARS:
            return error(f"Text is too long (limit {MAX_CHARS:,} characters).", 413)

        try:
            return jsonify(module.run(text.strip()))
        except ValueError as exc:
            return error(str(exc), 400)
        except Exception:
            current_app.logger.exception("Module %s failed", name)
            return error("Analysis failed. Check the server log for details.", 500)

    return bp
