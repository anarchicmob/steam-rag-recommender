import os
from typing import Any

from flask import Flask, jsonify, render_template, request
from recommender import create_search_engine


def create_app() -> Flask:
    app = Flask(__name__)
    search_engine = create_search_engine()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/search")
    def search() -> Any:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "A game description is required."}), 400
        try:
            return jsonify(search_engine.search(query))
        except Exception as exc:
            return jsonify({"error": "Search failed.", "details": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
