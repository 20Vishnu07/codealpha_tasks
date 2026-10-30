from pathlib import Path
from flask import Flask, jsonify, render_template, request
from chatbot import FAQBot

app = Flask(__name__)
bot = FAQBot(Path(__file__).with_name("faqs.json"))

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/ask")
def ask():
    msg = (request.get_json(silent=True) or {}).get("message", "")
    return jsonify(bot.ask(msg))

if __name__ == "__main__":
    app.run(debug=True)
