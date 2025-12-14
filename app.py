from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)

model = whisper.load_model("tiny")  # FASTEST

@app.route("/")
def home():
    return "Whisper server running"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if request.data is None:
        return jsonify({"error": "No audio received"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(request.data)
        temp_filename = f.name

    result = model.transcribe(temp_filename)
    os.remove(temp_filename)

    return jsonify({"text": result["text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
