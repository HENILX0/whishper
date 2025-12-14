import whisper
from flask import Flask, request, jsonify

app = Flask(__name__)

# Whisper model load (smallest & fastest)
model = whisper.load_model("tiny", device="cpu")

@app.route("/stt", methods=["POST"])
def stt():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    audio = request.files["file"]
    audio.save("audio.wav")

    result = model.transcribe("audio.wav")
    text = result["text"].lower()

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
