import whisper
from flask import Flask, request, jsonify

app = Flask(__name__)

model = None  # lazy load

def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny.en", device="cpu")
    return model

@app.route("/stt", methods=["POST"])
def stt():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    audio = request.files["file"]
    audio.save("audio.wav")

    m = get_model()
    result = m.transcribe("audio.wav")
    text = result["text"].lower()

    return jsonify({"text": text})

@app.route("/")
def health():
    return "Whisper server running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
