import os
import tempfile
from flask import Flask, request, jsonify
import whisper

app = Flask(__name__)
model = whisper.load_model("base")  # or "small", "medium", etc.

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Create a temporary file safely
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        try:
            temp_audio.write(file.read())
            temp_audio.flush()
            temp_filename = temp_audio.name

            # Optional: check if file is valid using ffmpeg
            import subprocess
            try:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", temp_filename, "-f", "null", "-"],
                    check=True
                )
            except subprocess.CalledProcessError:
                return jsonify({"error": "Invalid audio file"}), 400

            # Transcribe using Whisper
            result = model.transcribe(temp_filename)
            return jsonify({"text": result["text"]})

        finally:
            # Always remove the temp file
            os.remove(temp_filename)


if __name__ == "__main__":
    app.run(debug=True)
