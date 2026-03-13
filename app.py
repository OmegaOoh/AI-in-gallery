import os
from typing import Optional

import torch
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image  # ty:ignore[unresolved-import]
from transformers import BlipForConditionalGeneration, BlipProcessor
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size


model: Optional[BlipForConditionalGeneration] = None
processor: Optional[BlipProcessor] = None
device: str = "cpu"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def load_model():
    global model, processor, device
    if model is None or processor is None:
        # Load BLIP model and processor locally
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)  # ty:ignore[invalid-argument-type]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_albums():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    albums = []
    for item in os.listdir(app.config["UPLOAD_FOLDER"]):
        item_path = os.path.join(app.config["UPLOAD_FOLDER"], item)
        if os.path.isdir(item_path):
            albums.append(item)
    return sorted(albums)


@app.route("/")
def index():
    albums = get_albums()
    if not albums:
        # Create a default album if none exists
        os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], "default"), exist_ok=True)
        albums = ["default"]
    return render_template("index.html", albums=albums)


@app.route("/album/<album_name>")
def view_album(album_name):
    safe_album_name = secure_filename(album_name)
    album_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_album_name)

    if not os.path.exists(album_path) or not os.path.isdir(album_path):
        return redirect(url_for("index"))

    images = []
    for filename in os.listdir(album_path):
        if allowed_file(filename):
            caption = None
            caption_filepath = os.path.join(album_path, filename + ".txt")
            if os.path.exists(caption_filepath):
                try:
                    with open(caption_filepath, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                except Exception:
                    pass
            images.append({"name": filename, "caption": caption})

    images.sort(key=lambda x: x["name"])
    return render_template("album.html", album_name=safe_album_name, images=images)


@app.route("/create_album", methods=["POST"])
def create_album():
    album_name = request.form.get("album_name")
    if album_name:
        safe_album_name = secure_filename(album_name)
        if safe_album_name:
            album_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_album_name)
            os.makedirs(album_path, exist_ok=True)
    return redirect(url_for("index"))


@app.route("/upload/<album_name>", methods=["POST"])
def upload_file(album_name):
    safe_album_name = secure_filename(album_name)
    album_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_album_name)

    if not os.path.exists(album_path) or not os.path.isdir(album_path):
        return redirect(url_for("index"))

    if "file" not in request.files:
        return redirect(url_for("view_album", album_name=safe_album_name))

    files = request.files.getlist("file")

    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(album_path, filename))

    return redirect(url_for("view_album", album_name=safe_album_name))


@app.route("/caption/<album_name>/<filename>")
def get_caption(album_name, filename):
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500

    safe_album_name = secure_filename(album_name)
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_album_name, safe_filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    caption_filepath = filepath + ".txt"
    if os.path.exists(caption_filepath):
        try:
            with open(caption_filepath, "r", encoding="utf-8") as f:
                return jsonify({"caption": f.read().strip()})
        except Exception as _:
            pass

    try:
        raw_image = Image.open(filepath).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_new_tokens=50)
        caption_text = processor.decode(out[0], skip_special_tokens=True).capitalize()

        try:
            with open(caption_filepath, "w", encoding="utf-8") as f:
                f.write(caption_text)
        except Exception as _:
            pass

        return jsonify({"caption": caption_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
