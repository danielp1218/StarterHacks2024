from flask import Flask, jsonify, request

import os
import pathlib

from werkzeug.datastructures import FileStorage
import uuid

app = Flask(__name__)

def save_images(files: list[FileStorage]):
    unique = str(uuid.uuid4())
    p = pathlib.Path(__file__).parent / f"{unique}/"
    p.mkdir(parents=True, exist_ok=True)
    for file in files:
        sub = p / ((file.filename.split('-')[0] if file.filename else 'test') or 'test')
        sub.mkdir(parents=True, exist_ok=True)
        # filename = f'{unique}.{file.filename and file.filename.split(".")[1:] or "png"}'
        file.save(sub / (file.filename or f'{str(uuid.uuid4())}.png'))
    return unique


@app.route("/images", methods=['POST'])
def hello_world():
    # file format: label-#.ext
    training_images = request.files
    foldername = save_images(list(training_images.values()))
    return jsonify({'status': 'ok', 'folder': foldername})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
