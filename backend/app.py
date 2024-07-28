from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO

import os
import pathlib

from werkzeug.datastructures import FileStorage
import uuid
from PIL import Image

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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

import zipfile
import io

# @socketio.on('dataset')
@app.route('/dataset', methods=['POST'])
# types ??
def handle_dataset():
    files = {}
    file = request.files['file']
    with zipfile.ZipFile(file.stream, 'r') as input_zip:
        input_zip.extractall(pathlib.Path(__file__).parent / 'data')

    return 'ok'

        # files = {name: input_zip.read(name) for name in input_zip.namelist()}

    # return list(files.items())[0]
    # file_data = data['data']
    # file_name = data['name']

# RECEIVED AS ZIP
@app.route("/images", methods=['POST'])
def hello_world():
    # file format: label-#.ext
    training_images = request.files
    foldername = save_images(list(training_images.values()))
    return jsonify({'status': 'ok', 'folder': foldername})

# @app.route("/train", methods=['POST'])
@socketio.on('train')
def train(data: dict):
    from util import run
    import json
    print(1)
    run(data)
    print(2)
    return 'ok'
    # send accuracy, graph?, ...
    # return jsonify({'status': 'ok'})

@app.route("/test", methods=['POST'])
def test():
    from util import test_image
    file = request.files['file']    
    img = Image.open(file.stream)
    res = test_image(img)
    response = jsonify({'res': res})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@socketio.on('client_connected')
def handle_client_connect_event(data):
    print("connected")
    print(str(data))    
@socketio.on('disconnect')
def disconnected():
    print('disconnected')
@socketio.on('connect')
def connected():
    print('connected')


if __name__ == "__main__":
    socketio.run(app, use_reloader=True, log_output=True, debug=True, port=int(os.environ.get("PORT", 5000)))
