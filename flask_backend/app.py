import os

from flask import Flask, request, redirect, url_for, render_template, jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io
from diffusionModel import TestModel

# Configs
STATIC_FOLDER = '../vue_frontend/dist/_static'
TEMPLATE_FODER = '../vue_frontend/dist'
UPLOAD_IMG_PATH   = os.path.join(os.path.dirname(__file__), 'upload_imgs')
if not os.path.exists(UPLOAD_IMG_PATH):
    os.mkdir(UPLOAD_IMG_PATH)
UPLOAD_IMG_NAME   = 'up-img.png'
MASK_IMG_NAME     = 'mask-img.png'
EDITED_MASK_NAME  = 'edited_mask.png'
DOWNLOAD_IMG_NAME = 'down-img.png'

# start APP
app = Flask(__name__,
            static_folder=STATIC_FOLDER,
            template_folder=TEMPLATE_FODER)

# init Model
Model = TestModel(root_path=UPLOAD_IMG_PATH)

# API functions
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# upload img
@app.route('/upload', methods=['POST'])
def upload_img():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        img.save(os.path.join(UPLOAD_IMG_PATH, UPLOAD_IMG_NAME), 'PNG')
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_IMG_PATH, filename)

@app.route('/up-image-url', methods=['GET'])
def send_up_img_url():
    if request.method == 'GET':
        filename = UPLOAD_IMG_NAME
        if os.path.exists(os.path.join(UPLOAD_IMG_PATH, filename)):
            url = url_for('uploaded_file', filename=filename)
            info = {
                'url': url,
                'id': 0,
            }
            print(f">>> {info}")
            return jsonify(info)
        else:
            info = {
                'url': '',
                'id': 0,
            }
            return jsonify(info), 404

@app.route('/down-image-url', methods=['GET'])
def send_down_img_url():
    if request.method == 'GET':
        filename = DOWNLOAD_IMG_NAME
        if os.path.exists(os.path.join(UPLOAD_IMG_PATH, filename)):
            url = url_for('uploaded_file', filename=filename)
            info = {
                'url': url,
                'id': Model.get_final_id(),
            }
            return jsonify(info)
        else:
            info = {
                'url': '',
                'id': 0,
            }
            return jsonify(info), 404

@app.route('/mask-url', methods=['GET'])
def send_mask_img_url():
    if request.method == 'GET':
        filename = MASK_IMG_NAME
        if os.path.exists(os.path.join(UPLOAD_IMG_PATH, filename)):
            url = url_for('uploaded_file', filename=filename)
            info = {
                'url': url,
                'id': Model.get_mask_id(),
            }
            return jsonify(info)
        else:
            info = {
                'url': '',
                'id': 0,
            }
            return jsonify(info), 404
        
@app.route('/gen-image', methods=['POST'])
def gen_image():
    if request.method == 'POST':
        data = request.get_json()
        img_type = data.get('type')
        if img_type == 'final':
            image_data = data.get('image')
            image_data = image_data.split(',')[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_data))
            img_path = os.path.join(UPLOAD_IMG_PATH, EDITED_MASK_NAME)
            image.save(img_path, 'PNG')
            # call diffusion model to generate image
            final_id = Model.gen_final(EDITED_MASK_NAME, DOWNLOAD_IMG_NAME)
            return f'{final_id}', 200
        elif img_type == 'mask':
            # call diffusion model to generate mask
            mask_id = Model.gen_mask(UPLOAD_IMG_NAME, MASK_IMG_NAME)
            return f'{mask_id}', 200
        
@app.route('/clean', methods=['POST'])
def clean_file():
    print('remove temporal files...')
    file_list = os.listdir(UPLOAD_IMG_PATH)
    for name in file_list:
        if name.split('.')[-1] == 'png':
            os.remove(os.path.join(UPLOAD_IMG_PATH, name))
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989,debug=True)