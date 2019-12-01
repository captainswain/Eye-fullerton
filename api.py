#  EyeFullerton Image Classification Microservice
# supported operations:
#       Upload and classify image

from flask import Flask, request, url_for, redirect, flash
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
from werkzeug.utils import secure_filename
import numpy as np
import os

    


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SHAPE = (224, 224)
buildings = ["Education Building", "Engineering Building", "Pollack Library North"]
model_path = "./model_new/buildings_augmented"
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Start of routes

@app.route('/', methods=['GET'])
def home():
    return {'text': 'Eye Fullerton'}



@app.route('/upload', methods=['POST'])
def post():

    imagefile = request.files.get('imagefile', '')

    if not is_allowed_file(imagefile.filename):
        return {'error' : 'Invalid'}, 404

    filename = secure_filename(imagefile.filename)

    imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    uploadedImage = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    buildingImage = Image.open(uploadedImage).resize(IMAGE_SHAPE)
    
    buildingImage = np.array(buildingImage)/255.0

    result = model.predict(buildingImage[np.newaxis, ...])

    prediction = np.argmax(result[0], axis=-1)

    return {'id': str(prediction), 'accuracy' : str(result[0][prediction]), 'building' : buildings[prediction] }, 200






if __name__ == "__main__":
    # Working on a ubuntu VM that isn't accesible on localhost.
    app.run(debug=True, host="0.0.0.0", port=1337)