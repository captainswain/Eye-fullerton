#  EyeFullerton Image Classification Microservice
# supported operations:
#       Upload and classify image

from flask import Flask, request, url_for, redirect, flash
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
from werkzeug.utils import secure_filename
import numpy as numpy
import os

    

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

IMAGE_SHAPE = (224, 224)

# list of buildings
buildings = ["Education Building", "Engineering Building", "Pollack Library North"]
model_path = "./model_new/buildings_augmented"

# Load stored model
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)

# This file check is from flask documentation
def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Start of routes

@app.route('/', methods=['GET'])
def home():
    return {'text': 'Eye Fullerton'}



@app.route('/upload', methods=['POST'])
def post():

    imagefile = request.files.get('building', '')

    if not is_allowed_file(imagefile.filename):
        return {'error' : 'Invalid'}, 404

    filename = secure_filename(imagefile.filename)

    imagefile.save(os.path.join('./uploads', filename))

    uploadedImage = os.path.join('./uploads', filename)

    buildingImage = Image.open(uploadedImage).resize(IMAGE_SHAPE)
    
    buildingImage = numpy.array(buildingImage)/255.0

    result = model.predict(buildingImage[numpy.newaxis, ...])

    # get prediction by the indice of the maximum values along an axis.
    prediction = numpy.argmax(result[0], axis=-1)

    # Return prediction, accuracy, and building name
    return {'id': str(prediction), 'accuracy' : str(result[0][prediction]), 'building' : buildings[prediction] }, 200






if __name__ == "__main__":
    # start Api on port 1337
    app.run(debug=True, host="0.0.0.0", port=1337)