import tensorflow as tf

import time

import numpy
import PIL.Image as Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



IMAGE_SHAPE = (224, 224)
buildings = ["Education Building", "Engineering Building", "Pollack Library North"]
model_path = "./model_new/buildings_augmented"



reloaded = tf.keras.models.load_model(model_path)



while True:
    photo_number = input("Photo # [1-6]: ")
    building = Image.open("testdata/" + photo_number + ".jpg").resize(IMAGE_SHAPE)
    building = numpy.array(building)/255.0

    result = reloaded.predict(building[numpy.newaxis, ...])

    prediction = numpy.argmax(result[0], axis=-1)

    print (buildings[prediction] + " with " + str(result[0][prediction]) + " certainty.")