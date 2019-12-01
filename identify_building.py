import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
from tensorflow.keras import layers
import sys

import time

import numpy as np
import PIL.Image as Image


if len(sys.argv) != 2:
    exit("Incorrect arguments. correct usage python identify_building.py [1-4]")


IMAGE_SHAPE = (224, 224)
buildings = ["Education Building", "Engineering Building", "Pollack Library North"]
model_path = "./model_new/buildings_augmented"



reloaded = tf.keras.models.load_model(model_path)



building = Image.open("testdata/" + sys.argv[1] + ".jpg").resize(IMAGE_SHAPE)
building = np.array(building)/255.0

result = reloaded.predict(building[np.newaxis, ...])

prediction = np.argmax(result[0], axis=-1)

print (buildings[prediction] + " with " + str(result[0][prediction]) + " certainty.")