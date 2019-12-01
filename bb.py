import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import sys

import time

import PIL.Image as Image




IMAGE_SHAPE = (224, 224)
buildings = ["Education Building", "Engineering Building", "Pollack Library North"]

loaded = tf.saved_model.load("./model_new/buildings")
infer = loaded.signatures["serving_default"]
# print (dir(infer))


# print(infer.structured_outputs)

# print(callable(imported))
building = Image.open("testdata/" + sys.argv[1] + ".jpg").resize(IMAGE_SHAPE)
building = np.array(building)/255.0


print(loaded(building)).numpy()[0])
# result = imported.predict(building[np.newaxis, ...])

# prediction = np.argmax(result[0], axis=-1)

# print (buildings[prediction] + " with " + str(result[0][prediction]) + " certainty.")