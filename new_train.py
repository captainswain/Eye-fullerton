# EyeFullerton Model Training
# This code is modified from google tensorflows documentation found below:
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tensorflow2_image_retraining.ipynb

import os

import tensorflow
import tensorflow_hub as hub



IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 


# Directory containing dataset
data_dir = "/Users/slindsay/Documents/Code/Model-dataset-training/dataset"


# Args for flow_from directory and ImageDataGenerator
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

# Generate batches of tensor image data with real-time data augmentation.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2,
    **datagen_kwargs)





model = tensorflow.keras.Sequential([
    # Wrap mobilenet_v2 Hub modul as a Keras Layer.
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4", trainable=True),
    # Apply dropout to the layer to combat overfitting
    tensorflow.keras.layers.Dropout(rate=0.2),
    tensorflow.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                          kernel_regularizer=tensorflow.keras.regularizers.l2(0.0001))
])

model.build((None,)+IMAGE_SIZE+(3,))
model.summary()


## training the model
model.compile(
  optimizer=tensorflow.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tensorflow.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
  metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit_generator(
    train_generator,
    epochs=8, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history

# save model to model_new folder
tensorflow.saved_model.save(model, "./model_new/buildings_augmented")
