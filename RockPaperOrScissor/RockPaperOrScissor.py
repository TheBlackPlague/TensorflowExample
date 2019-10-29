# MIT License
#
# Copyright (c) 2019 Shaheryar Sohail
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Import path to ensure we aren't repeating ourselves.
import os.path
# Import wget to download database.
import wget

# Check if database & test database exist. If not, then download them.
if os.path.exists("/tmp/rps.zip") == True:
    print("Found the database zip. Not downloading again. :D")

if os.path.exists("/tmp/rps.zip") == False and os.path.exists("/tmp/rps/") == False:
    # Download the database.
    print("Downloading database...")
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
    wget.download(url, "/tmp/rps.zip")
    print("")

if os.path.exists("/tmp/rps-test-set.zip") == True:
    print("Found the test database zip. Not downloading again. :D")

if os.path.exists("/tmp/rps-test-set.zip") == False and os.path.exists("/tmp/rps-test-set/"):
    # Download the testing database so we can compare.
    print("Downloading test database...")
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip"
    wget.download(url, "/tmp/rps-test-set.zip")
    print("")

# Import modules to extract the zip file and save contents in an organized way.
import os
import zipfile

# Find the databases or extract them.
if os.path.exists("/tmp/rps/"):
    print("Database found!")

if os.path.exists("/tmp/rps/") == False:
    # Zip Extraction and saving.
    print("Extracting database. This may take a couple of seconds.")
    zipdir = "/tmp/rps.zip"
    zipreference = zipfile.ZipFile(zipdir, "r")
    zipreference.extractall("/tmp/")
    print("Extracted database.")
    zipreference.close()

if os.path.exists("/tmp/rps-test-set/"):
    print("Test Database found!")

if os.path.exists("/tmp/rps-test-set/") == False:
    # Zip Extraction and saving.
    print("Extracting test database. This may take a couple of seconds.")
    zipdir = "/tmp/rps-test-set.zip"
    zipreference = zipfile.ZipFile(zipdir, "r")
    zipreference.extractall("/tmp/")
    print("Extracted test database.")
    zipreference.close()

# Save directories as var.
rockdir = os.path.join("/tmp/rps/rock")
paperdir = os.path.join("/tmp/rps/paper")
scissordir = os.path.join("/tmp/rps/scissors")
print("Total Training Rock Images: ", len(os.listdir(rockdir)))
print("Total Training Paper Images: ", len(os.listdir(paperdir)))
print("Total Training Scissor Images: ", len(os.listdir(scissordir)))

# Save files as var.
rockfile = os.listdir(rockdir)
paperfile = os.listdir(paperdir)
scissorfile = os.listdir(scissordir)

# Import Tensorflow (for Neural Network) and Keras_preprocessing (for preprocessing).
import tensorflow as tf
import keras_preprocessing
# Import Image for generating image data.
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# Model has been confirmed to already exist.
if os.path.exists("/tmp/rps.h5") == True:
    print("Model has been found. Not going to train.")
    model = tf.keras.models.load_model("/tmp/rps.h5")
    model.summary()

# Check if model has already been trained & saved.
if os.path.exists("/tmp/rps.h5") == False:
    # Model wasn't found.
    print("The model wasn't found. The training time can take around 3 minutes, and all the way up to 2 hours.")

    # Set our training directory and make an image data generator for training.
    trainingdir = "/tmp/rps/"
    print("Setting training directory to: " + trainingdir)
    trainingdatagenerator = ImageDataGenerator(
        rescale = 1./255, 
        rotation_range = 40, 
        width_shift_range = 0.2, 
        height_shift_range = 0.2, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        fill_mode = "nearest"
        )
    # Set our testing directory and make an image data generator for testing.
    testingdir = "/tmp/rps-test-set/"
    print("Setting training directory to: " + testingdir)
    testingdatagenerator = ImageDataGenerator(rescale = 1./255)
    
    # Have our train generator and test generator be flowing from our training and test directories.
    print("Creating a training generator with flow from: " + trainingdir)
    traingenerator = trainingdatagenerator.flow_from_directory(
        trainingdir, 
        target_size = (150, 150), 
        class_mode = "categorical"
        )
    print("Creating a testing generator with flow from: " + testingdir)
    testgenerator = testingdatagenerator.flow_from_directory(
        testingdir, 
        target_size = (150, 150), 
        class_mode = "categorical"
        )

    # Create a DNNM (Deep Neural Network Model).
    print("Creating a DNNM (Deep Neural Network Model).")
    model = tf.keras.models.Sequential([
        # Convulations for Deep Neural Network (DNN).
        # First convulation.
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation = "relu",
            # Input Shape = (150 x 150) with 3 byte colors.
            input_shape = (150, 150, 3)
            ),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Second convulation.
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation = "relu"
            ),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Third convulation.
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation = "relu"
            ),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Fourth convulation.
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation = "relu"
            ),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results into the DNN.
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # Create a dense hidden layer with 512 neurons.
        tf.keras.layers.Dense(
            512,
            activation = "relu"
            ),
        # Create an output layer having 3 neurons (Rock, Paper, or Scissor).
        tf.keras.layers.Dense(
            3,
            activation = "softmax"
            )
        ])
    model.summary()
    
    # Compile the model.
    print("Compiling model.")
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = "rmsprop",
        metrics = [
        "accuracy"
            ]
        )
    print("Model compiled.")
    
    # Train the model via generators.
    print("Training the model.")
    history = model.fit_generator(
        traingenerator,
        epochs = 25,
        validation_data = testgenerator,
        verbose = 1
        )
    print("Model trained succesfully.")
    
    # Save the model.
    print("Saving the model.")
    saveloc = "/tmp/rps.h5"
    model.save(saveloc)
    print("Model saved to: " + saveloc + " succesfully.")

# Import Numpy (for Value management).
import numpy as np

# Set the testing image. Make sure to change path to location of image.
path = "master:TensorflowExample/RockPaperOrScissor/Data/t1.png"
img = image.load_img(path, target_size = (150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

# Create an array.
setofimage = np.vstack([x])
# Predict what it is.
prediction = model.predict(setofimage, batch_size = 10)[0]

# Output result to user.
if prediction[0] == 1:
    print("Paper!")

if prediction[1] == 1:
    print("Rock!")

if prediction[2] == 1:
    print("Scissor!")

