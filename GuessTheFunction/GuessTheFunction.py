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

# Import Tensorflow (for Neural Network) and Numpy (for Value management).
import tensorflow as tf
import numpy as np
# Import Keras for models and database.
from tensorflow import keras

# Create a NNM (Neural Network Model).
# Since it'll identify polynomial functions such as: Y = MX + C, we need only one variable so we'll take only one.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile with pre-built optimizer.
# sgd = Stochastic Gradient Descent
model.compile(optimizer="sgd", loss="mean_squared_error")

# Function we're evaluating.
fx = "Y(X) = X - 1"

# Provide the dataset. Function used: Y = X - 1
# You can change this to whatever function you want and it'll evaluate that.
xvalue = np.array([
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    ], dtype=float)
yvalue = np.array([
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
    ], dtype=float)

# Train the NNM.
model.fit(xvalue, yvalue, epochs=500)
print("Model trained succesfully!")

# Create blank lines.
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")

# Let the user define the value.
print(fx + ": X = ")
value = float(input())

# Model Predicted Value.
print("Evaluating...")
prediction = model.predict([value])[0][0]

# Print the result.
print("Y(" + str(value) + ") = " + str(int(round(prediction))))

# Wait for input before exiting.
print("Press any key to exit.")
exit = input()