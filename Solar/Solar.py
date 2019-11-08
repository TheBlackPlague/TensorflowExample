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

# This algorithm is based upon the scaled dot product attention algorithm. ATT(Q, K, V) = softmax_k((Q * K^(T)) / sqrt(dep_k))V

# Import Tensorflow & Tensorflow datasets for DNN & RNN.
import tensorflow as tf
import tensorflow_datasets as tfds

# Import the Operating System.
import os
import zipfile
import re

# Import numpy for arithematic value management.
import numpy as np

# Setting up data: Thank you Cornell University!!
print("Setting up the data.")
zipdir = "F:/Tensorflow/TensorflowExample/Solar/Data/cornell_movie_dialogs.zip"
zipreference = zipfile.ZipFile(zipdir, "r")
zipreference.extractall("C:/tmp/")
pathtomovieline = "C:/tmp/cornell movie-dialogs corpus/movie_lines.txt"
pathtomovieconversation = "C:/tmp/cornell movie-dialogs corpus/movie_conversations.txt"
print("Data set up correctly!")

# Maximum number of samples to pre-process.
maxsample = 50000

# Pre-process a sentence.
def ppsentence(sentence):
    # Better to deal with lower case sentences.
    sentence = sentence.lower().strip()
    # Adding a space between word and punctuation to remove tensor malfunctions.
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # Replacing unnecessary characters with space.
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    # Add a start and end token.
    sentence = sentence.strip()
    return sentence

# Load conversations.
def loadconv():
    # Use of a dictionary function. dict(line) => text
    id2line = {}
    with open(pathtomovieline, errors = "ignore") as file:
        # Read all the lines in "movie_lines.txt".
        lineread = file.readlines()

    for line in lineread:
        # Organize the lines read in an array.
        partarray = line.replace("\n", "").split(" +++$+++ ")
        # Handle that by setting an ID for each line.
        id2line[partarray[0]] = partarray[4]

    # Define our input (questions) and output (answers).
    inputarray, outputarray = [], []
    with open(pathtomovieconversation, "r") as file:
        # Read all the lines in "movie_conversations.txt".
        lineread = file.readlines()

    for line in lineread:
        # Organize the lines read in an array.
        partarray = line.replace("\n", "").split(" +++$+++ ")
        # Get conv in a list.
        conv = [line[1:-1] for line in partarray[3][1:-1].split(", ")]
        # Add to the input & output depending on the lenght of the conversation.
        for i in range(len(conv) - 1):
            # Add (append: add after last record).
            inputarray.append(ppsentence(id2line[conv[i]]))
            outputarray.append(ppsentence(id2line[conv[i + 1]]))
            # In case we're handling more samples than we need to.
            if len(inputarray) > maxsample:
                return inputarray, outputarray

    return inputarray, outputarray

# Set our questions and answers.
print("Defining the set of questions & answers.")
questionarray, answerarray = loadconv()
print("Set defined!")

# Build a tokenizer.
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questionarray + answerarray, target_vocab_size = 2**13)

# Define the start and end token to indicate beginning and end of sentence.
starttoken, endtoken = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary Size (and our start & end tokens).
vocabsize = tokenizer.vocab_size + 2

# Maximum sentence lenght.
maxlenght = 40

# Tokenize our sentence, then filter them.
def tokandfilter(i, o):
    # Define our tokenized input and output.
    toki, toko = [], []
    for (s1, s2) in zip(i, o):
        # Tokenize.
        s1 = starttoken + tokenizer.encode(s1) + endtoken
        s2 = starttoken + tokenizer.encode(s2) + endtoken
        # Check lenght.
        if len(s1) <= maxlenght and len(s2) <= maxlenght:
            # Add (append: add after last record).
            toki.append(s1)
            toko.append(s2)

    # Pad tokenized sentences.
    toki = tf.keras.preprocessing.sequence.pad_sequences(
        toki,
        maxlen = maxlenght,
        padding = "post"
        )
    toko = tf.keras.preprocessing.sequence.pad_sequences(
        toko,
        maxlen = maxlenght,
        padding = "post"
        )
    # Return the tokenized input and output.
    return toki, toko

# Set the questions and answers (after tokenized).
print("Tokenizing the set.")
questionarray, answerarray = tokandfilter(questionarray, answerarray)
print("Set tokenized!")

# Setting a batch size.
batchsize = 64
# Setting a buffer size.
buffersize = 20000

# Setting up the dataset. Decoder inputs use prev. target as input.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        "inputs": questionarray,
        "dec_inputs": answerarray[:, :-1]
    },
    {
        "outputs": answerarray[:, 1:]
    },
    ))
dataset = dataset.cache()
dataset = dataset.shuffle(buffersize)
dataset = dataset.batch(batchsize)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Scaled Dot Product Attention. (Code provided by Google)
def scaled_dot_product_attention(query, key, value, mask):
    # Calculate the attention weights.
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

# Multi-headed attention. (Code provided by Google)
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

# Padding Mask. (Code provided by Google)
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

# Look-ahead mask. (Code provided by Google)
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

# Positional Encoding. (Code provided by Google)
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Encoder Layer. (Code provided by Google)
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

# Encoder. (Code provided by Google)
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

# Decoder Layer. (Code provided by Google)
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

# Decoder. (Code provided by Google)
def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

# Transformer. (Code provided by Google)
def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

# Clearing any backend keras sessions.
tf.keras.backend.clear_session()

# Setting up model. (type: transformer)
print("Setting up model...")
model = transformer(
    vocab_size = vocabsize,
    num_layers = 2,
    units = 512,
    d_model = 256,
    num_heads = 8,
    dropout = 0.1
    )

# Defining the specific loss function. Thanks to Unity Community for helping me make any sense out of this.
def lossfunction(y_true, y_pred):
    y_true = tf.reshape(
        y_true, 
        shape = (-1, maxlenght - 1)
        )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, 
        reduction = "none"
        )(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

# Adam Optimizer Custom Learning Rate Scheduler. (Code provided by Google)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Defining the learning rate.
learningrate = CustomSchedule(256)

# Setting the optimizer.
optimizer = tf.keras.optimizers.Adam(
    learningrate,
    beta_1 = 0.9,
    beta_2 = 0.98,
    epsilon = 1e-9
    )

# Define the accuracy testing for the model.
def accuracy(y_true, y_pred):
    y_true = tf.reshape(
        y_true,
        shape = (-1, maxlenght - 1)
        )
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

print("Model setup completed!")

# Compiling the model.
print("Compiling the model...")
model.compile(
    optimizer = optimizer,
    loss = lossfunction,
    metrics = [accuracy]
    )
print("Model compiled!")

# Model summarry.
model.summary()

zipdir = "F:/Tensorflow/TensorflowExample/Solar/Data/solar-model.zip"
if os.path.exists(zipdir) == True:
    print("Found model! No need to train again.")
    zipreference = zipfile.ZipFile(zipdir, "r")
    zipreference.extractall("C:/tmp/solar-model/")
    # Loading model
    print("Loading model...")
    model.load_weights("C:/tmp/solar-model/tmp/solar-model/solar")
    print("Model loaded!")

if os.path.exists(zipdir) == False:
    # Training the model.
    print("Beginning model training...")
    model.fit(dataset, epochs = 20)
    print("Model trained succesfully!")
    # Saving model.
    print("Saving model for future uses...")
    model.save_weights("C:/tmp/solar-model/solar", save_format = "tf")
    zipreference = zipfile.ZipFile(zipdir, "w")
    zipreference.write("C:/tmp/solar-model/checkpoint", compress_type = zipfile.ZIP_DEFLATED)
    zipreference.write("C:/tmp/solar-model/solar.data-00000-of-00002", compress_type = zipfile.ZIP_DEFLATED)
    zipreference.write("C:/tmp/solar-model/solar.data-00001-of-00002", compress_type = zipfile.ZIP_DEFLATED)
    zipreference.write("C:/tmp/solar-model/solar.index", compress_type = zipfile.ZIP_DEFLATED)
    zipreference.close()
    print("Model saved succesfully!")

def evaluate(sentence):
    # Pre-process the sentence.
    sentence = ppsentence(sentence)
    sentence = tf.expand_dims(starttoken + tokenizer.encode(sentence) + endtoken, axis = 0)
    output = tf.expand_dims(starttoken, 0)
    for i in range(maxlenght):
        # Predict.
        predictions = model(inputs = [sentence, output], training = False)
        # Select the last word form the Seq_Len dimension.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis = -1), tf.int32)
        # Return if predicted ID == End token.
        if tf.equal(predicted_id, endtoken[0]):
            break

        # Concatenate the predicted ID to the output which is then given to the decoder as input.
        output = tf.concat([output, predicted_id], axis = -1)
    
    return tf.squeeze(output, axis = 0)

def predict(sentence):
    # Define our prediction.
    prediction = evaluate(sentence)
    # Decode the output by using it in the decoder as input.
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    # Output the result.
    # print('You: {}'.format(sentence))
    print('Solar: {}'.format(predicted_sentence))
    # Return in case I want to use it later on.
    return predicted_sentence;

# Define our sample input which we use on WHILE loop for continuous chat..
userinput = ""

# Define a while loop to allow continuous chat.
while userinput != "0x001":
    # Take Input.
    userinput = input("You: ")
    # Send Output.
    predict(userinput)