!pip install keras_nlp
import keras_nlp
import tensorflow as tf
from tensorflow import keras
import time


preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)


import tensorflow_datasets as tfds

import os
import json
import csv

songs = []
fieldNames = ['title', 'tag', 'artist', 'year', 'views', 'features', 'lyrics', 'id', 'language_cld3', 'language_ft', 'language']
with open('../song_lyrics.csv', newline='') as f:
    reader = csv.DictReader(f, fieldnames = fieldNames)
    csv_headings = next(reader)
    for i in range(50):
      songs.append(next(reader)['lyrics'])
    

paragraphs = songs


train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)


output = gpt2_lm.generate("Hummingbirds", max_length=200)
print(output)
