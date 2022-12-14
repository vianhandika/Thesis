from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import zipfile
import pickle
import math
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)

# df = pd.read_csv('test5.csv')
exclude_columns = ['src_ip', 'dst_ip', 'src_port', 'src_mac', 'dst_mac', 'timestamp']
# columns = [c for c in df.columns if c not in exclude_columns]
# X_train, y_train = df[columns].values, np.array(list(range(10))*((len(df)//10)+1))[:len(df)]

# clf = DecisionTreeClassifier(max_depth =3, random_state = 42)
# clf.fit(X_train, y_train)
df = pd.read_csv('vocabulary.csv',index_col=0)
vocab_list = df['0'].to_list()

df = pd.read_csv('CATEGORICAL_FEATURE_NAMES.csv',index_col=0)
CATEGORICAL_FEATURE_NAMES = df['0'].to_list()
NUMERIC_FEATURE_NAMES = []
TARGET_COLUMN_NAME = 'label'
numtarget = 2


# print(os.listdir("./"))
def create_model_inputs():
    inputs = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.string
        )

    return inputs

def create_embedding_encoder(size=None):
    inputs = create_model_inputs()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(pd.concat([df_test])[feature_name].unique())]
            )
            # vocabulary = sorted(
            #     [str(value) for value in vocab_list]
            # )
            # vocabulary = vocab_list
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = lookup(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            vocabulary_size = len(vocabulary)
            embedding_size = int(math.sqrt(vocabulary_size))
            feature_encoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = feature_encoder(value_index)
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features_concat = layers.concatenate(encoded_features, axis=1)
    # Apply dropout.
    encoded_features = layers.Dropout(rate=0.5)(encoded_features_concat)
    # # Perform non-linearity projection.
    encoded_features = layers.Dense(
        units=encoded_features_concat.shape[-1], activation="relu"
    # )(encoded_features_concat)
    )(encoded_features)
    # encoded_features = layers.Concatenate()([encoded_features, encoded_features_concat])
    encoded_features = layers.Dense(units=size if size else encoded_features_concat.shape[-1]//2, activation="linear", kernel_initializer="he_normal")(encoded_features)
    # encoded_features = layers.LayerNormalization()(encoded_features)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)

from gc import freeze

def pddataframe(dataframe, label = None, batch_size = 1000) -> tf.data.Dataset:
  
  # Make sure that missing values for string columns are not represented as
  # float(NaN).
  for col in dataframe.columns:
    if dataframe[col].dtype in [str, object]:
      dataframe[col] = dataframe[col].fillna("")

  if label is not None:
    features_dataframe = dataframe.drop(label, 1)
    output = (dict(features_dataframe), tf.keras.utils.to_categorical(dataframe[label].values))
    tf_dataset = tf.data.Dataset.from_tensor_slices(output)

  # The batch size does not impact the training of TF-DF.
  if batch_size is not None:
    tf_dataset = tf_dataset.batch(batch_size)

  # Seems to provide a small (measured as ~4% on a 32k rows dataset) speed-up.
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
  return tf_dataset

# numtarget = len(y.value_counts())

def create_nn_model(encoder):
    inputs = create_model_inputs()
    embeddings = encoder(inputs)
    if numtarget>2:
        final_layer = layers.Dense(units=numtarget, activation="softmax")
    else:
        final_layer = layers.Dense(units=numtarget, activation="sigmoid")
    # final_layer.trainable = False
    output = final_layer(embeddings)

    nn_model = keras.Model(inputs=inputs, outputs=output)
    
    if numtarget>2:
        nn_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy("categorical_accuracy")],
        )
    else:
        nn_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy("categorical_accuracy")],
        )
    return nn_model

# print("FINISH ---------------------------")

df_test = pd.read_csv('test6.csv')
# columns = [c for c in df_test.columns if c not in exclude_columns]
df_test.drop(exclude_columns, inplace=True, axis=1)
df_test['label'] = 0


# df_train = pd.DataFrame(X_train, columns=X.columns)
# df_test = pd.DataFrame(X_test, columns=X.columns)
# df_train['label'] = y_train 
# df_test['label'] = y_test 

for c in CATEGORICAL_FEATURE_NAMES:
    # df_train[c] = df_train[c].astype(str)
    print(c)
    df_test[c] = df_test[c].astype(str)

# train_dataset = pddataframe(df_train, label=TARGET_COLUMN_NAME)
test_dataset = pddataframe(df_test, label=TARGET_COLUMN_NAME)

embedding_encoder = create_embedding_encoder(size=8)
model = create_nn_model(embedding_encoder)
model.load_weights('best_embedding_model_bin2.h5')

# X_train_embedded = pd.DataFrame(embedding_encoder.predict(train_dataset))
X_test_embedded = pd.DataFrame(embedding_encoder.predict(test_dataset))
# X_test_embedded

loaded_model = pickle.load(open('DT_Model_Default_bin2.sav', 'rb'))
y_pred = loaded_model.predict(X_test_embedded)

print(y_pred)