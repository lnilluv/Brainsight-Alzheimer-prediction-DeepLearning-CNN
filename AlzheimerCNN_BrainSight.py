import os
import pathlib
import random
from datetime import datetime
from distutils.dir_util import copy_tree, remove_tree
from pathlib import Path
from random import randint

import boto3
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import splitfolders
import tensorflow as tf
import tensorflow_addons as tfa
from botocore.client import Config
from imblearn.over_sampling import SMOTE
from keras.utils.vis_utils import plot_model
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight
from tensorflow.keras import Input, Sequential, regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input, MaxPool2D,
                                     SeparableConv2D, concatenate)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

s3 = boto3.resource('s3',
                    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

# test upload file to minio
s3.Bucket('ipynb-py').upload_file(Path(__file__).name, str(datetime.now()) + Path(__file__).name)

# Set your variables for your environment
EXPERIMENT_NAME = "deep-learning-cnn-custom"

# Set tracking URI to your Heroku application
mlflow.set_tracking_uri("https://mlflow.brainsight.tech")

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.tensorflow.autolog()

with mlflow.start_run(experiment_id=experiment.experiment_id):

    data_root_orig = tf.keras.utils.get_file(origin='https://brain-disease-detector.s3.eu-west-3.amazonaws.com/alzheimer_no_split_folders_renamed.zip',
                                            fname='/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/alzheimer_no_split_folders_renamed.zip',
                                            cache_subdir="/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content",
                                            archive_format='zip',
                                            extract=True)
    class_names = os.listdir('/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/alzheimer_no_split_folders_renamed')
    splitfolders.ratio('/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/alzheimer_no_split_folders_renamed', output='/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/output_alzheimer_no_split_folders_renamed_v1', ratio=(0.64, 0.16, 0.20))
    image_dir_path = '.'
    paths = [path.parts[-3:] for path in
            Path('/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/output_alzheimer_no_split_folders_renamed_v1').rglob('*.jpg')]
    df = pd.DataFrame(data=paths, columns=['folder', 'class', 'file_name'])
    print(df.head(10))
    print(df.tail(10))
    df.groupby(['folder', 'class']).size()
    df[df["folder"]=="train"].groupby(["class"]).size()/len(df[df["folder"]=="train"])
    df[df["folder"]=="val"].groupby(["class"]).size()/len(df[df["folder"]=="val"])
    df[df["folder"]=="test"].groupby(["class"]).size()/len(df[df["folder"]=="test"])
    #define image dataset 


    #train_image_generator = ImageDataGenerator(rescale=1/255, brightness_range=[0.5,1.5])
    train_image_generator = ImageDataGenerator(rescale=1/255)

    val_image_generator = ImageDataGenerator(rescale=1/255)

    test_image_generator = ImageDataGenerator(rescale=1/255)


    train_dataset = train_image_generator.flow_from_directory(batch_size=8,
                                                    directory='/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/output_alzheimer_no_split_folders_renamed_v1/train',
                                                    shuffle=True,
                                                    target_size=(176,208), 
                                                    class_mode='categorical')

    validation_dataset = val_image_generator.flow_from_directory(batch_size=8,
                                                    directory='/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/output_alzheimer_no_split_folders_renamed_v1/val',
                                                    shuffle=True,
                                                    target_size=(176,208), 
                                                    class_mode='categorical')

    test_dataset = test_image_generator.flow_from_directory(batch_size=1300,
                                                    directory='/mnt/c/Users/Laure/Documents/Dev/ProjetOpenBrain/content/output_alzheimer_no_split_folders_renamed_v1/test',
                                                    shuffle=True,
                                                    target_size=(176,208))
    train_images, train_labels = train_dataset.next()
    validation_images, validation_labels = validation_dataset.next()
    test_images, test_labels = test_dataset.next()
    train_dataset.class_indices
    test_dataset.class_indices
    
    model = tf.keras.Sequential([

        Conv2D(filters = 16, kernel_size = (3,3), strides = 1, padding = "same",
                activation = "relu", input_shape = (176, 208, 3)), # the input shape (height, width, channels)
        MaxPool2D(pool_size=2, # the size of the pooling window
                strides=2), # the movement of the pooling on the input
        Dropout(0.1),
        Conv2D(filters = 32, kernel_size = (3,3), strides = 1, padding = "same",
                activation = "relu"),
        MaxPool2D(2,2),
        Dropout(0.1),
        Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = "same",
                activation = "relu"),
        MaxPool2D(2,2),
        Dropout(0.1),
        tf.keras.layers.Flatten(), # this layer turns multi-dimensional images into flat objects
        tf.keras.layers.Dense(128, activation="relu"),
        Dropout(0.1),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax")
        ]
    )

    model.summary()
    # Defining metrics and optimization

    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc'),
        tfa.metrics.F1Score(num_classes=4, average='macro', name='f1_score')
        ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3,
                                                restore_best_weights=False)
    initial_learning_rate = 0.0005

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = METRICS)
    # tester d'autres optimizers ex SGD
    #optimizer=tfa.optimizers.LazyAdam(0.001)
    #optimizer=tf.keras.optimizers.Adam()
    model.summary()

    EPOCHS = 50

    history = model.fit(train_dataset, validation_data=validation_dataset,shuffle=True, epochs=EPOCHS)
    train_dataset.class_indices.values()
    type(train_dataset.class_indices.values())
    # Evaluation on validation dataset
    _ = model.evaluate(validation_dataset)
    # Evaluation on test dataset
    _ = model.evaluate(test_dataset)
    test_images, test_labels = test_dataset.next()
    test_dataset.class_indices.keys()
    test_labels
    len(test_labels)
    #Predicting the test data
    predicted_labels = model.predict(test_dataset)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    target_names = [k + ' : ' + str(v) for k,v in test_dataset.class_indices.items()]
    print(classification_report(test_labels, predicted_labels, target_names = target_names))
    #Saving the model for future use
    custom_cnn_model_dir = './models/' + "alzheimer_cnn_dropout_layers_adam.h5"
    model.save(custom_cnn_model_dir, save_format='h5')
    os.listdir('./models')