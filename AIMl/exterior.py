import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import logging
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix , classification_report
from pathlib import Path

# Path("/my/directory").mkdir(parents=True, exist_ok=True)
# defin constants
DATA_SET_PATHS = 'data-set/exterior'
SLASH = '/'
x_train = []

# Data-set paths
x_data_path = "tranning/data/x_data.npy"
y_dataset_path = "tranning/data/y_dataset.npy"
x_dataset_path = "tranning/data/x_dataset.npy"
x_train_path = "tranning/data/x_train.npy"
y_train_path = "tranning/data/y_train.npy"
x_test_path = "tranning/data/x_test.npy"
y_test_path = "tranning/data/y_test.npy"
MODEL_PATH = '../model/VGG16_new_HL.keras'

LEARNING_RATE = 1e-5
EPOCHS = 200

# create a log
Path("../logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='../logs/aiml.log', filemode='w', level=logging.DEBUG)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')

'''
    Load pre-trained model (e.g., VGG16)
    we can change base model and re-train
    here is the list of some base model (https://keras.io/api/applications)
'''
def loadBaseModel():
    return tf.keras.applications.VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = (128, 128, 3)
    )

# Preprocessing function
def preprocess_image(image, input_width, input_height, num_channels):
    # Resize the image to the desired input shape of your model
    resized_image = cv2.resize(image, (input_width, input_height))
    # Expand dimensions to match the input shape of your model
    expanded_image = resized_image
    # Repeat the image across the channel dimension if it's grayscale
    if num_channels == 1:
        expanded_image = np.repeat(expanded_image, 3, axis=-1)

    return expanded_image

def createAndLoadData():
    # Get a list of all files in the folder
    x_data = {}  # Create an empty dictionary to store the loaded data

    for bucket in os.listdir(DATA_SET_PATHS):
        x_train = []  # Create an empty list to store the preprocessed images
        num_instances = []
        folder_path = DATA_SET_PATHS + SLASH + bucket
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg") or file.endswith(".JPEG"):
                # Read the image
                image_path = os.path.join(folder_path, file)
                image = cv2.imread(image_path)

                # Append the preprocessed image to x_train
                preprocessed_image = preprocess_image(image, 128, 128, 3)
                x_train.append(preprocessed_image)

        # Convert x_train to a numpy array
        x_train = np.array(x_train)
        save_path = "tranning/x_train_" + bucket + ".npy"
        # Save x_train as a NumPy binary file
        np.save(save_path, x_train)
        logging.info(bucket + " data Saved successfully!!")

        # Load the x_train array from the NumPy binary file
        x_data[bucket] = np.load(save_path)
        # Get the number of instances for each class
        num_instances.append(np.load(save_path).shape[0])  # Append the value to the list

        logging.info(bucket + " data Load successfully!!")

        # Create an array with consecutive numbers representing the classes
        y_data = np.repeat(np.arange(len(num_instances)), num_instances)
        y_data = np.array(y_data)

        # Combine the arrays along the first axis (axis=0)
        x_data = np.concatenate(tuple(x_data.values()), axis=0)

        # Save x_data as a NumPy binary file
        np.save(x_data_path, x_data)

        return x_data, y_data

def shuffleData(x_data, y_data):
    # Randomly shuffle the indices
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)

    # Shuffle x_data and Y_data using the shuffled indices
    x_data = x_data[indices]
    y_data = y_data[indices]

    # save x and y dataset
    # Save x_dataset as a NumPy binary file
    np.save(x_dataset_path, x_data)
    # Path to save the y_dataset array
    # Save y_dataset as a NumPy binary file
    np.save(y_dataset_path, y_data)
    logging.info('Data is Suffled.')

    return x_data, y_data

# shuffle and save the final x and y
def prepareData():
    x_data, y_data = createAndLoadData()

    x_data, y_data = shuffleData(x_data, y_data)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

    logging.info('Data is Splited.')

    # After running the above cell run this cell once to normalize the data
    # Normalize the data
    x_train = x_train/255
    x_test = x_test/255

    np.save(x_train_path, x_train)
    x_train = np.load(x_train_path)

    np.save(y_train_path, y_train)
    y_train = np.load(y_train_path)

    np.save(x_test_path, x_test)
    x_test = np.load(x_test_path)

    np.save(y_test_path, y_test)
    y_test = np.load(y_test_path)

    logging.info('Data loaded and saved.')

    return x_train, y_train, x_test, y_test

def testModel(model, x_test, y_test):
    # Evaluate the model on test data using the built-in evaluate function of the keras
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

    y_pred = model.predict(x_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(y_test, y_pred_classes))


def startTrainModel():
    x_train, y_train, x_test, y_test = prepareData()

    base_model = loadBaseModel()

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Create the transfer learning model
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(11, activation='linear')
    ])

    # Compile the model
    # Learning rate can be changed as per requirement
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model using the fit method
    # Number of Epochs can be changed as per requirement
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = EPOCHS)

    # Save the entire model as a `.keras` zip archive. Change the name of the model in the path
    model.save(MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    testModel(model, x_test, y_test)


startTrainModel()
