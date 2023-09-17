import os
from sklearn.model_selection import train_test_split
from keras.applications import VGG16, ResNet50  # Replace with desired model
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import cv2
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

# Set the data directory
DATA_SET_PATHS = 'data-set/interior'

# Set up the data generator
image_size = (128, 128)  # Set the desired target size
batch_size = 32

# Set up the class names directly
class_names = ["Rear_View_Mirror", "Rear_Seat_Covers", "Power_Windows", "Front_Seat_Covers", "Floor_Mats", "Central_Lock", "Dashboard", "Door_Panel"]

# Load and split the data
x_train, x_val, y_train, y_val = [], [], [], []
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_SET_PATHS, class_name)
    images = os.listdir(class_dir)
    train_images, val_images = train_test_split(images, test_size=0.1, random_state=42)

    for image_name in train_images:
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        x_train.append(image)
        y_train.append(class_index)

    for image_name in val_images:
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        x_val.append(image)
        y_val.append(class_index)

# Convert the data to numpy arrays and normalize
x_train = np.array(x_train) / 255.
y_train = np.array(y_train)
x_val = np.array(x_val) / 255.
y_val = np.array(y_val)

# Load the desired pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))  # Replace with desired model

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 7
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_val, y_val)
)
