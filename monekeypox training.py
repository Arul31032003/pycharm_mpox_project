from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Data augmentation
train_datagen1 = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,        # Randomly rotate images in the range (degrees)
    width_shift_range=0.2,    # Randomly translate images horizontally (fraction of total width)
    height_shift_range=0.2,   # Randomly translate images vertically (fraction of total height)
    shear_range=0.2,          # Shear angle in counter-clockwise direction (degrees)
    zoom_range=0.2,           # Randomly zoom into images
    horizontal_flip=True,     # Randomly flip images
    fill_mode='nearest'       # Filling in new pixels
)
train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('Fold1/Fold1/Fold1/Fold1/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary',subset="training")
test_set = test_datagen.flow_from_directory('Fold1/Fold1/Fold1/Fold1/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')
val_set = valid_datagen.flow_from_directory('Fold1/Fold1/Fold1/Fold1/Val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')
train_y = training_set.classes
test_y = test_set.classes
val_y = val_set.classes

training_set.class_indices
train_y.shape, test_y.shape, val_y.shape

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class VGG_16:
    def __init__(self):
        super(VGG_16, self).__init__()

        # Adding layers to the model
        self.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # Optionally add flattening and fully connected layers for classification
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(10, activation='softmax'))  # Change 10 to the number of classes in your problem

    def add(self, param):
        pass


vgg16 = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in vgg16.layers:
    layer.trainable = False

x = Flatten()(vgg16.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
prediction = Dense(1, activation='sigmoid')(x)

model3 = Model(inputs=vgg16.input, outputs=prediction)

# view the structure of the model
model3.summary()

model3.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

#Early stopping to avoid overfitting of model
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
# fit the model
history2 = model3.fit(
    training_set,
    validation_data=val_set,
    epochs=50,
    callbacks=[early_stop],
    batch_size=32, shuffle=True)

# accuracies

plt.plot(history2.history['accuracy'], label='train acc')
plt.plot(history2.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# loss
plt.plot(history2.history['loss'], label='train loss')
plt.plot(history2.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Predict on test set
test_set.reset()  # Reset the test set for prediction
predictions = model3.predict(test_set)
predictions = (predictions > 0.5).astype(int)  # Convert
# Create confusion matrix
cm = confusion_matrix(test_y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
# Save the trained model
model3.save('model.h5')

from sklearn.metrics import precision_score, recall_score

