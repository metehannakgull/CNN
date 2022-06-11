import numpy as np 
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir='C:/Users/cvanAkgul/Desktop/F/src/data/train'
val_dir='C:/Users/cvanAkgul/Desktop/F/src/data/test'
train_datagen=ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)
validation_generator=val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

emotion_model=Sequential()
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
emotion_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25)) 
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7,activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])
emotion_model.summary()
'''
emotion_model_info=emotion_model.fit(
    train_generator,
    steps_per_epoch=28709//64,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=7178//64
)
'''

history =emotion_model.fit(train_generator,
                           epochs=50, 
                           batch_size=64,
                           validation_data=validation_generator)

print(history.history['loss'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['val_accuracy'])
epochs = range(1,51)
plt.plot(epochs, history.history['loss'], 'g', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
history =emotion_model.fit(train_generator,
                           epochs=50, 
                           batch_size=64,
                           validation_data=validation_generator)

print(history.history['loss'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['val_accuracy'])
epochs = range(1,51)
plt.plot(epochs, history.history['accuracy'], 'g', label='Training acc')
plt.plot(epochs, history.history['val_accuracy'], 'b', label='validation acc')
plt.title('Training and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend() 
plt.show()
'''