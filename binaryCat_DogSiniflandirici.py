# -*- coding: utf-8 -*-
"""

Conv2D katmanı ile binary (Cat/Dog) sınıflandırıcı

https://github.com/fchollet/deep-learning-with-python-notebooks/
"""
import os #operation system
#The directory where we will store our smaller dataset
base_dir='E:\WORKSPACE\deep_cat_dog\kagglecatsanddogs_3367a\Pet'
#Directories for our training
#validation and test splits
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

#Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir,'cats')
validation_cats_dir = os.path.join(validation_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
validation_dogs_dir = os.path.join(validation_dir,'dogs')
test_cats_dir = os.path.join(test_dir,'cats')
test_dogs_dir = os.path.join(test_dir,'dogs')

print('total training cat images:',len(os.listdir(train_cats_dir)))#kullandığım klasörüm içerisinde ne olduğunu görmek için
print('total training dog images:',len(os.listdir(train_dogs_dir)))
print('total validation cat images:',len(os.listdir(validation_cats_dir)))
print('total validation dog images:',len(os.listdir(validation_dogs_dir)))

from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3),
                        padding='valid',
                        activation='relu',
                        input_shape=(150,150,3))) #3 kanal
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),
                        activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3),
                        activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3),
                        activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,
                       activation='relu'))
model.add(layers.Dense(1,
                       activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=5e-4),
              metrics=['acc'])

#data augmentation = veri zenginleştirme
from keras.preprocessing.image import ImageDataGenerator
#All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255) #0-1 arasında

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    #This is the target directory
    train_dir,
    #All images will be resized to 150x150
    target_size=(150,150),
    #No. of images to be yielded from the generator per ba
    batch_size=20,
    #Since we use binary_crossentropy loss
    #we need binary labels
    class_mode='binary')

validation_gennerator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')
#for data_batch,labels_batch in train_generator:
    #print('data batch shape:',data_batch.shape)
    #print('labels batch shape:',labels_batch.shape)
    #break
#
#import matplotlib.pyplot as plt
#for data_batch, labels_batch in train_generator:
    #print('data batch shape:', data_batch.shape)
    #print('labels batch shape:', labels_batch.shape)
    #plt.imshow(data_batch[0,:,:,:])
    #break

history = model.fit( #data augmentation = veri zenginleştirme
    train_generator,
    steps_per_epoch=100, #20*100=2000 kere çalışırı
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#Eğitilmiş modeli kaydet
model.save('cats_and_dogs_small_1.h5')

#
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()











