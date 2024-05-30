from os import listdir, makedirs
from shutil import copyfile

import numpy as np
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from random import random
from sklearn.model_selection import train_test_split

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout
from keras._tf_keras.keras.utils import to_categorical

# folder = 'DogCat/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
#     # determine class
#     output = 0.0
#     if file.startswith('dog'):
#         output = 1.0
#     # load image
#     photo = load_img(folder + file, target_size=(200, 200))
#     # convert to numpy array
#     photo = img_to_array(photo)
#     photos.append(photo)
#     labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)

# # create directories
# dataset_home = 'dataset_dogs_vs_cats/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
#     # create label subdirectories
#     labeldirs = ['dogs/', 'cats/']
#     for labldir in labeldirs:
#         newdir = dataset_home + subdir + labldir
#         makedirs(newdir, exist_ok=True)
#
# # define ratio of pictures to use for validation
# val_ratio = 0.25
# # copy training dataset images into subdirectories
# src_directory = 'DogCat/'
# for file in listdir(src_directory):
#     src = src_directory + '/' + file
#     dst_dir = 'train/'
#     if random() < val_ratio:
#         dst_dir = 'test/'
#     if file.startswith('cat'):
#         dst = dataset_home + dst_dir + 'cats/' + file
#         copyfile(src, dst)
#     elif file.startswith('dog'):
#         dst = dataset_home + dst_dir + 'dogs/' + file
#         copyfile(src, dst)
test_photos = []
test_labels = []
train_photos = []
train_labels = []
train_cat_image = 'C:/Users/psxkb5/PycharmProjects/ML_Classification_Dog_vs_Cats/dataset_dogs_vs_cats/train/cats/'
for file in listdir(train_cat_image):
    image = load_img(train_cat_image + '/' + file)
    image = img_to_array(image)
    train_photos.append(image)
    train_labels.append(0)
    print(file)
train_dog_image='C:/Users/psxkb5/PycharmProjects/ML_Classification_Dog_vs_Cats/dataset_dogs_vs_cats/train/dogs/'

for file in listdir(train_dog_image):
    image = load_img(train_dog_image + '/' + file)
    image = img_to_array(image)
    train_photos.append(image)
    train_labels.append(1)
    print(file)

print(train_photos)
print(train_labels)


test_cat_image = 'C:/Users/psxkb5/PycharmProjects/ML_Classification_Dog_vs_Cats/dataset_dogs_vs_cats/test/cats/'
for file in listdir(test_cat_image):
    image = load_img(test_cat_image + '/' + file)
    image = img_to_array(image)
    test_photos.append(image)
    test_labels.append(0)
    print(file)
test_dog_image = 'C:/Users/psxkb5/PycharmProjects/ML_Classification_Dog_vs_Cats/dataset_dogs_vs_cats/test/dogs/'

for file in listdir(test_dog_image):
    image = load_img(test_dog_image + '/' + file)
    image = img_to_array(image)
    test_photos.append(image)
    test_labels.append(1)
    print(file)

train_photos = np.array(train_photos)
test_photos = np.array(test_photos)
print(train_photos.shape)
print(test_photos.shape)

#Normalize the images
train_photos = (train_photos / 255) - 0.5
test_photos = (test_photos / 255) - 0.5
print(train_labels)
print(test_labels)
train_labels=to_categorical(train_labels,2)
test_labels=to_categorical(test_labels,2)

X_train, X_test, Y_train, Y_test = train_test_split(train_photos, train_labels, test_size=0.2, random_state=41)

print(X_train)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
test_loss, test_acc=model.evaluate(X_test,
                                   Y_test)
print(f"Test Accuracy: {test_acc}")
