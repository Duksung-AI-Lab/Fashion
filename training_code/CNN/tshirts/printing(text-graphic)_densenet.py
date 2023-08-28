import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import densenet
from keras.callbacks import ModelCheckpoint,EarlyStopping

np.random.seed(0)
tf.set_random_seed(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

data_path = "셔츠 원본 데이터셋"

image_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(os.path.join(data_path, 'train'), shuffle=True,
                                                    target_size=image_size, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
val_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'val'), shuffle=True,
                                                target_size=image_size, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
test_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'test'), shuffle=True,
                                                 target_size=image_size, class_mode='categorical')

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

transfer_model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size + (3,))
transfer_model.trainable = False
transfer_model.summary()

model = models.Sequential()
model.add(transfer_model)
model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.2))  # Regularize with dropout
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0002),
              metrics=['acc'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=7)
epochs = 30
history = model.fit_generator(train_generator,
                              epochs=21,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=train_generator.samples // epochs)
                              # callbacks=[early_stopping])

# 테스트 결과
print("\n Valid Accuracy: %.4f" % (model.evaluate_generator(val_generator)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

# 그래프
acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))
plt.ylim([0.5, 1])
plt.legend()

plt.figure()

plt.plot(epochs, y_loss, 'go', label='Training Loss')
plt.plot(epochs, y_vloss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.ylim([0, 0.5])
# plt.ylim([0, 0.2])
plt.legend()

plt.show()

# fine tuning
transfer_model.trainable = True

# for layer in transfer_model.layers[:53]:
#     layer.trainable = False

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
epochs = 30
history = model.fit_generator(train_generator,
                              epochs=5,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator)
                              validation_steps=train_generator.samples // epochs,
#                              callbacks=[early_stopping])

# 모델 저장
model.save('graphic text 모델.h5')

# 테스트 결과
print("\n Valid Accuracy: %.4f" % (model.evaluate_generator(val_generator)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

# 그래프
acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))
plt.ylim([0.5, 1])
plt.legend()

plt.figure()

plt.plot(epochs, y_loss, 'go', label='Training Loss')
plt.plot(epochs, y_vloss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.ylim([0, 0.5])
# plt.ylim([0, 0.2])
plt.legend()

plt.show()
