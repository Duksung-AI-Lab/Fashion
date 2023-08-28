import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import densenet
from keras.callbacks import EarlyStopping

# import tensorflow as tf
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # 텐서플로가 세 번째 GPU만 사용하도록 제한
#     try:
#         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#     except RuntimeError as e:
#         # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
#         print(e)

import tensorflow as tf

# tf.disable_v2_behavior()
np.random.seed(0)
tf.set_random_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = '/home/csa4608/python/Washing/'
image_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   fill_mode='nearest')

# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(os.path.join(data_path, 'train'), shuffle=True,
                                                    target_size=image_size, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
val_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'val'), shuffle=True,
                                                target_size=image_size, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
test_generator = test_datagen.flow_from_directory(os.path.join(data_path, 'test'), shuffle=True,
                                                 target_size=image_size, class_mode='categorical')

transfer_model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size + (3,))

transfer_model.trainable = False
model = models.Sequential()
model.add(transfer_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=optimizers.Adam(0.0001),  # Very low learning rate
              loss='mean_squared_error',
              metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
epochs = 20
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // epochs,
                              callbacks=[early_stopping])

# history = model.fit_generator(train_generator,
#                               epochs=epochs,
#                               steps_per_epoch=50,
#                               validation_data=val_generator,
#                               validation_steps=100)

print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="blue", label='Val_acc')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.ylim([0.5, 1.1])
plt.title("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

plt.figure()

# plt.plot(x_len, y_loss, marker='.', c="red", label='Train_loss')
# plt.plot(x_len, y_vloss, marker='.', c="blue", label='Val_loss')
#
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss/acc')
# plt.ylim([0.0, 1.0])
# plt.title('Training and Validation Loss')

plt.show()

plt.savefig(os.path.join('/hdd_ext/hdd1/lhy0807/models', 'densenet_freeze'))


# Unfreeze the base model
transfer_model.trainable = True

# # 기본 모델에 몇 개의 층이 있는지 확인 합니다.
# print("Number of layers in the base model: ", len(transfer_model.layers))
#
# # 해당 층 이후부터 미세 조정
# fine_tune_at = 120
#
# # `fine_tune_at` 층 이전의 모든 층을 고정
# for layer in transfer_model.layers[:fine_tune_at]:
#   layer.trainable =  False

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=optimizers.Adam(lr=1e-5),  # Very low learning rate
              loss='mean_squared_error',
              metrics=['acc'])

# Train end-to-end. Be careful to stop before you overfit!
# history = model.fit(train_generator, epochs=30, validation_data=val_generator)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // epochs,
                              callbacks=[early_stopping])

print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="blue", label='Val_acc')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.ylim([0.5, 1.1])
plt.title("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

plt.figure()

# plt.plot(x_len, y_loss, marker='.', c="red", label='Train_loss')
# plt.plot(x_len, y_vloss, marker='.', c="blue", label='Val_loss')
#
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss/acc')
# plt.ylim([0.0, 1.0])
# plt.title('Training and Validation Loss')

plt.show()

plt.savefig(os.path.join('/hdd_ext/hdd1/lhy0807/models', 'densenet_unfreeze'))

model.save(os.path.join('/hdd_ext/hdd1/lhy0807/models', 'models/transfer_densenet.h5'))