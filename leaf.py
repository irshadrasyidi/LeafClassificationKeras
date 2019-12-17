import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import datetime
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from keras.utils import plot_model
import numpy as np


def train():
    train_dir = 'DatasetV2/Comp/training'
    validation_dir = 'DatasetV2/Comp/validation'

    train_datagen = ImageDataGenerator(rescale=1. / 255,  # normalize images
                                       rotation_range=20,
                                       width_shift_range=0.12,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       zoom_range=0.15,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=20,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=20,
        class_mode='categorical')

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    log_dir = "logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir)

    # Jalankan di cmd untuk membuka Tensorboard
    # activate TFGPU2
    # tensorboard --logdir C:/Users/IRSHAD/PycharmProjects/VISKOM/FPVK/logs/scalars/

    es_callback = EarlyStopping(monitor='val_accuracy',
                                min_delta=0.002,
                                patience=10,
                                mode='max',
                                baseline=None,
                                restore_best_weights=True)

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=7,
                                       verbose=1,
                                       epsilon=1e-4,
                                       mode='auto')

    # model = tf.keras.models.load_model('model1.h5')

    opt = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=126,
        epochs=500,
        callbacks=[tensorboard, es_callback, reduce_lr_loss],
        validation_data=validation_generator,
        validation_steps=24)

    model.save('VKmodel_2.h5')


def test():
    test_dir = 'DatasetV2/Comp/test'
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(200, 200),  # Resizes all images to 200 x 200
        batch_size=20,
        class_mode='categorical')
    model = tf.keras.models.load_model('VKmodel_1.h5')     # PENTING! Kasih nama file model dulu
    print(model.summary())
    print("accuracy:", model.evaluate_generator(test_generator, steps=100)[1])


def test_per_image():
    model = tf.keras.models.load_model('VKmodel_1.h5')

    img = image.load_img('DatasetV2/Comp/test/H/IMG_20191119_163524_BURST11.jpg', target_size=(200, 200, 3))
    img2predict = image.img_to_array(img)
    img2predict = np.expand_dims(img2predict, axis=0)

    images = np.vstack([img2predict])
    result = model.predict(images, batch_size=None)
    print(result)

    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    max_val = result[0][0]
    final_label = label[0]

    for i in range(1, 10):
        if max_val < result[0][i]:
            max_val = result[0][i]
            final_label = label[i]
    print(max_val)
    print("Daun Kelas " + final_label)

    # for i in range(0, 10):
    #     print(result[0][i])

    # print(max_val)


if __name__ == "__main__":
    train()
    test()
    test_per_image()