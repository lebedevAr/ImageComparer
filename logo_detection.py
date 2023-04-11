import os
import random
from PIL import Image
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from tqdm import tqdm

from keras.utils import load_img, img_to_array
from livelossplot.inputs.keras import PlotLossesCallback
import numpy as np


class LogoDetector:
    def __init__(self, weights_path):
        self.weights = weights_path
        self.BATCH_SIZE = 64
        self.train_dir = ""
        self.test_dir = ""
        self.train_gen = ImageDataGenerator
        self.valid_gen = ImageDataGenerator
        self.test_gen = ImageDataGenerator

    @staticmethod
    def augment_images(dataset_path, mode="train"):
        def add_noise(img):
            deviation = 10 * random.random()
            noise = np.random.normal(0, deviation, img.shape)
            img += noise
            np.clip(img, 0., 255.)
            return img

        categories = os.listdir('{}/{}'.format(dataset_path, mode))
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')

        for category in tqdm(categories, total=len(categories), desc='Augmenting Images'):
            num_images = len(os.listdir('{}/{}/{}'.format(dataset_path, mode, category)))
            if num_images < 800:
                images_to_augment = os.listdir('{}/{}/{}'.format(dataset_path, mode, category))
                num_augments_per_image = (800 - num_images) / num_images
                if num_augments_per_image == 0:
                    images_to_augment = np.random.choice(images_to_augment, (800 - num_images))
                    num_augments_per_image = 1
                for image_path in images_to_augment:
                    image = load_img('{}/{}/{}/{}'.format(dataset_path, mode, category, image_path))
                    x = img_to_array(image)
                    x = x.reshape((1,) + x.shape)
                    x = add_noise(x)
                    i = 0
                    for _ in datagen.flow(x, batch_size=1,
                                          save_to_dir='{}/{}/{}'.format(dataset_path, mode, category),
                                          save_prefix=image_path[:-4], save_format='jpg'):
                        i += 1
                        if i > num_augments_per_image:
                            break
        print('Done augmenting images!')

    def learning_data_generate(self):
        train_generator = ImageDataGenerator(rotation_range=90,
                                             brightness_range=[0.1, 0.7],
                                             width_shift_range=0.5,
                                             height_shift_range=0.5,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             validation_split=0.15,
                                             preprocessing_function=preprocess_input)

        test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        class_subset = sorted(os.listdir(self.train_dir))

        self.train_gen = train_generator.flow_from_directory(self.train_dir,
                                                             target_size=(224, 224),
                                                             class_mode='categorical',
                                                             classes=class_subset,
                                                             subset='training',
                                                             batch_size=self.BATCH_SIZE,
                                                             shuffle=True,
                                                             seed=42)

        self.valid_gen = train_generator.flow_from_directory(self.train_dir,
                                                             target_size=(224, 224),
                                                             class_mode='categorical',
                                                             classes=class_subset,
                                                             subset='validation',
                                                             batch_size=self.BATCH_SIZE,
                                                             shuffle=True,
                                                             seed=42)

        self.test_gen = test_generator.flow_from_directory(self.test_dir,
                                                           target_size=(224, 224),
                                                           class_mode=None,
                                                           classes=class_subset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           seed=42)

    @staticmethod
    def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
        conv_base = VGG16(include_top=False,
                          weights='imagenet',
                          input_shape=input_shape)

        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        model = Model(inputs=conv_base.input, outputs=output_layer)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_vgg(self, n_epochs=20, weights_filename="vgg_improved"):
        n_steps = self.train_gen.samples // self.BATCH_SIZE
        n_val_steps = self.valid_gen.samples // self.BATCH_SIZE

        vgg_model = self.create_model((224, 224, 3), 2, Adam(learning_rate=0.001), fine_tune=0)

        plot_loss_1 = PlotLossesCallback()

        tl_checkpoint_1 = ModelCheckpoint(filepath=weights_filename,
                                          save_best_only=True,
                                          verbose=1)

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   mode='min')

        vgg_model.fit(self.train_gen,
                      batch_size=self.BATCH_SIZE,
                      epochs=n_epochs,
                      validation_data=self.valid_gen,
                      steps_per_epoch=n_steps,
                      validation_steps=n_val_steps,
                      callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                      verbose=1)

    @staticmethod
    def prepro_img(img_path):
        img_obj = load_img(img_path, target_size=(224, 224, 3))
        numpy_img = img_to_array(img_obj)
        image_batch = np.expand_dims(numpy_img, axis=0)
        processed_image = preprocess_input(image_batch.copy())
        return processed_image

    def get_prediction(self, img_path):
        vgg_model = self.create_model((224, 224, 3), 2, Adam(learning_rate=0.001), fine_tune=0)
        vgg_model.load_weights(self.weights)
        img = self.prepro_img(img_path)
        vgg_improved_pred = vgg_model.predict(img)
        vgg_class = np.argmax(vgg_improved_pred, axis=1)
        image = Image.open(img_path)
        image.show()
        return "Picture contains logo" if int(vgg_class[0]) == 0 else "Picture not contains logo"


if __name__ == "__main__":
    sber_detector = LogoDetector("weights/tl_model_v1.weights.best.hdf5")
    print(sber_detector.get_prediction("test4.jpg"))
