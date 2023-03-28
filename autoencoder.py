import pandas as pd
import numpy as np
import urlextract
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.api.keras.preprocessing import image
from keras.applications.convnext import preprocess_input
from keras.datasets import cifar10, cifar100
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from urllib.request import urlopen
from urllib3.util import url

from PIL import Image

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))


# Функция генерации автокодировщика
def create_deep_conv_ae():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


# Генерируем модели
d_encoder, d_decoder, d_autoencoder = create_deep_conv_ae()
d_autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
d_autoencoder.load_weights("weights/cifar100_weights(new).hdf5")

# weights_file = "cifar100_weights(new).hdf5"  # Имя файла в который сохранются веса
#
# # Чекпоинты для сохранения в процессе обучения
# checkpoint = ModelCheckpoint(weights_file, monitor='acc', mode='max', save_best_only=True, verbose=1)
# d_autoencoder.fit(x_train, x_train,
#                   epochs=50,
#                   batch_size=64,
#                   callbacks=[checkpoint],
#                   shuffle=True,
#                   validation_data=(x_test, x_test)
#                   )


# Функция подготовки изображения для обработки
def get_img(img):
    x = image.img_to_array(img)
    new_img = np.reshape(x, (32, 32, 3))
    new_img = np.expand_dims(new_img, axis=0)
    new_img = preprocess_input(new_img)
    return new_img


def get_pca_metrics(img):
    features = [str(x + 1) for x in range(8)]  # Названия колонок для датафрейма

    # Перевод numpy массива в стандартный
    array = []
    for i in range(len(img[0])):
        for e in img[0][i]:
            array += [[e[j] for j in range(len(e))]]
    new_arr = [[el for el in array[0]], [el for el in array[1]], [el for el in array[2]], [el for el in array[3]]]

    # Создание pandas dataframe
    df = pd.DataFrame(data=new_arr, columns=list('12345678'))
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # Стандартизация данных и переход к 2 числам
    pca = PCA(n_components=2)
    principal_comps = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_comps, columns=['1', '2'])
    return principal_df


def square_rooted(num):
    return round(sqrt(sum([a * a for a in num])), 3)


# Функция сравнения косинусного расстояния между 2 векторами
def get_cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


# Получение вектора из pd dataframe
def get_vector(dictionary, index):
    x, y = 0, 0
    for key, value in dictionary.items():
        if key == "1":
            for k, v in dictionary[key].items():
                if k == index:
                    x = v
        elif key == "2":
            for k, v in dictionary[key].items():
                if k == index:
                    y = v
    return [x, y]


# Функция для получения коэфицента похожести 2 изображений
def compare_images(img1, img2):
    image1 = get_img(img1)
    image2 = get_img(img2)

    encoded_img1 = d_encoder.predict(image1)
    encoded_img2 = d_encoder.predict(image2)

    arr = get_pca_metrics(encoded_img1).to_dict()
    arr2 = get_pca_metrics(encoded_img2).to_dict()

    comparer = 0
    # Среднее арифмитическое 4 метрик
    for i in range(4):
        first_vector = get_vector(arr, i)
        second_vector = get_vector(arr2, i)
        comparer += abs(get_cosine_similarity(first_vector, second_vector))
    return comparer / 4  # >= metrica


# Функция открытия изображения по ссылке или пути в файловой системе
def open_img(url_path):
    extractor = urlextract.URLExtract()
    urls = extractor.find_urls(url_path)

    if len(urls) == 0:
        img = Image.open(url_path)
    else:
        img = Image.open(urlopen(url_path)).convert("RGB")

    r, g, b = img.split()
    r = r.point(lambda i: i * 1.2)
    g = g.point(lambda i: i * 0.9)
    res_img = Image.merge('RGB', (r, g, b))

    return res_img.resize((32, 32))


# Функция для нахождения ключа по значению в словаре
def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k


# Функция для нахождения максимально похожего изображения из 2 массива
def get_picture_similarity(pic_arr):
    result_dict = []
    for url_dict in pic_arr:
        maximum = max(url_dict.values())
        best_sim = get_key(url_dict, maximum)
        result_dict += [best_sim + " " + str(maximum)]

    return result_dict


def get_similarities(img_arr1, img_arr2):
    container = []
    for url1 in img_arr1:
        img1 = open_img(url1)
        pred = {}
        for url2 in img_arr2:
            img2 = open_img(url2)
            pred[str(url1) + " - " + str(url2)] = compare_images(img1, img2)
        container.append(pred)
    return container


# Функция для сортировки словаря (прямая, обратная)
def sort_dict(dictionary, param):
    sorted_dict = {}
    sorted_values = sorted(dictionary.values())

    for i in sorted_values:
        for k in dictionary.keys():
            if dictionary[k] == i:
                sorted_dict[k] = dictionary[k]

    if param == "direct":
        return sorted_dict
    elif param == "reverse":
        copied_dict = {}
        for key in reversed(sorted_dict.copy()):
            copied_dict[key] = sorted_dict.get(key)
        return copied_dict

