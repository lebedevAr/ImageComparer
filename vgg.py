import os
import re
from urllib.request import urlopen
import httplib2

import pandas as pd
import urlextract
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.api.keras.preprocessing import image
from keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np

base_model = VGG16(weights='imagenet')  # , input_shape=(224, 224, 3)

feat_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)


def prepro(img_obj):
    numpy_img = img_to_array(img_obj)
    image_batch = np.expand_dims(numpy_img, axis=0)
    processed_image = preprocess_input(image_batch.copy())
    return processed_image


def open_img_with_path(img_path, imgs_model_width=224, imgs_model_height=224):
    img = load_img(img_path, target_size=(imgs_model_width, imgs_model_height))
    preprocessed_img = prepro(img)
    return preprocessed_img


def save_image(img_url_list, site_num):
    for i, url in enumerate(img_url_list):
        h = httplib2.Http('.cache')
        responce, content = h.request(url)
        out = open(rf'/home/artyom/PycharmProjects/ImageComparer/test_imgs/site{site_num}/{i}img.jpg', 'wb')
        out.write(content)
        out.close()
    return "done"


def get_cosine_similarity(images):
    images = [open_img_with_path(img) for img in images]
    np_img_array = np.vstack(images)
    image_features = feat_extractor.predict(np_img_array)
    image_features.shape
    cosine_similarities = cosine_similarity(image_features)
    cos_similarities_df = pd.DataFrame(cosine_similarities)
    cos_similarities_df.head()
    return cosine_similarities


def get_hamming_similarity(image_arr):
    pass


def convert_np_to_list(sim_arr):
    res = [np_arr.tolist() for np_arr in sim_arr]
    for arr in res:
        arr.pop(0)
    for new_arr in res:
        for double in new_arr:
            double = round(double, 3)
    return res


def get_max_similarity(similarity_arr, dir1, arr1, arr2):
    res = []
    for i in range(len(os.listdir(dir1))):
        pic_url = arr1[i]
        similarity = max(similarity_arr[i])
        index = similarity_arr[i].index(similarity)
        pic_url2 = arr2[index]
        res.append(f"{pic_url} <b>similar to</b> {pic_url2} <b>with similarity</b> {similarity}")
    return res


def compare_images(url_img_arr1, url_img_arr2, dir1, dir2):
    url_arr1 = [rf"{dir1}/{i}img.jpg" for i in range(len(os.listdir(dir1)))]
    url_arr2 = [rf"{dir2}/{i}img.jpg" for i in range(len(os.listdir(dir2)))]
    container = []
    for index, img1_url in enumerate(url_arr1):
        current_arr = [img1_url]
        for img2_url in url_arr2:
            current_arr.append(img2_url)
        container.append(get_cosine_similarity(current_arr)[0])
    return container


def remove_all():
    for d in os.listdir(r"/home/artyom/PycharmProjects/ImageComparer/test_imgs/site1"):
        os.remove(rf"/home/artyom/PycharmProjects/ImageComparer/test_imgs/site1/{d}")
    for d in os.listdir(r"/home/artyom/PycharmProjects/ImageComparer/test_imgs/site2"):
        os.remove(rf"/home/artyom/PycharmProjects/ImageComparer/test_imgs/site2/{d}")


def main(base_dir, url_arr1, url_arr2):
    dir1 = rf"{base_dir}/site1"
    dir2 = rf"{base_dir}/site2"
    remove_all()
    save_image(url_arr1, "1")
    save_image(url_arr2, "2")
    img_arr1 = [os.path.abspath(img_path) for img_path in os.listdir(dir1)]
    img_arr2 = [os.path.abspath(img_path) for img_path in os.listdir(dir2)]
    similarities = compare_images(img_arr1, img_arr2, dir1, dir2)
    res = convert_np_to_list(similarities)
    return get_max_similarity(res, dir1, url_arr1, url_arr2)


if __name__ == "__main__":
    arr1 = ["https://belbraer.by/storage/products/CLAAA5/md/oblitsovochnyy_kirpich_krasnyy_gladkiy_07_nf.jpg",
            "https://www.architime.ru/specarch/gustave_eiffel/1.jpg",
            "https://combo.staticflickr.com/pw/images/favicons/favicon-57.png",
            "https://cdn.lifehacker.ru/wp-content/uploads/2022/01/orig-8_1671536374-576x288.jpg"]

    arr2 = ["https://texfilterkazan.ru/wp-content/themes/texfilter/images/logo.png",
            "https://combo.staticflickr.com/pw/images/favicons/favicon-228.png",
            "https://belbraer.by/storage/products/CLAAA5/md/oblitsovochnyy_kirpich_krasnyy_gladkiy_07_nf.jpg"]
    print(main("/home/artyom/PycharmProjects/ImageComparer/test_imgs",arr1, arr2))
    #remove_all()
    # save_image(arr1, "1")
    # save_image(arr2, "2")
    # img_arr1 = [os.path.abspath(img_path) for img_path in
    #             os.listdir("/home/artyom/PycharmProjects/ImageComparer/test_imgs/site1")]
    # img_arr2 = [os.path.abspath(img_path) for img_path in
    #             os.listdir("/home/artyom/PycharmProjects/ImageComparer/test_imgs/site2")]
    #
    # res = compare_images(img_arr1, img_arr2)
    # res3 = convert_np_to_list(res)
    # print(get_max_similarity(res3))
    # img_paths = r"/home/artyom/PycharmProjects/ImageComparer/test_imgs"
    # for path in os.listdir(img_paths):
    #     print(path)
    # arr = [f"{img_paths}/{path}" for path in os.listdir(img_paths)]
    # df = get_cosine_similarity(arr)
    # for line in df:
    #     print(line)
    #     print()
    # print(get_cosine_similarity(arr))
