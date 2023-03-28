import os
import httplib2
import pandas as pd
import numpy as np

from keras import Model
from keras.applications.vgg16 import VGG16
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


class ImageComparer:
    def __init__(self, directory: str = "test_imgs", model_width: int = 224, model_height: int = 224):
        self.__base_model = VGG16(weights='imagenet')  # , input_shape=(224, 224, 3)
        self.__feat_extractor = Model(inputs=self.__base_model.input, outputs=self.__base_model.get_layer("fc2").output)
        self.__directory = os.path.abspath(directory)
        self.__model_width = model_width
        self.__model_height = model_height
        self.__site1_images_list = []
        self.__site2_images_list = []

    @staticmethod
    def __prepro(img_obj: object):
        numpy_img = img_to_array(img_obj)
        image_batch = np.expand_dims(numpy_img, axis=0)
        processed_image = preprocess_input(image_batch.copy())
        return processed_image

    def __open_img_with_path(self, img_path: str):
        img = load_img(img_path, target_size=(self.__model_width, self.__model_height))
        preprocessed_img = self.__prepro(img)
        return preprocessed_img

    def SAVE_IMAGES(self, img_url_list: list[str], site_num: int):
        for i, url in enumerate(img_url_list):
            h = httplib2.Http('.cache')
            responce, content = h.request(url)
            out = open(rf'{self.__directory}/site{int(site_num)}/{i}img.jpg', 'wb')
            out.write(content)
            out.close()
        return "done"

    def get_cosine_similarity(self, images: list[str]):
        images = [self.__open_img_with_path(img) for img in images]
        np_img_array = np.vstack(images)
        image_features = self.__feat_extractor.predict(np_img_array)
        image_features.shape
        cosine_similarities = cosine_similarity(image_features)
        cos_similarities_df = pd.DataFrame(cosine_similarities)
        cos_similarities_df.head()
        return cosine_similarities

    @staticmethod
    def __convert_np_to_list(sim_arr: list):
        res = [np_arr.tolist() for np_arr in sim_arr]
        for arr in res:
            arr.pop(0)
        for new_arr in res:
            for double in new_arr:
                double = round(double, 3)
        return res

    def __get_max_similarity(self, similarity_arr: list, pic_url_arr1: list[str], pic_url_arr2: list[str]):
        dir1 = rf"{self.__directory}/site1"
        res = []
        for i in range(len(os.listdir(dir1))):
            pic_url1 = pic_url_arr1[i]
            similarity = max(similarity_arr[i])
            index = similarity_arr[i].index(similarity)
            pic_url2 = pic_url_arr2[index]
            res.append(f"{str(pic_url1)} - {str(pic_url2)}:  --- {str(similarity)}")
        return res

    def compare_images(self):
        url_arr1 = [f"{self.__directory}/site1/{i}img.jpg" for i in range(len(os.listdir(f"{self.__directory}/site1")))]
        url_arr2 = [f"{self.__directory}/site2/{i}img.jpg" for i in range(len(os.listdir(f"{self.__directory}/site2")))]
        container = []
        for index, img1_url in enumerate(url_arr1):
            current_arr = [img1_url]
            for img2_url in url_arr2:
                current_arr.append(img2_url)
            container.append(self.get_cosine_similarity(current_arr)[0])
        return container

    def clear_directory(self):
        for dir in os.listdir(self.__directory):
            for file in os.listdir(rf"{self.__directory}/{dir}"):
                try:
                    os.remove(rf"{self.__directory}/{dir}/{file}")
                except FileNotFoundError as exc_f:
                    print(f"file not found: {exc_f.filename}")
                except PermissionError as exc_p:
                    print(f"not enough permisson for: {exc_p.filename}")

    def main(self, url_arr1, url_arr2):
        self.clear_directory()
        self.SAVE_IMAGES(url_arr1, 1)
        self.SAVE_IMAGES(url_arr2, 2)
        similarities = self.compare_images()
        res = self.__convert_np_to_list(similarities)
        return self.__get_max_similarity(res, url_arr1, url_arr2)


if __name__ == "__main__":
    arr1 = ["https://belbraer.by/storage/products/CLAAA5/md/oblitsovochnyy_kirpich_krasnyy_gladkiy_07_nf.jpg",
            "https://www.architime.ru/specarch/gustave_eiffel/1.jpg",
            "https://combo.staticflickr.com/pw/images/favicons/favicon-57.png",
            "https://cdn.lifehacker.ru/wp-content/uploads/2022/01/orig-8_1671536374-576x288.jpg"]

    arr2 = ["https://texfilterkazan.ru/wp-content/themes/texfilter/images/logo.png",
            "https://combo.staticflickr.com/pw/images/favicons/favicon-228.png",
            "https://belbraer.by/storage/products/CLAAA5/md/oblitsovochnyy_kirpich_krasnyy_gladkiy_07_nf.jpg"]
    comparer = ImageComparer()
    print(comparer.main(arr1, arr2))
    comparer.clear_directory()

