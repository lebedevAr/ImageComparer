import os
import httplib2
import pandas as pd
import numpy as np
import logging

from keras import Model
from keras.applications.vgg16 import VGG16
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

from PIL import UnidentifiedImageError

logging.basicConfig(format="%(levelname)s | %(name)s | %(asctime)s | %(message)s", level="INFO")
logger = logging.getLogger("ImageComparer")


class ImageComparer:
    def __init__(self, directory: str = "test_imgs", model_width: int = 224, model_height: int = 224):
        self.__base_model = VGG16(weights='imagenet')  # , input_shape=(224, 224, 3)
        self.__feat_extractor = Model(inputs=self.__base_model.input, outputs=self.__base_model.get_layer("fc2").output)
        self.__directory = os.path.abspath(directory)
        self.__model_width = model_width
        self.__model_height = model_height
        self.__url_arr1 = []
        self.__url_arr2 = []

    def create_dirs(self):
        if not os.path.exists(self.__directory):
            os.mkdir(self.__directory)
            path1 = f"{self.__directory}/site1"
            path2 = f"{self.__directory}/site2"
            os.mkdir(path1)
            os.mkdir(path2)
            logger.debug("Папки успешно созданы")
        else:
            logger.debug("Папки были созданы ранее")

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
        logger.info(f"Успешно скачаны изображения с {site_num} сайта")

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
        res_dict = {}
        for i in range(len(os.listdir(dir1))):
            pic_url1 = pic_url_arr1[i]
            similarity = max(similarity_arr[i])
            index = similarity_arr[i].index(similarity)
            pic_url2 = pic_url_arr2[index]
            res_dict[f"{str(pic_url1)} - {str(pic_url2)}:  --- "] = round(similarity * 100)
        logger.debug("Получен результирующий словарь")
        return res_dict

    def compare_images(self):
        url_arr1 = [f"{self.__directory}/site1/{img}" for img in os.listdir(f"{self.__directory}/site1")]
        url_arr2 = [f"{self.__directory}/site2/{img}" for img in os.listdir(f"{self.__directory}/site1")]
        container = []
        for index, img1_url in enumerate(url_arr1):
            current_arr = [img1_url]
            for img2_url in url_arr2:
                current_arr.append(img2_url)
            container.append(self.get_cosine_similarity(current_arr)[0])
        logger.info("Получены признаки схожести изображений")
        return container

    def clear_directories(self):
        for dir in os.listdir(self.__directory):
            if os.listdir(rf"{self.__directory}/{dir}") != 0:
                for file in os.listdir(rf"{self.__directory}/{dir}"):
                    try:
                        os.remove(rf"{self.__directory}/{dir}/{file}")
                    except PermissionError as exc_p:
                        logger.warning(f"not enough permisson for: {exc_p.filename}")
        logger.debug("Папки успешно очищены")

    def __check_images(self):
        base_dir = self.__directory
        for site in os.listdir(base_dir):
            for img in os.listdir(f"{base_dir}/{site}"):
                try:
                    self.__open_img_with_path(f"{base_dir}/{site}/" + img)
                except UnidentifiedImageError as e:
                    logger.warning(f"{str(e).split('<')[1]} is invalid, удаление...")
                    os.remove(f"{base_dir}/{site}/" + img)

    def clean_cache(self):
        path = os.path.abspath(self.__directory).split('/')
        path_to_cache = "/".join(path[:-1]) + "/.cache"
        for f in os.listdir(path_to_cache):
            try:
                os.remove(f"{path_to_cache}/{f}")
            except FileNotFoundError:
                logger.debug("Кэш уже очищен")

    def main(self, url_arr1, url_arr2):
        self.create_dirs()
        self.clear_directories()
        self.SAVE_IMAGES(url_arr1, 1)
        self.SAVE_IMAGES(url_arr2, 2)
        self.__check_images()
        similarities = self.compare_images()
        sim_arr = self.__convert_np_to_list(similarities)
        res_dict = self.__get_max_similarity(sim_arr, url_arr1, url_arr2)
        # self.clear_directories()
        self.clean_cache()
        avg = f"{round(sum(res_dict.values()) / len(res_dict))} %"
        logger.info(f"{self.__class__.__name__} завершил работу")
        return avg


if __name__ == "__main__":
    a = (['https://bananashow.ru/wp-content/themes/sitestyle/img/logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/logo_clear.png',
          'https://i3.ytimg.com/vi/Hc0OcKh0ZMw/hqdefault.jpg', 'https://i3.ytimg.com/vi/RxVJ1btfEgg/hqdefault.jpg',
          'https://i3.ytimg.com/vi/SGez4lqkxAo/hqdefault.jpg', 'https://i3.ytimg.com/vi/mlXp5rl-Tak/hqdefault.jpg',
          'https://i3.ytimg.com/vi/OV-cwSBw3SI/hqdefault.jpg', 'https://i3.ytimg.com/vi/JoaR5il2CWQ/hqdefault.jpg',
          'https://i3.ytimg.com/vi/zlOiSCZMfzs/hqdefault.jpg', 'https://i3.ytimg.com/vi/y1vyl0xVoQM/hqdefault.jpg',
          'https://i3.ytimg.com/vi/t8jFUbJZ0yQ/hqdefault.jpg', 'https://i3.ytimg.com/vi/bkhjeHWT3aM/hqdefault.jpg',
          'https://i3.ytimg.com/vi/c-LMTzYtFhQ/hqdefault.jpg', 'https://i3.ytimg.com/vi/OhuKION4CsQ/hqdefault.jpg',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/themes/sitestyle/img/yandex-review-logo.png',
          'https://bananashow.ru/wp-content/uploads/2018/05/favicon.png',
          'https://bananashow.ru/wp-content/uploads/2018/05/favicon.png',
          'https://bananashow.ru/wp-content/uploads/2018/05/favicon.png'],
         ['https://designinterior.moscow/wp-content/uploads/thegem-logos/logo_917acf37f9e832a814063157ff7373c2_1x.png',
          'https://designinterior.moscow/wp-content/uploads/thegem-logos/logo_917acf37f9e832a814063157ff7373c2_1x.png',
          'https://designinterior.moscow/wp-content/uploads/thegem-logos/logo_05629ed766312692f13cc9a8f4bd0be9_1x.png',
          'https://designinterior.moscow/wp-content/uploads/2021/04/zagolovok-proekta-kutuzovskij-life-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2021/04/zilart-holl-aleksej-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/10/20000-131-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/03/60000-6-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/03/gh30000-2-2.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/08/1r-60000-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/01/12gost0000-2800.jpg',
          'https://designinterior.moscow/wp-content/uploads/2020/10/klub-senatorov-10000-fin-3-thegem-portfolio-justified.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/1-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/2-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/3-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/4-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/5-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/6-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/7-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/8-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/9-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/10-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/11-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/12-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/13-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/14-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/15-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/16-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/17-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/18-thegem-post-thumb-small.jpg',
          'https://designinterior.moscow/wp-content/uploads/2019/07/20163.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/icon-233.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/19039.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-4.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-4.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/20119.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/icon-4.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/icon-1-2.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-3.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-3.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/icon-5.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/icon-19.png',
          'https://designinterior.moscow/wp-content/uploads/2019/07/18973.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-4.png',
          'https://designinterior.moscow/wp-content/uploads/2019/12/construction_horizontal_arrow-4.png',
          'https://designinterior.moscow/wp-content/uploads/2016/04/snimok-1-thegem-person-240.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/01/time-1.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/reshenie-1.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/sogl-1.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/help10.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/consultation1-kopija-1.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/time-2.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/dogovor-1.png',
          'https://designinterior.moscow/wp-content/uploads/2022/01/payment-1.png',
          'https://designinterior.moscow/wp-content/uploads/2021/10/s1200-thegem-blog-slider-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2019/09/zhk-ilove-thegem-blog-slider-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2019/09/7-2-thegem-blog-slider-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2021/01/favicon.png',
          'https://designinterior.moscow/wp-content/uploads/2021/01/favicon.png',
          'https://designinterior.moscow/wp-content/uploads/2021/01/favicon.png',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fdobriy-dom-proekt-interiera-3x-komnatnoy&description=%D0%94%D0%BE%D0%B1%D1%80%D1%8B%D0%B9+%D0%B4%D0%BE%D0%BC.+%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82+%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80%D0%B0+3%D1%85+%D0%BA%D0%BE%D0%BC%D0%BD%D0%B0%D1%82%D0%BD%D0%BE%D0%B9+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B+%D0%B2+%D0%96%D0%9A+LIFE+%26%23171%3B%D0%9A%D1%83%D1%82%D1%83%D0%B7%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9%26%23187%3B&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2021%2F04%2Fzagolovok-proekta-kutuzovskij-life-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fzefir-interior-rvartiri-amerikanskaya-klassika-ziart&description=%D0%97%D0%B5%D1%84%D0%B8%D1%80.+%D0%98%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B+%D0%B2+%D1%81%D1%82%D0%B8%D0%BB%D0%B5+%D0%90%D0%BC%D0%B5%D1%80%D0%B8%D0%BA%D0%B0%D0%BD%D1%81%D0%BA%D0%B0%D1%8F+%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D0%BA%D0%B0.&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2021%2F04%2Fzilart-holl-aleksej-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fhranitel-tradicii-interior-studii-jk-zilart&description=%D0%A5%D1%80%D0%B0%D0%BD%D0%B8%D1%82%D0%B5%D0%BB%D1%8C+%D1%82%D1%80%D0%B0%D0%B4%D0%B8%D1%86%D0%B8%D0%B9.+%D0%A1%D1%82%D0%B8%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9+%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B-%D1%81%D1%82%D1%83%D0%B4%D0%B8%D0%B8+%D0%B2+%D0%96%D0%9A+%D0%97%D0%B8%D0%BB%D0%B0%D1%80%D1%82.&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F10%2F20000-131-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fneoklassika-interer-3-komnatnoj-kvartiri-jk-simvol&description=%D0%9D%D0%B5%D0%BE%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D0%BA%D0%B0.+%D0%98%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80+3+%D0%BA%D0%BE%D0%BC%D0%BD%D0%B0%D1%82%D0%BD%D0%BE%D0%B9+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B+%D0%B2+%D0%96%D0%9A+%D0%A1%D0%B8%D0%BC%D0%B2%D0%BE%D0%BB&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F03%2F60000-6-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fstolichnie-polyani-proekt-3-komnatnoj-kvartiri&description=%D0%A1%D1%82%D0%BE%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D0%B5+%D0%BF%D0%BE%D0%BB%D1%8F%D0%BD%D1%8B.+%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82+3+%D0%BA%D0%BE%D0%BC%D0%BD%D0%B0%D1%82%D0%BD%D0%BE%D0%B9+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B+%D0%B2+%D0%96%D0%9A+%D0%A1%D1%82%D0%BE%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D0%B5+%D0%BF%D0%BE%D0%BB%D1%8F%D0%BD%D1%8B&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F03%2Fgh30000-2-2-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fkvartira-studiya-jk-river-park&description=%D0%98%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80+%D0%B2+%D1%81%D1%82%D0%B8%D0%BB%D0%B5+%D1%85%D0%B0%D0%B9-%D1%82%D0%B5%D0%BA+%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D1%8B-%D1%81%D1%82%D1%83%D0%B4%D0%B8%D0%B8+%D0%B2+%D0%96%D0%9A+%D0%A0%D0%98%D0%92%D0%95%D0%A0+%D0%9F%D0%90%D0%A0%D0%9A&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F08%2F1r-60000-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fsolnechnoe-utro-kvartira-v-ghk-simvol&description=%D0%A1%D0%BE%D0%BB%D0%BD%D0%B5%D1%87%D0%BD%D0%BE%D0%B5+%D1%83%D1%82%D1%80%D0%BE.+%D0%9A%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80%D0%B0+%D0%B2+%D0%96%D0%9A+%D0%A1%D0%B8%D0%BC%D0%B2%D0%BE%D0%BB&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F01%2F12gost0000-2800-thegem-blog-timeline-large.jpg',
          'https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fdesigninterior.moscow%2Fportfolio%2Fproekt-interiera-banketnogo-zala-v-mockve&description=%D0%9A%D0%BB%D1%83%D0%B1+%D0%A1%D0%B5%D0%BD%D0%B0%D1%82%D0%BE%D1%80%D0%BE%D0%B2.+%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82+%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D1%8C%D0%B5%D1%80%D0%B0+%D0%B1%D0%B0%D0%BD%D0%BA%D0%B5%D1%82%D0%BD%D0%BE%D0%B3%D0%BE+%D0%B7%D0%B0%D0%BB%D0%B0+%D0%B2+%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B5.&media=https%3A%2F%2Fdesigninterior.moscow%2Fwp-content%2Fuploads%2F2020%2F10%2Fklub-senatorov-10000-fin-3-thegem-blog-timeline-large.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/1-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/2-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/3-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/4-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/5-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/6-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/7-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/8-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/9-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/10-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/11-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/12-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/13-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/14-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/15-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/16-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/17-thegem-gallery-fullwidth.jpg',
          'https://designinterior.moscow/wp-content/uploads/2022/02/18-thegem-gallery-fullwidth.jpg'])

    comp = ImageComparer()
    comp.main(a[0], a[1])
    comp.clear_directories()
    # for site in os.listdir("/home/artyom/PycharmProjects/ImageComparer/test_imgs"):
    #     for img in os.listdir(f"/home/artyom/PycharmProjects/ImageComparer/test_imgs/{site}"):
    #         try:
    #             comp.open_img_with_path(f"/home/artyom/PycharmProjects/ImageComparer/test_imgs/{site}/" + img)
    #         except UnidentifiedImageError as e:
    #             print(f"{str(e)} is invalid")


