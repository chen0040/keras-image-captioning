from keras_image_captioning.library.img_cap_loader import load_img_cap
from keras_image_captioning.library.vgg16_lstm import Vgg16LstmImgCap
import numpy as np
from random import shuffle


def main():
    seed = 42

    np.random.seed(seed)

    img_dir_path = './data/pokemon/img'
    txt_dir_path = './data/pokemon/txt'
    model_dir_path = './models/pokemon'
    data = load_img_cap(img_dir_path, txt_dir_path)

    shuffle(data)

    img_cap = Vgg16LstmImgCap()
    img_cap.load_model(model_dir_path)

    for img_path, actual_caption in data[:20]:
        predicted_caption = img_cap.predict_image_caption(img_path)
        actual_caption = actual_caption.lower()
        print('Origin: ', actual_caption)
        print('Predict: ', predicted_caption, '...')


if __name__ == '__main__':
    main()
