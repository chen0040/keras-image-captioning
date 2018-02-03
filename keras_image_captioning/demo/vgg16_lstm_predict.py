from keras_image_captioning.library.vgg16_lstm import Vgg16LstmImgCap
import numpy as np
import os


def main():
    seed = 42

    np.random.seed(seed)
    img_dir_path = './data/pokemon/img'
    model_dir_path = './models/pokemon'

    image_paths = []
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.png'):
            image_paths.append(filepath)

    img_cap = Vgg16LstmImgCap()
    img_cap.load_model(model_dir_path)

    for img_path in image_paths[:20]:
        print(img_cap.predict_image_caption(img_path))


if __name__ == '__main__':
    main()
