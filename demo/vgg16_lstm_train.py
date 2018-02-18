from keras_image_captioning.library.img_cap_loader import load_img_cap
from keras_image_captioning.library.text_fit import fit_text
from keras_image_captioning.library.vgg16_lstm import Vgg16LstmImgCap
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    seed = 42
    max_vocab_size = 5000

    np.random.seed(seed)
    img_dir_path = './data/pokemon/img'
    txt_dir_path = './data/pokemon/txt'
    model_dir_path = './models/pokemon'
    data = load_img_cap(img_dir_path, txt_dir_path)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

    text_data = [txt for _, txt in data]
    text_data_model = fit_text(text_data, max_vocab_size=max_vocab_size, max_allowed_seq_length=20)

    img_cap = Vgg16LstmImgCap()
    epochs = 100
    img_cap.fit(text_data_model, train_data, test_data, model_dir_path=model_dir_path, epochs=epochs)


if __name__ == '__main__':
    main()
