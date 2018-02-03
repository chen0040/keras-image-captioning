# keras-image-captioning

Image captioning using recurrent network and convolutional network in Keras

The implementation of the image captioning algorithms can be found in the [library](keras_image_captioning/library):

* Vgg16LstmImgCap in [library/vgg16_lstm.py](keras_image_captioning/library/vgg16_lstm.py) implements an image captioning
algorithms based on VGG16 (top included) and LSTM encoder and decoder, with one-hot encoding for the text description
* Vgg16LstmImgCap in [library/vgg16_lstm_v2.py](keras_image_captioning/library/vgg16_lstm_v2.py) implements an image 
captioning algorithms based on VGG16 (top not included) and LSTM encoder and decoder, with word embedding for the text 
description

# Usage

### Vgg16LstmImgCap

The [sample codes](keras_image_captioning/demo/vgg16_lstm_train.py) below shows how to train the pokemon image 
captioning using image and text data in the [demo/data/pokemon](keras_image_captioning/demo/data/pokemon):

```python
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

    config = fit_text(data, max_vocab_size=max_vocab_size, max_allowed_seq_length=20)

    img_cap = Vgg16LstmImgCap()
    epochs = 100
    img_cap.fit(config, train_data, test_data, model_dir_path=model_dir_path, epochs=epochs)


if __name__ == '__main__':
    main()

```

The [sample codes](keras_image_captioning/demo/vgg16_lstm_predict.py) below shows how to predict the captioning
 for the pokemon image in the [demo/data/pokemon/img](keras_image_captioning/demo/data/pokemon/img):
 
```python
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

```