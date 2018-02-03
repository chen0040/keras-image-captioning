# keras-image-captioning

Image captioning using recurrent network and convolutional network in Keras

The implementation of the image captioning algorithms can be found in the [library](keras_image_captioning/library):

* Vgg16LstmImgCap in [library/vgg16_lstm.py](keras_image_captioning/library/vgg16_lstm.py) implements an image captioning
algorithms based on VGG16 and LSTM encoder and decoder, with one-hot encoding for the text description

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

```