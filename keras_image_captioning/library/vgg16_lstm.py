from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate, Input, Reshape, Dense
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os
import nltk


class Vgg16LstmImgCap(object):
    model_name = "vgg16-lstm"

    def __init__(self):
        self.model = None
        self.config = None
        self.vgg16_model = None
        self.max_seq_length = None
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None

    def load_model(self, model_dir_path):
        config_file_path = Vgg16LstmImgCap.get_config_file_path(model_dir_path)
        weight_file_path = Vgg16LstmImgCap.get_weight_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.max_seq_length = self.config['max_seq_length']
        self.vocab_size = self.config['vocab_size']
        self.word2idx = self.config['word2idx']
        self.idx2word = self.config['idx2word']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)
        self.vgg16_model = VGG16(weights='imagenet', include_top=True)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCap.model_name + '-config.npy')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCap.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCap.model_name + '-weights.h5')

    def create_model(self):
        vgg16_input = Input(shape=(1000,))
        vgg16_feature_dense = Dense(units=5)(vgg16_input)
        vgg16_feature_repeat = RepeatVector(self.max_seq_length)(vgg16_feature_dense)

        language_input = Input(shape=(self.max_seq_length, self.vocab_size))
        language_model = LSTM(units=5, return_sequences=True)(language_input)

        decoder = concatenate([vgg16_feature_repeat, language_model])

        decoder = LSTM(units=5, return_sequences=False)(decoder)
        decoder_dense = Dense(units=self.vocab_size, activation='softmax')(decoder)

        model = Model([vgg16_input, language_input], decoder_dense)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print(model.summary())
        return model

    def generate_batch(self, img_features, txt_inputs, targets, batch_size):
        num_batches = len(img_features) // batch_size

        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                yield [img_features[start:end], txt_inputs[start:end]], targets[start:end]

    def transform_encoding(self, data):
        txt_inputs = []
        targets = []
        images = []

        for t in data:
            img_path, txt = t
            img = img_to_array(load_img(img_path, target_size=(224, 224)))
            txt = 'START ' + txt.lower() + ' END'
            words = nltk.word_tokenize(txt)

            for i in range(1, len(words)):
                input_seq = np.zeros(shape=(self.max_seq_length, self.vocab_size))
                output_seq = np.zeros(shape=self.vocab_size)

                if words[i] in self.word2idx:
                    output_seq[self.word2idx[words[i]]] = 1

                for j in range(0, i):
                    if words[j] in self.word2idx:
                        k = self.max_seq_length - i + j
                        input_seq[k, self.word2idx[words[j]]] = 1

                txt_inputs.append(input_seq)
                targets.append(output_seq)
                images.append(img)
        images = np.array(images, dtype=float)
        images = preprocess_input(images)
        img_features = self.vgg16_model.predict(images)

        return img_features, np.array(txt_inputs), np.array(targets)

    def fit(self, config, train_data, test_data, model_dir_path, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = 16
        if epochs is None:
            epochs = 10

        config_file_path = Vgg16LstmImgCap.get_config_file_path(model_dir_path)
        weight_file_path = Vgg16LstmImgCap.get_weight_file_path(model_dir_path)
        architecture_file_path = Vgg16LstmImgCap.get_architecture_file_path(model_dir_path)

        self.config = config
        self.max_seq_length = self.config['max_seq_length']
        self.vocab_size = self.config['vocab_size']
        self.word2idx = self.config['word2idx']
        self.idx2word = self.config['idx2word']
        self.vgg16_model = VGG16(weights='imagenet', include_top=True)
        self.model = self.create_model()

        np.save(config_file_path, self.config)

        with open(architecture_file_path, 'w') as f:
            f.write(self.model.to_json())

        checkpoint = ModelCheckpoint(weight_file_path)

        img_train, txt_train, target_train = self.transform_encoding(train_data)
        img_test, txt_test, target_test = self.transform_encoding(test_data)

        train_gen = self.generate_batch(img_train, txt_train, target_train, batch_size)
        test_gen = self.generate_batch(img_test, txt_test, target_test, batch_size)

        train_num_batches = len(img_train) // batch_size
        test_num_batches = len(img_test) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])

        return history


def main():
    img_cap = Vgg16LstmImgCap()


if __name__ == '__main__':
    main()
