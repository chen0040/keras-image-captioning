from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate, Input, Reshape, Dense
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os
import nltk


class Vgg16LstmImgCapV2(object):
    model_name = "vgg16-lstm-v2"

    def __init__(self):
        self.model = None
        self.config = None
        self.vgg16_model = None
        self.max_seq_length = None
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None
        self.vgg16_top_included = True

    def load_model(self, model_dir_path):
        config_file_path = Vgg16LstmImgCapV2.get_config_file_path(model_dir_path)
        weight_file_path = Vgg16LstmImgCapV2.get_weight_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.max_seq_length = self.config['max_seq_length']
        self.vocab_size = self.config['vocab_size']
        self.word2idx = self.config['word2idx']
        self.idx2word = self.config['idx2word']
        self.vgg16_top_included = self.config['vgg16_top_included']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)
        self.vgg16_model = VGG16(weights='imagenet', include_top=self.vgg16_top_included)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCapV2.model_name + '-config.npy')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCapV2.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Vgg16LstmImgCapV2.model_name + '-weights.h5')

    def create_model(self):
        vgg16_input = Input(shape=(25088 if not self.vgg16_top_included else 1000, ))
        vgg16_feature_dense = Dense(units=128)(vgg16_input)
        vgg16_feature_repeat = RepeatVector(self.max_seq_length)(vgg16_feature_dense)

        language_input = Input(shape=(self.max_seq_length, ))
        language_input_embed = Embedding(
            output_dim=200,
            input_dim=self.vocab_size,
            input_length=self.max_seq_length)(language_input)
        language_model = LSTM(units=128, return_sequences=True)(language_input_embed)
        language_model = LSTM(units=128, return_sequences=True)(language_model)
        language_model = TimeDistributed(Dense(128, activation='relu'))(language_model)

        decoder = concatenate([vgg16_feature_repeat, language_model])

        decoder = LSTM(units=128, return_sequences=False)(decoder)
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
        img_features = []

        for t in data:
            img_path, txt = t
            print(txt)
            img = img_to_array(load_img(img_path, target_size=(224, 224)))
            img = np.expand_dims(img, axis=0)
            img_feature = self.vgg16_model.predict(img).ravel()

            txt = 'START ' + txt.lower() + ' END'

            words = nltk.word_tokenize(txt)
            words = [word for word in words if word.isalnum()]

            if len(words) > self.max_seq_length:
                words = words[:self.max_seq_length - 1] + ['END']

            print(words)

            for i in range(1, len(words)):
                input_seq = np.zeros(shape=self.max_seq_length)
                output_seq = np.zeros(shape=self.vocab_size)

                if words[i] in self.word2idx:
                    output_seq[self.word2idx[words[i]]] = 1

                for j in range(0, i):
                    if words[j] in self.word2idx:
                        k = self.max_seq_length - i + j
                        input_seq[k] = self.word2idx[words[j]]

                txt_inputs.append(input_seq)
                targets.append(output_seq)
                img_features.append(img_feature)

        print('samples encoded: ', len(img_features))

        return np.array(img_features), np.array(txt_inputs), np.array(targets)

    def fit(self, config, train_data, test_data, model_dir_path, vgg16_top_included=None, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = 16
        if epochs is None:
            epochs = 10
        if vgg16_top_included is not None:
            self.vgg16_top_included = vgg16_top_included

        config_file_path = Vgg16LstmImgCapV2.get_config_file_path(model_dir_path)
        weight_file_path = Vgg16LstmImgCapV2.get_weight_file_path(model_dir_path)
        architecture_file_path = Vgg16LstmImgCapV2.get_architecture_file_path(model_dir_path)

        self.config = config
        self.max_seq_length = self.config['max_seq_length']
        self.vocab_size = self.config['vocab_size']
        self.word2idx = self.config['word2idx']
        self.idx2word = self.config['idx2word']
        self.config['vgg16_top_included'] = self.vgg16_top_included
        self.vgg16_model = VGG16(weights='imagenet', include_top=self.vgg16_top_included)

        print('vocab_size: ', self.vocab_size)
        print('max_seq_length: ', self.max_seq_length)

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

    def predict_image_caption(self, img_path):
        img = img_to_array(load_img(img_path, target_size=(224, 224)))
        img = np.expand_dims(img, axis=0)
        img_feature = self.vgg16_model.predict(img).ravel()
        input_seq = np.zeros(shape=self.max_seq_length)
        input_seq[self.max_seq_length-1] = self.word2idx['START']
        input_seq = np.expand_dims(input_seq, axis=0)
        wid_list = ['START']
        while wid_list[len(wid_list)-1] != 'END':
            output_tokens = self.model.predict([img_feature, input_seq])
            output_idx = np.argmax(output_tokens[0, :])
            output_word = self.idx2word[output_idx]
            wid_list.append(output_word)

            if len(wid_list) > self.max_seq_length:
                break

            input_seq = np.zeros(shape=self.max_seq_length)
            for j in range(0, len(wid_list)):
                k = self.max_seq_length - len(wid_list) + j
                input_seq[k] = self.word2idx[wid_list[j]]
            input_seq = np.expand_dims(input_seq, axis=0)

        return ' '.join(wid_list).replace('START', '').replace('END', '').strip()

