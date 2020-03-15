import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import regularizers

from configs import date_col, train_end_date, features_data_path, auto_encoder_model_paths, feature_path, run_gpu
from data_access import decide_feature_name
from data_access import get_data
from logger import get_time


def cal_hidden_layer_of_units(hidden_layers, _encoding_dim):
    """
    allows you to gether hidden layer units for AutoEncoder
    :param hidden_layers: number of hidden layer for not work (both encoder and decoder)
    :param _encoding_dim: Start encoder for AtoEncoder
    :return: return units starts from encoder hidden layers
    """
    count = 1
    _unit = _encoding_dim
    h_l_units = []
    while count != hidden_layers + 1:
        h_l_units.append(int(_unit))
        _unit /= 2
        count += 1
    return h_l_units


class ModelTrainAutoEncoders:
    def __init__(self, hyper_parameters=None, last_day_predictor=None, params=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.features = list(decide_feature_name(feature_path).keys())
        self.params = hyper_parameters
        self.last_day_predictor = last_day_predictor
        self.model_params = params
        self.train, self.test = None, None
        self.X, self.y_pred, self.y = None, None, None
        self.input, self.fr_output = None, None
        self.model_ae, self.model_ae_l, self.model_u = None, None, None
        self.gpu_devices = [d for d in device_lib.list_local_devices() if d.device_type == "GPU"] if run_gpu else []

    def model_from_to_json(self, path, model=None, is_writing=False):
        if is_writing:
            model_json = model.to_json()
            with open(path, "w") as json_file:
                json_file.write(model_json)
        else:
            json_file = open(path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            return model_from_json(loaded_model_json)

    def train_test_split(self):
        if self.last_day_predictor == 1:
            self.train = self.data[self.data['is_last_day'] == 0]
            self.test = self.data[self.data['is_last_day'] == 1]
        else:
            self.train = self.data[self.data[date_col] < train_end_date]
            self.test = self.data[self.data[date_col] >= train_end_date]
        print("train set :", len(self.train), " || test set :", len(self.test))

    def get_x_values(self, is_for_prediction):
        self.X = self.test[self.features].values if is_for_prediction else self.train[self.features].values

    def build_model(self):
        self.h_units = cal_hidden_layer_of_units(self.params['h_layers'], self.params['encode_dim'])
        self.input = Input(shape=(len(self.features),))
        self.hidden = self.input
        for _unit in self.h_units:
            self.hidden = Dense(_unit,
                                activation=self.params['activation'],
                                activity_regularizer=regularizers.l1(self.params['l1']))(self.hidden)
        self.fr_output = Dense(len(self.features), activation='relu')(self.hidden)
        self.model = Model(inputs=self.input, outputs=self.fr_output)
        self.model.compile(loss='mae', optimizer=RMSprop(lr=self.params['lr']), metrics=['mae'])
        print(self.model.summary())



    def auto_encoder(self):
        self.input = Input(shape=(len(self.features),))
        encoder1 = Dense(1000, activation='tanh')(self.input)
        encoder2 = Dense(500, activation='tanh')(encoder1)
        encoder3 = Dense(100, activation='tanh')(encoder2)
        encoder4 = Dense(50, activation='tanh')(encoder3)
        encoder5 = Dense(10, activation='tanh')(encoder4)
        code = Dense(len(self.features), activation='relu')(encoder5)
        decoder1 = Dense(len(self.features), activation='relu')(code)
        decoder2 = Dense(10, activation='tanh')(decoder1)
        decoder3 = Dense(50, activation='tanh')(decoder2)
        decoder4 = Dense(100, activation='tanh')(decoder3)
        decoder5 = Dense(500, activation='tanh')(decoder4)
        decoder6 = Dense(1000, activation='tanh')(decoder5)
        self.fr_output = Dense(len(self.features), activation='sigmoid')(decoder6)
        self.model_ae = Model(inputs=self.input, outputs=self.fr_output)
        if len(self.gpu_devices) != 0:
            self.model_ae = multi_gpu_model(self.model_ae, gpus=len(self.gpu_devices))
        self.model_ae.compile(loss='mse', optimizer=RMSprop(lr=0.001), metrics=['mse'])
        print(self.model_ae.summary())

    def model_train(self):
        print("Auto Encoder is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_values(is_for_prediction=False)
        self.auto_encoder()
        if len(self.gpu_devices) != 0:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                config = tf.ConfigProto(log_device_placement=True)
                self.model_ae.fit(self.X, self.X,
                                  epochs=int(self.params['epochs']),
                                  batch_size=int(self.params['batch_size']),
                                  validation_split=0.2, shuffle=True)
        else:
            self.model_ae.fit(self.X, self.X,
                              epochs=int(self.params['epochs']),
                              batch_size=int(self.params['batch_size']),
                              validation_split=0.2, shuffle=True)
        self.model_from_to_json(auto_encoder_model_paths['ae'], self.model_ae, is_writing=True)

    def calculating_loss_function(self):
        self.train_test_split()
        self.get_x_values(is_for_prediction=True)
        self.model_ae = self.model_from_to_json(auto_encoder_model_paths['ae'], is_writing=False)
        self.y_pred, self.y = self.model_ae.predict(self.X), self.X
        anomaly_calculations = list(map(lambda x: np.mean([abs(x[0][f] - x[1][f]) for f in range(len(self.features))]),
                                        zip(self.y_pred, self.y)))
        self.test[self.model_params['args']['pred_field']] = pd.Series(anomaly_calculations)

