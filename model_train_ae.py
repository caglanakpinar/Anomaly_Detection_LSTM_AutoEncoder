from pandas import Series
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import TensorBoard

from configs import date_col, train_end_date, features_data_path, auto_encoder_model_paths, feature_path
from data_access import get_data, decide_feature_name
from logger import get_time


class ModelTrainAutoEncoder:
    def __init__(self, hyper_parameters=None, last_day_predictor=None, test_data=None, is_multivariate=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.features = list(decide_feature_name(feature_path).keys())
        self.tuned_parameters = hyper_parameters
        self.last_day_predictor = last_day_predictor
        self.train, self.test = None, None if test_data is None else test_data
        self.X, self.y_pred, self.y = None, None, None
        self.input, self.fr_output = None, None
        self.model_ae, self.model_ae_l = None, None
        self.is_multivariate = is_multivariate
        self.log_dir = "logs/fit/"
        self.tensorboard_callback = {}

    def model_logs(self, m_name):
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir + m_name + '_' + str(datetime.datetime.now())[0:10],
                                                histogram_freq=1)

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

    def auto_encoder(self):
        self.input = Input(shape=(len(self.features),))
        encoder1 = Dense(200, activation='tanh')(self.input)
        encoder2 = Dense(100, activation='tanh')(encoder1)
        encoder3 = Dense(50, activation='tanh')(encoder2)
        encoder4 = Dense(10, activation='tanh')(encoder3)
        self.fr_output = Dense(1, activation='tanh')(encoder4)
        decoder1 = Dense(10, activation='tanh')(self.fr_output)
        decoder2 = Dense(50, activation='tanh')(decoder1)
        decoder3 = Dense(100, activation='tanh')(decoder2)
        decoder4 = Dense(200, activation='tanh')(decoder3)
        decoder5 = Dense(len(self.features), activation='sigmoid')(decoder4)
        self.model_ae = Model(inputs=self.input, outputs=decoder5)
        self.model_ae.compile(loss='mse', optimizer=RMSprop(learning_rate=0.01), metrics=['mse'])

    def auto_encoder_linear(self):
        self.input = Input(shape=(len(self.features),))
        self.fr_output = Dense(1, activation='relu')(self.input)
        decoder = Dense(len(self.features), activation='sigmoid')(self.fr_output)
        self.model_ae_l = Model(inputs=self.input, outputs=decoder)
        self.model_ae_l.compile(loss='mse', optimizer=RMSprop(learning_rate=0.01), metrics=['mse'])

    def feature_reduction(self):
        return Model(inputs=self.input, outputs=self.fr_output)

    def model_train(self):
        print("Auto Encoder is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_values(is_for_prediction=False)
        self.auto_encoder()
        self.auto_encoder_linear()
        self.model_logs('ae')
        self.model_ae.fit(self.X, self.X,
                          epochs=int(self.tuned_parameters['feature_reduction']['epochs']),
                          batch_size=int(self.tuned_parameters['feature_reduction']['batch_size']),
                          validation_split=0.2, callbacks=[self.tensorboard_callback])
        self.model_from_to_json(auto_encoder_model_paths['ae'], self.feature_reduction(), is_writing=True)
        self.model_logs('ae_l')
        self.model_ae_l.fit(self.X, self.X,
                            epochs=int(self.tuned_parameters['feature_reduction']['epochs']),
                            batch_size=int(self.tuned_parameters['feature_reduction']['batch_size']),
                            validation_split=0.2, callbacks=[self.tensorboard_callback])
        self.model_from_to_json(auto_encoder_model_paths['ae_l'], self.feature_reduction(), is_writing=True)

    def calculating_loss_function(self, is_for_prediction):
        self.model_ae = self.model_from_to_json(auto_encoder_model_paths['ae'], is_writing=False)
        self.model_ae_l = self.model_from_to_json(auto_encoder_model_paths['ae_l'], is_writing=False)
        self.get_x_values(is_for_prediction=True)
        self.y_pred = self.model_ae.predict(self.X)
        self.y = self.model_ae_l.predict(self.X)
        # AE works better when we try to find out the outliers
        anomaly_calculations = list(map(lambda x: abs(x[0] - x[1])[0], zip(self.y_pred, self.y)))
        self.test['anomaly_ae_values'] = Series(anomaly_calculations)
        self.test.to_csv("prediction.csv", index=False)

