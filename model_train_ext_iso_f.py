import joblib
import eif as iso

from configs import date_col, train_end_date, features_data_path, feature_path
from data_access import decide_feature_name
from configs import model_ext_iso_f_path
from data_access import get_data
from logger import get_time


class ModelTrainExtendedIsolationForest:
    """
    This class works for isolation forest. It gathers paramteres from hyper_parmaters.json.
    For now, it seems that save mode is not available
    1. Split data to train and test set according to last_day_predictor
    2. get X fautere set assign to X. X values shape according to prediction or train env.
    3. Train model by using scikit-learn Isolation Forest Module.
    4. save model as .sav file by using pickle.
    5. predict values by call .sav file and use test data.
    """
    def __init__(self, hyper_parameters=None, last_day_predictor=None, params=None):
        get_time()
        self.data = get_data(features_data_path, True)  # data that created at feature engineering
        # TODO: get specific feature from specific model.
        self.features = list(decide_feature_name(feature_path).keys())
        self.params = hyper_parameters  # get hyper parameters for model: hyper_parameters.json
        self.model_params = params
        self.train, self.test = None, None
        self.X = None
        self.model_e_iso_f = None
        self.last_day_predictor = last_day_predictor  # splitting data indicator

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

    def learning_process_ext_iso_f(self):
        print("Extended isolation forest train process is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_values(is_for_prediction=False)
        self.model_e_iso_f = iso.iForest(self.X,
                                         ntrees=self.params['num_of_trees'],
                                         sample_size=self.params['sample_size'],
                                         ExtensionLevel=len(self.features)-1)
        print("Extended Isolation Forest Model Train Process Done!")

    def prediction_ext_iso_f(self):
        print("Extended Isolation Forest Prediction Process Initialized!")
        get_time()
        self.learning_process_ext_iso_f()
        self.train_test_split()
        self.get_x_values(is_for_prediction=True)
        self.test[self.model_params['args']['pred_field']] = self.model_e_iso_f.compute_paths(X_in=self.X)
        print("Extended Isolation Forest Prediction Process Done!")


