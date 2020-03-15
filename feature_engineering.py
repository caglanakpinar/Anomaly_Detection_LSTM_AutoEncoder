import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from configs import data_path, features_data_path, feature_path, is_min_max_norm
from data_access import get_data, write_to_csv, decide_feature_name
from logger import get_time


class CreateFeatures:
    """
    creates features for models. Gethers functions for features from data_manipulation.py
    data_path: if all creates all features at features.json seperately. for specific features assign argument as feature
                Ex: python main.py feature_engineering slope
    data: gethers row data send this into the feature.sjon of each feature of data key. It uses for creating features.
    columns: raw data of columns
    features: feature dictionary from features.json. each object represents a feature.
    model_deciding:
    """
    def __init__(self, model_deciding=None):
        get_time()
        self.data_path = data_path if model_deciding == 'all' else features_data_path
        self.data = get_data(path=self.data_path, is_for_model=False)
        self.columns = list(self.data.columns)
        self.features = decide_feature_name(feature_path)
        self.model_deciding = model_deciding

    def deciding_computing_features(self):
        if self.model_deciding != 'all':
            feature_2 = {}
            for f in self.model_deciding.split("-"):
                self.check_features_existed(self.features[f]['args']['feature'],
                                            self.features[f]['args']['related_columns'],
                                            self.features[f]['args']['using_normalization']
                                            )
                feature_2[f] = self.features[f]
            self.features = feature_2
            print("features : ", self.model_deciding.split("-"))

    def check_features_existed(self, feature_col, related_cols, using_normalization):
        if feature_col in self.columns:  # if there is no feature name is updated
            self.data = self.data.drop(feature_col, axis=1)
        if using_normalization:  # if feature
            feature_col = feature_col + '_min_max_p_value' if is_min_max_norm else feature_col + '_p_value'
        if feature_col in list(self.columns):
            print("feature columns :", feature_col, " is deleted!!")
            self.data = self.data.drop(feature_col, axis=1)
        if len(set(related_cols) & set(list(self.columns))) != 0:
            print("removing columns related to feature :", list(set(related_cols) & set(list(self.columns))))
            self.data = self.data.drop(related_cols, axis=1)

    def features_data_arrange(self):
        for f in self.features:
            self.features[f]['args']['data'] = self.data

    def assign_last_day_label(self):
        if 'is_last_day' not in list(self.data.columns):
            self.data = self.data.merge(self.data.rename(columns={'Created_Time': 'day_max'}
                                                         ).pivot_table(index='customer_id', aggfunc={'day_max': 'max'}
                                                                ).reset_index(), on='customer_id', how='left')
            self.data['is_last_day'] = self.data.apply(lambda row: 1 if row['Created_Time'] == row['day_max'] else 0, axis=1)

    def compute_features(self):
        print("*" * 20, "Feature Engineering Process", "*" * 20)
        get_time()
        self.deciding_computing_features()
        self.features_data_arrange()
        for f in self.features:
            print("Feature :", f)
            if self.features[f]['args']['num_of_transaction_removing']:
                self.data = self.features[f]['args']['noisy_data_remover'](self.data,
                                                                           self.features[f]['args'][
                                                                               'num_of_transaction_removing'],
                                                                           self.features[f]['args'][
                                                                               'num_of_days_removing'],
                                                                           )
            self.data = self.features[f]['calling'](self.data, f)
            print("data sample size :", len(self.data))
        self.assign_last_day_label()
        write_to_csv(self.data, features_data_path)
        print("*" * 20, "Feature Engineering Process Has Done", "*" * 20)


