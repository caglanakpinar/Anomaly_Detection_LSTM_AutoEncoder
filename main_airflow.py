import sys

from feature_engineering import CreateFeatures
from configs import is_local_run, sample_args, pred_data_path
from model_train_iso_f import ModelTrainIsolationForest
from dashboard import create_dahboard
import logger
from data_access import model_from_to_json, get_data
from model_processor import trainModel


def main(args):
    logger.get_time()
    if is_local_run:
        args = sample_args
    sys.stdout = logger.Logger()
    print("*"*3, " args :", sys.argv)
    if len(args) != 0:
        if (args[1]) == 'feature_engineering':
            """
            run from terminal: python main.py feature_engineering all
            all: create all features which are at features.json
            Ex: 'python main.py feature_engineering c_m_ratios' create only 'c_m_ratios' adds to features set.
            """
            create_feature = CreateFeatures(model_deciding=sys.argv[2])
            create_feature.compute_features()

        if (args[1]) == 'train_process':
            # TODO: description must be updated
            """
            run from terminal: python main.py train_process 0
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            Models: isolation forest and AutoEncoder for Multivariate and Univariate Models
            """
            train = trainModel(args=args)
            train.process()

        if sys.argv[1] == 'prediction':
            # TODO: description must be updated
            """
            run from terminal: python main.py prediction 0
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            It creates prediction values for each transaction is added to raw data set
            """
            prediction = trainModel(args=args, is_prediction=True)
            prediction.process()

        if args[1] == 'dashboard':
            # TODO: description must be updated
            """
            run from terminal: python main.py dashboard 0 # 10.20.10.196:3030
            run from terminal: python main.py dashboard 0 uni # 10.20.10.196:3031
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            uni: creates only for univariate models. I order to run for multivariate dashboard assign null
            Dashboard for Multi - Uni Models is created
            """
            # TODO: get prediction data from predicted .csv file
            model = ModelTrainIsolationForest(last_day_predictor=int(sys.argv[2]))
            model.train_test_split()
            create_dahboard(model.train, get_data(pred_data_path, True))
        logger.get_time()








