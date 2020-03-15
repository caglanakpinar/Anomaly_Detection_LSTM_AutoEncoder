import sys

from feature_engineering import CreateFeatures
from configs import is_local_run, sample_args, pred_data_path
from model_train_iso_f import ModelTrainIsolationForest
from dashboard import create_dahboard
import logger
from data_access import model_from_to_json, get_data
from model_processor import trainModel

if __name__ == "__main__":
    logger.get_time()
    if is_local_run:
        sys.argv = sample_args
    sys.stdout = logger.Logger()
    print("*"*3, " args :", sys.argv)
    if len(sys.argv) != 0:
        if sys.argv[1] == 'feature_engineering':
            """
            run from terminal: python main.py feature_engineering all
            all: create all features which are at features.json
            Ex: 'python main.py feature_engineering c_m_ratios' create only 'c_m_ratios' adds to features set.
            """
            create_feature = CreateFeatures(model_deciding=sys.argv[2])
            create_feature.compute_features()

        if sys.argv[1] == 'train_process':
            """
            run from terminal: python main.py train_process 0
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            Models: isolation forest & AutoEncoder & DBScan 
            """
            train = trainModel(args=sys.argv)
            train.process()

        if sys.argv[1] == 'prediction':
            """
            run from terminal: python main.py prediction 0
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            It creates prediction values for each transaction is added to raw data set
            """
            prediction = trainModel(args=sys.argv, is_prediction=True)
            prediction.process()

        if sys.argv[1] == 'dashboard':
            """
            run from terminal: python main.py dashboard 0
            0/1: 0; test data splits from date
                 1: test data is last day of each customer
            """
            model = ModelTrainIsolationForest(last_day_predictor=int(sys.argv[2]))
            model.train_test_split()
            create_dahboard(model.train, get_data(pred_data_path, True))
        logger.get_time()








