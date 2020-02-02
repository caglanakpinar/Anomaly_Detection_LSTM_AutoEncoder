import sys
from feature_engineering import CreateFeatures
from configs import feature, is_local_run, sample_args, hyper_parameter_path
from model_train import ModelTrain
from dashboard import create_dahboard
import logger
from data_access import model_from_to_json

if __name__ == "__main__":
    logger.get_time()
    if is_local_run:
        sys.argv = sample_args
    sys.stdout = logger.Logger()
    print(sys.argv)
    if len(sys.argv) != 0:
        if (sys.argv[1]) == 'feature_engineering':
            create_feature = CreateFeatures(model_deciding=sys.argv[2])
            create_feature.compute_features()
        if (sys.argv[1]) == 'train_process':
            model_train = ModelTrain(model_deciding=sys.argv[2],
                                     hyper_parameters=model_from_to_json(hyper_parameter_path, [], False),
                                     last_day_predictor=int(sys.argv[3]))
            model_train.model_running()
            model_train.test_data_prediction()
        if sys.argv[1] == 'prediction':
            model_train = ModelTrain().test_data_prediction()
        if (sys.argv[1]) == 'dashboard':
            model_train = ModelTrain()
            model_train.test_data_prediction()
            create_dahboard(model_train.data, model_train.test, feature)
    logger.get_time()








