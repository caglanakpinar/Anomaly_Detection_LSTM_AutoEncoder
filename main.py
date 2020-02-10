import sys
from feature_engineering import CreateFeatures
from configs import feature, is_local_run, sample_args, hyper_parameter_path
from model_train_iso_f import ModelTrainIsolationForest
from model_train_ae import ModelTrainAutoEncoder
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
            hyper_parameters = model_from_to_json(hyper_parameter_path, [], False)
            model_iso_f = ModelTrainIsolationForest(hyper_parameters=hyper_parameters,
                                                    last_day_predictor=int(sys.argv[2]))
            model_ae = ModelTrainAutoEncoder(hyper_parameters=hyper_parameters,
                                             last_day_predictor=int(sys.argv[2]))
            model_iso_f.learning_process_iso_f()
            model_ae.model_train()
            model_ae.calculating_loss_function(is_for_prediction=True)
        if sys.argv[1] == 'prediction':
            model_iso_f = ModelTrainIsolationForest().train_test_split()
            prediction = model_iso_f.prediction_iso_f(is_for_prediction=True).test
            prediction = ModelTrainAutoEncoder(test_data=prediction).calculating_loss_function().test

        if (sys.argv[1]) == 'dashboard':
            model_iso_f = ModelTrainIsolationForest().train_test_split()
            prediction = model_iso_f.prediction_iso_f(is_for_prediction=True).test
            prediction = ModelTrainAutoEncoder(test_data=prediction).calculating_loss_function().test
            create_dahboard(model_iso_f.data, prediction, feature)
    logger.get_time()








