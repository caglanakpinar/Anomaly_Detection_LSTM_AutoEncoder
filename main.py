import sys
import webbrowser
from feature_engineering import CreateFeatures
from feature_engineering_lstm_ae import CreateFeaturesLSTM
from configs import is_local_run, sample_args, hyper_parameter_path, port, host
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
        if (sys.argv[1]) == 'feature_engineering_lstm':
            create_feature = CreateFeaturesLSTM(model_deciding=sys.argv[2])
            create_feature.compute_features()
        if (sys.argv[1]) == 'train_process':
            hyper_parameters = model_from_to_json(hyper_parameter_path, [], False)
            model_iso_f = ModelTrainIsolationForest(hyper_parameters=hyper_parameters,
                                                    last_day_predictor=int(sys.argv[2]))
            model_ae = ModelTrainAutoEncoder(hyper_parameters=hyper_parameters,
                                             last_day_predictor=int(sys.argv[2]))
            model_iso_f.learning_process_iso_f()
            model_ae.model_train()
        if sys.argv[1] == 'prediction':
            model_iso_f = ModelTrainIsolationForest(last_day_predictor=int(sys.argv[2]))
            model_iso_f.train_test_split()
            model_iso_f.prediction_iso_f(is_for_prediction=True)
            prediction = model_iso_f.test
            model_ae = ModelTrainAutoEncoder(test_data=prediction)
            model_ae.calculating_loss_function(is_for_prediction=True)
            prediction = model_ae.test
        if (sys.argv[1]) == 'dashboard':
            model_iso_f = ModelTrainIsolationForest(last_day_predictor=int(sys.argv[2]))
            model_iso_f.train_test_split()
            model_iso_f.prediction_iso_f(is_for_prediction=True)
            prediction = model_iso_f.test
            model_ae = ModelTrainAutoEncoder(test_data=prediction)
            model_ae.calculating_loss_function(is_for_prediction=True)
            prediction = model_ae.test
            create_dahboard(model_iso_f.data, prediction)

    logger.get_time()








