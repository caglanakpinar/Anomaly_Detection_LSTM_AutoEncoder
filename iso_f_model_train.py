import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import random

class best_prediction_model:
    def __init__(self, data, search_criteria, hyper_p_gbm, hyper_p_drf, hyper_p_dnn, hyper_p_glm, y, X, split_ratio):
        self.data = data
        self.y = y
        self.X = X
        self.split_ratio = split_ratio
        self.feature_split = []
        self.test = []
        self.train = []
        self.search_criteria = search_criteria
        self.hyper_p_gbm = hyper_p_gbm
        self.hyper_p_drf = hyper_p_drf
        self.hyper_p_dnn = hyper_p_dnn
        self.hyper_p_glm = hyper_p_glm
        self.search_criteria = search_criteria
        self.gbm_best = None
        self.drf_best = None
        self.glm_best = None
        self.dnn_best = None
        self.ensemble = None
        self.min_error = 5000
        self.best_model = None
        self.gbm_rand_grid = None
        self.drf_rand_grid = None
        self.glm_rand_grid = None
        self.dnn_rand_grid = None
        self.all_ids = None
        self.ensemble = None
        self.gbm_best_id = None
        self.drf_best_id = None
        self.glm_best_id = None
        self.dnn_best_id = None

    def session_init(self):
        h2o.init(nthreads=-1)  # checks cores and assign for all
        self.data = h2o.H2OFrame(self.data)

    def split_train_test(self):
        self.feature_split = self.data.split_frame(ratios=[self.split_ratio], seed=1234)
        self.train = self.feature_split[0]  # using 80% for training
        self.test = self.feature_split[1]  # using the rest 20% for out-of-bag evaluation

    def init_model(self):
        self.gbm_rand_grid = H2OGridSearch(H2OGradientBoostingEstimator(model_id='gbm_rand_grid' + \
                                                                                 str(random.sample(list(range(101)), 1)[
                                                                                         0]),
                                                                        nfolds=5,
                                                                        fold_assignment="Modulo",
                                                                        keep_cross_validation_predictions=True,
                                                                        stopping_rounds=10,
                                                                        score_tree_interval=1),
                                           search_criteria=self.search_criteria,
                                           hyper_params=self.hyper_p_gbm)
        self.drf_rand_grid = H2OGridSearch(H2ORandomForestEstimator(model_id='drf_rand_grid' + \
                                                                             str(random.sample(list(range(101)), 1)[0]),
                                                                    seed=1234,
                                                                    nfolds=5,
                                                                    fold_assignment="Modulo",
                                                                    balance_classes=True,
                                                                    keep_cross_validation_predictions=True),
                                           search_criteria=self.search_criteria,
                                           hyper_params=self.hyper_p_drf)

        self.glm_rand_grid = H2OGridSearch(H2OGeneralizedLinearEstimator(family="gaussian",
                                                                         nfolds=5,
                                                                         seed=1234,
                                                                         max_iterations=30,
                                                                         keep_cross_validation_predictions=True,
                                                                         compute_p_values=False),
                                           search_criteria=self.search_criteria,
                                           hyper_params=self.hyper_p_glm)

        self.dnn_rand_grid = H2OGridSearch(H2ODeepLearningEstimator(
            model_id='dnn_rand_grid' + \
                     str(random.sample(list(range(101)), 1)[0]),
            seed=1234,
            nfolds=5,
            fold_assignment="Modulo",
            keep_cross_validation_predictions=True),
            search_criteria=self.search_criteria,
            hyper_params=self.hyper_p_dnn)

    def compute_gbm(self):
        self.gbm_rand_grid.train(x=self.X, y=self.y, training_frame=self.train)
        self.gbm_best_id = self.gbm_rand_grid.get_grid(sort_by='mse', decreasing=False).model_ids[0]
        self.gbm_best = h2o.get_model(self.gbm_best_id)

    def compute_drf(self):
        self.drf_rand_grid.train(x=self.X, y=self.y, training_frame=self.train)
        self.drf_best_id = self.drf_rand_grid.get_grid(sort_by='mse', decreasing=False).model_ids[0]
        self.drf_best = h2o.get_model(self.drf_best_id)

    def compute_glm(self):
        self.glm_rand_grid.train(x=self.X, y=self.y, training_frame=self.train)
        self.glm_best_id = self.glm_rand_grid.get_grid(sort_by='mse', decreasing=False).model_ids[0]
        self.glm_best = h2o.get_model(self.glm_best_id)

    def compute_dnn(self):
        self.dnn_rand_grid.train(x=self.X, y=self.y, training_frame=self.train)
        self.dnn_best_id = self.dnn_rand_grid.get_grid(sort_by='mse', decreasing=False).model_ids[0]
        self.dnn_best = h2o.get_model(self.dnn_best_id)

    def compute_stack_ensemble(self):
        self.ensemble = H2OStackedEnsembleEstimator(model_id="ensemble_" + str(random.sample(list(range(100)), 1)[0]),
                                                    base_models=self.all_ids)
        self.ensemble.train(x=self.X, y=self.y, training_frame=self.train)

    def compute_best_model(self):
        self.min_error = [self.gbm_best.model_performance(self.test).mse(),
                          self.drf_best.model_performance(self.test).mse(),
                          self.glm_best.model_performance(self.test).mse(),
                          self.dnn_best.model_performance(self.test).mse()]
        if self.gbm_best.model_performance(self.test).mse() == self.min_error:
            self.best_model = self.gbm_best
        if self.drf_best.model_performance(self.test).mse() == self.min_error:
            self.best_model = self.drf_best
        if self.glm_best.model_performance(self.test).mse() == self.min_error:
            self.best_model = self.glm_best
        if self.dnn_best.model_performance(self.test).mse() == self.min_error:
            self.best_model = self.dnn_best

        print(" Best Model :", self.best_model)
        self.all_ids = [self.gbm_best_id, self.drf_best_id, self.dnn_best_id]
        self.compute_stack_ensemble()
        self.best_model = self.ensemble
        self.min_error = self.ensemble.model_performance(self.test).mse()
        print(" Best model of combination :", [i.split("_")[1] for i in self.all_ids])

    def compute_train_process(self):
        self.session_init()
        self.split_train_test()
        self.init_model()
        self.compute_gbm()
        self.compute_drf()
        self.compute_glm()
        self.compute_dnn()
