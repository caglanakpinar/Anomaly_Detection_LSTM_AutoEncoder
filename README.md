
# Anomaly Detection Framework
### Overview
### How it works?
### Metrics
### Features
### Libraries
### Data Sets (Random Data Generator)
### Ensemble Model (Isolation Forest)
### Deep Learning Model (AutoEncoder)
### Deep Learning Model (LSTM - AutoEncoder)
### Visualizations

## Overview

You only have customer, transaction, merchant of unique ids and transaction date and and transactions of Amounts. How can you define the Abnormalities of each Transaction? This process allows us to find the abnomalities with deciding metrics. This project can be integrated any business which has transactions with amount and date and customers and merchants. It finds the abnormal transaction with 4 metrics which they can be define as generic features for each business.

## How it works?

There are 3 main processes. 1st You can create features. If there is not observed data set, it generates random data set. It is possible to assign a data set with .csv file or data connection. Data of file path must be assigned at configs.py data_path. It genreated features from the feature.json file. All information how to generates are assigned at feature.json file. 2ndYou can tran with features set with Isolation Forest, AutoEncoder (Multivariate - Univariate), LSTM - AutoEncoder. 3rd, it is possible to show results with dashbord

**1st generate random data and create features:**

```
main.py feature_engineering all
```

**all:** when all features generating. If there is new feature to add or regenerate a feature it is possible write feature name instead of 'all'. Make sure this feature name is exactly same as in feature.json file.

**2nd train generated data set:**

```
main.py train_process 0
```

**3rd show dashboard data set:** 

```
main.py dashboard 0 
```
**0:** test data set dividing according to date which is assigned at configs.py

**1:** test data is related to each customer of last transaction day. 

## Metrics

### 1. Customer Of Merchant Transactions Counts:

 If a transaction occurs from a merchant that the customer has no transaction from, it would be abnormal transaction.
 
### 2. Transaction of frequencies of each customer:

For instance, let's saya customer has transaction of each month suddenlt if the customer use the card for every hour this would abnormal transactions

### 3. Sudden drop and peak of each customer - merchant of amounts on each transaction:

It can assume that each customer - merchant can not be huge drop or increase suddenly. This would be abnormal transaction

## Features

In order to add feature to project, feature.json file must be updated.

**calling:** Each feature needs a function to be calculated. Each function must be added to data_manipulation.py. Calling key of this dictionary allows us to use for calculating the related feature. 

**noisy_data_remover:** FRor each feature raw data set of noisy data removing separately. Data removing function must be added to data_manipulation.py

**using_normalization:** If we need to use normalization for the feature assign True. If this field is Tru make sure 'is_min_max_norm' is True depending on Min - Max Normalization or - Gaussian P Value Normalization

### 1. c_m_ratios

```
"c_m_ratios": {"args": {
                        "data": "-",
                        "noisy_data_remover": "remove_noisy_data",
                        "num_of_transaction_removing": 5,
                        "num_of_days_removing": 5,
                        "feature": "c_m_ratios",
                        "using_normalization": "False",
                        "related_columns": "-"
                    },
                   "calling": "clustered_merchant_ratios_feature",
                   "name": "C. - M. Transaction Ratio Scores"
    }
 ```
  
clustered_merchant_ratios_feature: First, This function starts with clustering merchants into the categories. If we have categories of merchants as like -commerce, retail, Goverment, retail, we don't need to cluster the merhants. 

### *  Merchant Clustering (RFM)

In order to label the merchants, it better using metrics of one of the popular segmentation technique which is RFM. Recenecy how recently a merchant has transactions, monetary, what the aveage spent on the merchant, how frequently the merchant has transactions

![Image description](<img width="589" alt="merchant_rfm_kmeans_clustering" src="https://user-images.githubusercontent.com/26736844/74594122-da140f00-5043-11ea-866e-3d3158587b33.png">)

### * How is the c_m_ratios calculated?

After labeling merchants total merchant label  total transactions of each customer has been calculated. 

ratio  = customer-merchant total transaction / customer-merchant label total transaction

### 2. c_freq_diff

This feature refers to 2nd metric. In order to calculate this metrics exploratory analysis will help us to see how can this feature is distributed. 

```
"c_freq_diff": {"args": {
                                    "data": "-",
                                    "noisy_data_remover": "remove_noisy_data",
                                    "num_of_transaction_removing": 5,
                                    "num_of_days_removing": 5,
                                    "feature": "c_freq_diff",
                                    "using_normalization": "True",
                                    "related_columns": "-"
                                    },
                                "calling": "customer_transaction_day_diff_feature",
                                "name": "C. Difference Of Each Transaction Score"
}
```
                                

### * How is the c_m_ratios calculated?

**-** Every customer of transactions hourly differences are calculated.

**-** Each customer of average frequency is calculated. 

**-** Each Transaction Hourly Difference from the average daily differences are calculated. Negative Differences as their values, however we are ignoring (+) and assigning them 0. Beacuse if a customer is not engaged to online card enviroment, it is posibity of Fraud will decrease.

**-** Now, it is time to Normalize these values into the range 0 - 1. 

#### 3. Customer - Merchant Amount of Peak And Drop Values Of Scores

THis features allows us to see very huge drop and peak of each customer engagement on each merchants.

```
"c_m_peak_drop_min_max_p_value": {"args": {
                                                   "data": "-",
                                                   "noisy_data_remover": "remove_noisy_data",
                                                   "num_of_transaction_removing": 5,
                                                   "num_of_days_removing": 5,
                                                   "feature": "c_m_med_amount_change", "related_columns": "-",
                                                   "using_normalization": "False"
},
                                          "name": "C. M. AbNormal Peak And Drop Score",
                                          "calling": "customer_merchant_peak_drop"
                                          }
```

If there is a historic transaction as like below probably last transaction will have really high score.

<img width="472" alt="huge_drop" src="https://user-images.githubusercontent.com/26736844/74594863-2ebb8800-504c-11ea-9114-c812cbd7d6ba.png">

At the end we have a score distribution as like below

![Customer Merchant Amount Peak And Drop Scores](https://user-images.githubusercontent.com/26736844/74594810-7d1c5700-504b-11ea-8453-a0f24872060d.png)
                                                 
## Libraries

```
pip install tensorflow>=1.15.0
pip install sklearn>=0.22.1
```

## Data Sets (Random Data Generator)

Whole data set is not related to any data set. However randaom data generates with rules in order to make sense to catch Abnormalities. This process at random_data_generator.py and function is generate_random_data_v2.

1. It is a loop of each customer. So, Each random generated customer has transaction from each random generated merchant

2. Customer and merchant count and other random generation constants are at configs.py.

### Ensemble Model (Isolation Forest)

Return the anomaly score of each sample using the IsolationForest algorithm

The IsolationForest 'isolates' observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

**ModelTrainIsolationForest** 

**data:** it is the transition of raw data with generated features

**features:** key of dictionary from feature.json file

**train:** train data set of Isolation Forest. 

**test:** test data set of Isolation Forest. This is splitting according to last_day_predictor

**model_iso:** isolation model aim to train

**last_day_predictor:** shows how data set splitting into train and test. If it is 0 that means splitting according to train_date_end. If is is 1 it is splitting according to is_last_day

**train_test_split:** splits raw data into the train and test

**get_x_values:** gets values of test or train with features

**model_from_to_pickle:** model saving to json file

**learning_process_iso_f:** fit the model

**prediction_iso_f:** predicting model with trained model








