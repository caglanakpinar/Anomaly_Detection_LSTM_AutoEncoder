# A/B Test Bayesian Approaches
### Overview
### How it works?
### Metrics
### Features
### Dependencies
### Data Sets (Random Data Generator)
### Parameters
### Example
### Ensamble Model (Isolation Forest)
### Deep Learning Model (AutoEncoder)
### Deep Learning Model (LSTM - AutoEncoder)
### Visualizations

## Overview

You only have customer, transaction, merchant of unique ids and transaction date and and transactions of Amounts. How can you define the Abnormalities of each Transaction? This process allows us to find the abnomalities with deciding metrics. This project can be integrated any business which has transactions with amount and date and customers and merchants. It finds the abnormal transaction with 4 metrics which they can be define as generic features for each business.

## How it works?

There are 3 main processes. !st You can create features. If there is not observed data set, it generates random data set. It is possible to assign a data set with .csv file or data connection. Data of file path must be assigned at configs.py data_path. It genreated features from the feature.json file. All information how to generates are assigned at feature.json file. 2ndYou can tran with features set with Isolation Forest, AutoEncoder (Multivariate - Univariate), LSTM - AutoEncoder. 3rd, it is possible to show results with dashbord
1st generate random data and create features: main.py feature_engineering all
2nd train generated data set: main.py train_process 0
3rd show dashboard data set: main.py dashboard 0 

## Metrics
# 1. Customer Of Merchant Transactions Counts:
 If a transaction occurs from a merchant that the customer has no transaction from, it would be abnormal transaction.
# 2. Transaction of frequencies of each customer:
For instance, let's saya customer has transaction of each month suddenlt if the customer use the card for every hour this would abnormal transactions
# 3. Sudden drop and peak of each customer - merchant of amounts on each transaction:
It can assume that each customer - merchant can not be huge drop or increase suddenly. This would be abnormal transaction

## Features
In order to add feature to project, feature.json file must be updated.
**calling:** Each feature needs a function to be calculated. Each function must be added to data_manipulation.py. Calling key of this dictionary allows us to use for calculating the related feature. 
**noisy_data_remover:** FRor each feature raw data set of noisy data removing separately. Data removing function must be added to data_manipulation.py
**using_normalization:** If we need to use normalization for the feature assign True. If this field is Tru make sure 'is_min_max_norm' is True depending on Min - Max Normalization or - Gaussian P Value Normalization
### 1. c_m_ratios
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
clustered_merchant_ratios_feature: First, This function starts with clustering merchants into the categories. If we have categories of merchants as like -commerce, retail, Goverment, retail, we don't need to cluster the merhants. 

### *  Merchant Clustering (RFM)
In order to label the merchants, it better using metrics of one of the popular segmentation technique which is RFM. Recenecy how recently a merchant has transactions, monetary, what the aveage spent on the merchant, how frequently the merchant has transactions

![Image description](link-to-image)

### * How is the c_m_ratios calculated?
After labeling merchants total merchant label  total transactions of each customer has been calculated. 
ratio  = customer-merchant total transaction / customer-merchant label total transaction

### 2. c_freq_diff
This feature refers to 2nd metric. In order to calculate this metrics exploratory analysis will help us to see how can this feature is distributed. 
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

### * How is the c_m_ratios calculated?
**-** Every customer of transactions hourly differences are calculated. 
**-** Each customer of average frequency is calculated. 
**-** Each Transaction Hourly Difference from the average daily differences are calculated. Negative Differences as their values, however we are ignoring (+) and assigning them 0. Beacuse if a customer is not engaged to online card enviroment, it is posibity of Fraud will decrease.
**-** Now, it is time to Normalize these values into the range 0 - 1. 







### Data Sets (Random Data Generator)


### Parameters
One of them is parameter of probabilty which is P(Q) and it has Beta Distribution. 

### 2. P(X | Q)
The another term is P(X | Q). 
With given X data conditon, probability of appearing Q parameters. 
Let`s explain this with an coin toasting;
- X will be Binary (Head or Tails). ratio will be 1/2 for each instance. Q = 1/2.
- We toast the coin 5 times each test of toasting coins will have to outputs which are head or tail. 
- X = [head, head, head, tails, tails]. What is the probability of each instance of Toasting coin?
- The answer is X has Bernolli Distribution. P(X | Q) will be the Bernolli Probabilty Density Fonction (PDF)
Let`s we implement an A/B Test on web page of user Interface of a button of color.
- First user land your page with test data set. New Developed feature has seen the user. 
- Click!!!
- Bayesian approach our parameters are updates with a = click count, b = non-click count. In This case a = 1, b = 0
- Now, We need to calculate X given data set what is the Q parameter. With Beyes Theorem, it is possible.

### Assumption Of Distributions:
The assumption of Bayesian Approach is related to priors and posteriors. With given parameter, sample of data X of distribution will be Bernolli Distribution (P(X | Q)). This trerm is Bernolli Distributed. It has combinotion of two choices of Q parameter "0 or 1", "True or False". Another term which is parameter of estimated value (P(Q)) when it is sampled many times, Max,mum likelihood Theorem will be worked at it will shape as Beta Distribution. P(Q) values provide two main assumption of beta distribution 1st) Every each P(Q) values are independent. 2nd) P(Q) values are distributed around [0,1]. 
Bernolli And Beta Distribution are going to be the Priors.
Multiplication of Bernolli And Beta Distribution are going to be the Posteriors and it will be Beta Distributed when Likelihood Estimation is worked.
