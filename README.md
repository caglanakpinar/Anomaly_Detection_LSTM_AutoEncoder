# A/B Test Bayesian Approaches
### Overview
### Metrics
### Features
### Dependencies
### Data Sets (Random Data Generator)
### Parameters
### How it works?
### Example
### Ensamble Model (Isolation Forest)
### Deep Learning Model (AutoEncoder)
### Deep Learning Model (LSTM - AutoEncoder)
### Visualizations

## Overview

You only have customer, transaction, merchant of unique ids and transactşon date and and transactions of Amounts. How can you define the Anomalities of each Transaction? This process allows us to find the abnomalities with deciding metrics. This project can be integrated any business which has transactions with amount and date and customers and merchants. It finds the abnormal transaction with 4 metrics which they can be define as generic features for each business.

### Metrics
1. 
2. 
3. 
4. 


### Features



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
