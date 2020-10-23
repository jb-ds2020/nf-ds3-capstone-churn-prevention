# Model for churn prevention die Zeit Verlag

**Team:** Jonas Bechthold, Silas Mederer & Carlotta Ulm

**Business Case:** Churn prevention is your proactive strategy for keeping customers around. It involves looking at the underlying reasons for churn and then formulating a plan to combat issues that may lead to churn before they happen.

**Goal:** Beat the Model of ‘Die Zeit Verlag’

**Metric:** F1 and ROC/AUC to compare models 

**Basic idea:** We want to use different ML (supervised and unsupervised) approaches to predict churns of subscriptions of the german newspaper “Die Zeit”. Supervised ML methods can be used to predict (classification methods) subscription churns based on the given dataset. Since a large number of features exists, feature selection as part of an extensive EDA is essential. Unsupervised ML methods could be used first to do clustering on the dataset to identify if there are certain “groups of subscribers”, who share certain features. This clustering could be used to investigate subscriber group specific churn mitigation methods (not only writing emails, as given in the dataset description). 

For the supervised (with label) ML the data imbalance must be handled. The aim is to identify as many as possible “real” subscription churns as possible. Incorrectly identifying “non real” churns is not of the highest priority (Recall vs. Accuracy), since we would recommend churn mitigation methods for these subscribers. The main target is therefore to understand the behavioral patterns of customers and to optimize the churn prevention while reducing the overall cost. 

**Bonus:** Customer lifecycle target groups (cluster), GUI, SQL database

**Methods and Technologies:** Preprocessing (Cleaning, EDA), unsupervised learning for clustering, supervised ML classification, advanced methods consisting of ANN (Artificial Neural Networks), CNN (baseline model BG-NBD MODEL)

