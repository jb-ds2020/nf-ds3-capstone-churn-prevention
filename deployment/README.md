# Zeitverlag churn predictor

author = "Carlotta von Ulm-Erbach, Jonas Bechthold, Silas Mederer"
version = "0.1"
maintainer = "+++"
status = "Production"


### How to use:
 1. Put the datafile you want to run in the directory of the script. It has to be a .csv file (comma separated).
 2. Start your terminal and navigate to the directory of the files (datafile & script).
 3. Start the script by by calling "python3 churn-predictor.py" you need to do this via the terminal.
 4. Follow the instructions:
4.1 Input filename (name and extension)
4.2 Choose the classifier (we recommend Stacking): 1=Stacking or 2=Voting
5. Have a look on the outputs to see what happened.
6. You will find a new .csv file after prediction with the columns "auftrag_new_id", "prediction" and "probability". We recommend to contact all customers with a prediction of 1, cause the model predicts they will churn. The column "probability" is just an extra output for you to see the performance.
7. After the program finished, it will show you a plot of the prediction distribution.

If you have any problems or further questions contact us.

![picture by: Lending Club 2020](script-screenshot.png)


### Churn predictor can be used to predict churn of subscribers from a given csv. file.
**Input:** It needs 79 features, a few of them are going to be engineered, most of them are used for prediction. The predictions are done by a stacking classifier (scores on test Recall: 0.772, Precision: 0.611, Accuracy: 0.782, F1: 0.682).
**Output:** csv file with probabilities.
**Requirements:**

    modules:
    - pandas
    - numpy
    - pickle
    - datetime
    - time

    files:
    - trained_models/stacking_CVC.pckl
    - trained_models/votingcf.pckl
    - trained_models/scaler.pckl

If you miss any of the requirements, please contact your admin.
