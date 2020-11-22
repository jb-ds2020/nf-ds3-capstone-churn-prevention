"""
Churn predictor can be used to predict churn of subscribers from a given csv. file.

Input: It needs 79 features, a few of them are going to be enginnered, most of them are
used for prediction. The predictions are done by a stacking classifier (scores on test
Recall: 0.772, Precision: 0.611, Accuracy: 0.782, F1: 0.682).

Output: csv file with probabilities.

Requirements:

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

__author__ = "Carlotta Ulm, Jonas Bechthold, Silas Mederer"
__version__ = "0.1"
__maintainer__ = "+++"
__email__ = "bechthold.jonas@gmail.com, mederersilas@gmail.com"
__status__ = "Production"
"""

######################################
#TO DO!!!                            #
######################################

# Documentation
# package dependencies
# system requirements
# directly install the packages
# check input
# model selection


# imports
try:
    import pandas as pd
    import numpy as np
    from time import time
    from datetime import datetime
    from pickle import load
    import os
    print("Finished module import")
except ImportError:
    print("Could not import all models. Please check dependencies")
    pass

# Features to import
zon_features = ['zon_che_opt_in', 'zon_sit_opt_in', 'zon_zp_grey', 'zon_premium',
                'zon_boa', 'zon_kommentar', 'zon_sonstige', 'zon_zp_red', 'zon_rawr',
                'zon_community', 'zon_app_sonstige', 'zon_schach',
                'zon_blog_kommentare', 'zon_quiz']
reg_features = ['boa_reg', 'che_reg', 'sit_reg', 'sso_reg']
nl_features = ['opened_anzahl_bestandskunden_1m', 'opened_anzahl_produktnews_1m',
               'opened_anzahl_hamburg_1m', 'opened_anzahl_zeitbrief_1m',
               'unsubscribed_anzahl_bestandskunden_6m', 'unsubscribed_anzahl_produktnews_6m',
               'unsubscribed_anzahl_hamburg_6m', 'unsubscribed_anzahl_zeitbrief_6m',
               'clicked_anzahl_bestandskunden_3m', 'openedanzahl_bestandskunden_6m',
               'received_anzahl_bestandskunden_6m', 'unsubscribed_anzahl_hamburg_1m',
               'received_anzahl_6m', 'openedanzahl_6m', 'unsubscribed_anzahl_1m',
               'clicked_anzahl_6m', 'unsubscribed_anzahl_6m']
cat_features = ['kanal', 'objekt_name', 'aboform_name', 'zahlung_rhythmus_name',
                'zahlung_weg_name', 'plz_1', 'plz_2', 'land_iso_code',
                'anrede', 'titel']
num_features = ['rechnungsmonat', 'received_anzahl_6m', 'openedanzahl_6m', 'nl_zeitbrief',
                'nl_aktivitaet', 'liefer_beginn_evt', 'cnt_umwandlungsstatus2_dkey',
                'clickrate_3m', 'unsubscribed_anzahl_1m', 'studentenabo',
                'received_anzahl_bestandskunden_6m', 'openrate_produktnews_3m',
                'nl_zeitshop', 'nl_opt_in_sum', 'clicked_anzahl_6m',
                'unsubscribed_anzahl_hamburg_1m', 'unsubscribed_anzahl_6m', 'shop_kauf',
                'openrate_zeitbrief_3m', 'openrate_produktnews_1m', 'openrate_3m', 'openrate_1m',
                'nl_fdz_organisch', 'metropole', 'cnt_abo_magazin', 'cnt_abo_diezeit_digital',
                'cnt_abo', 'clicked_anzahl_bestandskunden_3m']
time_features = ['abo_registrierung_min', 'nl_registrierung_min', 'liefer_beginn_evt']
id_marker = ['auftrag_new_id']

features = id_marker + zon_features + reg_features + cat_features + num_features + time_features + nl_features



# import data
print('Import Data')
filename = input("Ihre Eingabe? ")
PATH = os.path.abspath("scribt.ipynb").replace("scribt.ipynb", "")
print(f"Path: {PATH}")
print(f"Filename: {filename}")

# load dataset
try:
    df = pd.read_csv(PATH + filename, usecols=features)
except NameError:
    print(f"Column names might changed. The following columns must be included: {features}")

######################################
#TO DO!!!                            #
######################################
# check of data types and values

# create auftrag_new_id df
auftrag_new_id = df.auftrag_new_id

# remove major customer
df = df[df.cnt_abo < 5]

# hanlde missings (drop)
size_0 = df.shape[0]
df = df.dropna()
size_1 = df.shape[0]
print(f"{size_0-size_1} customers have been deleted due to missings.")

print('Converting Data')

# datetime conversion
df['abo_registrierung_min'] = pd.to_datetime(df['abo_registrierung_min'])
df['nl_registrierung_min']  = pd.to_datetime(df['nl_registrierung_min'], format='%Y-%m-%d')
df['liefer_beginn_evt']  = pd.to_datetime(df['liefer_beginn_evt'], format='%Y-%m-%d')
df['liefer_beginn_evt'] = df['liefer_beginn_evt'].map(lambda x: x.year + x.month/12.0)
df['MONTH_DELTA_abo_min'] = (df.abo_registrierung_min - df.abo_registrierung_min.min()).dt.days
df['MONTH_DELTA_abo_min'] = df['MONTH_DELTA_abo_min'].map(lambda x: x/30)
df['MONTH_DELTA_nl_min'] = (df.nl_registrierung_min - df.nl_registrierung_min.min()).dt.days
df['MONTH_DELTA_nl_min'] = df['MONTH_DELTA_nl_min'].map(lambda x: x/30)

# zones are special areas that need registration
df_zon = df[['zon_che_opt_in', 'zon_sit_opt_in', 'zon_zp_grey', 'zon_premium',
       'zon_boa', 'zon_kommentar', 'zon_sonstige', 'zon_zp_red', 'zon_rawr',
       'zon_community', 'zon_app_sonstige', 'zon_schach',
       'zon_blog_kommentare', 'zon_quiz']]

# newsletter interactions
df_reg = df[['boa_reg', 'che_reg', 'sit_reg', 'sso_reg']]

# newsletter opened 1m
nl_opened = ['opened_anzahl_bestandskunden_1m', 'opened_anzahl_produktnews_1m', 'opened_anzahl_hamburg_1m', 'opened_anzahl_zeitbrief_1m']

# newsletter unsubscribed 6m
nl_unsubscribed = ['unsubscribed_anzahl_bestandskunden_6m','unsubscribed_anzahl_produktnews_6m',  'unsubscribed_anzahl_hamburg_6m',
 'unsubscribed_anzahl_zeitbrief_6m']

# Other newsletter features
nl_to_flat = ['clicked_anzahl_bestandskunden_3m', 'openedanzahl_bestandskunden_6m', 'received_anzahl_bestandskunden_6m', 'unsubscribed_anzahl_hamburg_1m']

print('Engineering Features')

# engineering functions
def flatten_greater_1(flat):
    if flat > 1:
        return 1
    else:
        return 0

def flatten_greater_0(flat):
    if flat > 0:
        return 1
    else:
        return 0

# flatten, sum and join zon
for i in df_zon:
    df[i] = df[i].apply(flatten_greater_1)
sum_zon = df_zon.sum(axis=1)
sum_zon = sum_zon.to_frame(name="sum_zon")
df = df.join(sum_zon)

# sum and join reg
sum_reg = df_reg.sum(axis=1)
sum_reg = sum_reg.to_frame(name="sum_reg")
df = df.join(sum_reg)

# newletter engineering flatting
for i in nl_opened:
    df[i] = df[i].apply(flatten_greater_0)
for i in nl_unsubscribed:
    df[i] = df[i].apply(flatten_greater_0)
for i in nl_to_flat:
    df[i] = df[i].apply(flatten_greater_0)

# rename columns
df.rename(columns={'openedanzahl_bestandskunden_6m': 'opened_anzahl_bestandskunden_6m'}, inplace=True)

# Create new columns
df['nl_unsubscribed_6m'] = df['unsubscribed_anzahl_bestandskunden_6m'] + df['unsubscribed_anzahl_produktnews_6m'] + df['unsubscribed_anzahl_hamburg_6m'] + df['unsubscribed_anzahl_zeitbrief_6m']

df['nl_opened_1m'] = df['opened_anzahl_bestandskunden_1m'] + df['opened_anzahl_produktnews_1m'] + df['opened_anzahl_hamburg_1m'] + df['opened_anzahl_zeitbrief_1m']

# get get_dummies
df = pd.get_dummies(df, columns = ['kanal', 'objekt_name', 'aboform_name', 'zahlung_rhythmus_name',
                                   'zahlung_weg_name', 'plz_1', 'plz_2', 'land_iso_code',
                                   'anrede','titel'], drop_first=False)

# 51 most important features result of selection
important_features =['zahlung_weg_name_Rechnung',
                     'zahlung_rhythmus_name_halbjährlich',
                     'rechnungsmonat',
                     'received_anzahl_6m',
                     'openedanzahl_6m',
                     'objekt_name_ZEIT Digital',
                     'nl_zeitbrief',
                     'nl_aktivitaet',
                     'liefer_beginn_evt',
                     'cnt_umwandlungsstatus2_dkey',
                     'clickrate_3m',
                     'anrede_Frau',
                     'aboform_name_Geschenkabo',
                     'unsubscribed_anzahl_1m',
                     'studentenabo',
                     'received_anzahl_bestandskunden_6m',
                     'openrate_produktnews_3m',
                     'opened_anzahl_bestandskunden_6m',
                     'objekt_name_DIE ZEIT - CHRIST & WELT',
                     'nl_zeitshop',
                     'nl_opt_in_sum',
                     'nl_opened_1m',
                     'kanal_andere',
                     'kanal_B2B',
                     'clicked_anzahl_6m',
                     'che_reg',
                     'MONTH_DELTA_nl_min',
                     'zon_zp_red',
                     'zahlung_rhythmus_name_vierteljährlich',
                     'unsubscribed_anzahl_hamburg_1m',
                     'unsubscribed_anzahl_6m',
                     'sum_zon',
                     'sum_reg',
                     'shop_kauf',
                     'plz_2_10',
                     'plz_1_7',
                     'plz_1_1',
                     'openrate_zeitbrief_3m',
                     'openrate_produktnews_1m',
                     'openrate_3m',
                     'openrate_1m',
                     'nl_unsubscribed_6m',
                     'nl_fdz_organisch',
                     'metropole',
                     'cnt_abo_magazin',
                     'cnt_abo_diezeit_digital',
                     'cnt_abo',
                     'clicked_anzahl_bestandskunden_3m',
                     'aboform_name_Probeabo',
                     'aboform_name_Negative Option',
                     'MONTH_DELTA_abo_min']


# Min Max Scaler on the initial training set
scaler = load(open('trained_models/scaler.pckl', 'rb'))

df = df[important_features]
df = scaler.transform(df)

print('Shape of df',df.shape)


df = df[important_features]
df = scaler.transform(df)

# load choosen classifier
if classifier == 1:
    print('Loading Stacking Classifier')
    model = load(open(PATH + 'trained_models/stacking_CVC.pckl', 'rb'))
if classifier == 2:
    print('Loading Voting Classifier')
    model = load(open(PATH + 'trained_models/votingcf.pckl', 'rb'))

# predict
print('Doing the Predictions')
predictions = model.predict(df)
predictions_proba = model.predict_proba(df)[:,1]
print('Prediction Results (0=No Churn, 1=Churn): ',predictions)

# save to df
predictions_df = pd.DataFrame()
predictions_df["auftrag_new_id"] = auftrag_new_id
predictions_df["prediction"] = predictions
predictions_df["probability"] = predictions_proba.round(3)

# save to csv
predictions_df.to_csv("predictions.csv")

try:
    import matplotlib.pyplot as plt
    _ = plt.hist(predictions, bins='auto')  # arguments are passed to np.histogram
    plt.title("Prediction Results")
    plt.show()

except ImportError:
    print("Plot can not be showed. Missing matplotlib or numpy.")
    pass
