#!/usr/bin/env python

"""Churn predictor can be used to predict churn of subscribers from a given csv. file.

Input:

Output:

Requirements:

    files:
    - trained_models/stacking_CVC.pckl
    - trained_models/scaler.pckl

__author__ = "Carlotta Ulm, Jonas Bechthold, Silas Mederer"
__version__ = "0.1"
__maintainer__ = "+++"
__email__ = "bechthold.jonas@gmail.com"
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
    import math
    import itertools
    import joblib
    from time import time
    from datetime import datetime
    # metrics
    from sklearn.metrics import fbeta_score, accuracy_score, f1_score, recall_score, precision_score
    from pickle import load
    # plot for output
    import matplotlib.pyplot as plt
except ImportError:
    print("One or more modules needs to be installed. Please check README")
    pass


######################################
#TO DO!!!                            #
######################################
# input path
#PATH = str(input("Copy path here")

# load dataset
df = pd.read_csv('data/f_chtr_churn_traintable_nf_2.csv')

######################################
#TO DO!!!                            #
######################################
# check of dataframe or csv file

# drop not used columns
df = df.drop(["Unnamed: 0", "auftrag_new_id"], axis=1)

# remove major customer with more than 4 subscriptions per household
df = df[df.cnt_abo < 5]

# removing of NaNs
print('df before drop',df.shape)
# does not work drops all non kuendigungseingangsdatum, check over columns with missings
df = df.dropna(subset=['ort','email_am_kunden'])
print('df after drop',df.shape)

# datetime conversion
df['abo_registrierung_min'] = pd.to_datetime(df['abo_registrierung_min'])
df['nl_registrierung_min']  = pd.to_datetime(df['nl_registrierung_min'], format='%Y-%m-%d')
#df['date_x'] = pd.to_datetime(df['date_x'], format='%Y-%m-%d')
#df['date_x'] = df['date_x'].dt.date
#df['kuendigungs_eingangs_datum'] = pd.to_datetime(df['kuendigungs_eingangs_datum'],errors='coerce',format='%Y-%m-%d')
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

# cnt is the number of subscribtions the contract holds (families, libaries etc.)
df_cnt = df[['cnt_abo', 'cnt_abo_diezeit', 'cnt_abo_diezeit_digital',
       'cnt_abo_magazin', 'cnt_umwandlungsstatus2_dkey']]               

# Newsletter information
df_nl = df[['nl_zeitbrief', 'nl_zeitshop', 'nl_zeitverlag_hamburg',
       'nl_fdz_organisch', 'nl_blacklist_sum', 'nl_bounced_sum',
       'nl_aktivitaet', 'nl_registrierung_min', 'nl_sperrliste_sum',
       'nl_opt_in_sum']]

# newsletter drop technical details
df_nl.drop(["nl_blacklist_sum", "nl_bounced_sum", "nl_sperrliste_sum", "nl_opt_in_sum", "nl_fdz_organisch", "nl_registrierung_min"], axis=1, inplace=True)
# newsletter interactions
df_reg = df[['boa_reg', 'che_reg', 'sit_reg', 'sso_reg']]
               
# newsletter existing customers without rates
df_nl_bestandskunden_1 = df[['received_anzahl_bestandskunden_1w',
       'received_anzahl_bestandskunden_1m',
       'received_anzahl_bestandskunden_3m',
       'received_anzahl_bestandskunden_6m', 'opened_anzahl_bestandskunden_1w',
       'opened_anzahl_bestandskunden_1m', 'opened_anzahl_bestandskunden_3m',
       'openedanzahl_bestandskunden_6m', 'clicked_anzahl_bestandskunden_1w',
       'clicked_anzahl_bestandskunden_1m', 'clicked_anzahl_bestandskunden_3m',
       'clicked_anzahl_bestandskunden_6m',
       'unsubscribed_anzahl_bestandskunden_1w',
       'unsubscribed_anzahl_bestandskunden_1m',
       'unsubscribed_anzahl_bestandskunden_3m',
       'unsubscribed_anzahl_bestandskunden_6m']]

# productnews (kind of newsletter but more commercial)without rates
df_nl_produktnews_1 = df[['received_anzahl_produktnews_1w', 'received_anzahl_produktnews_1m',
       'received_anzahl_produktnews_3m', 'received_anzahl_produktnews_6m',
       'opened_anzahl_produktnews_1w', 'opened_anzahl_produktnews_1m',
       'opened_anzahl_produktnews_3m', 'openedanzahl_produktnews_6m',
       'clicked_anzahl_produktnews_1w', 'clicked_anzahl_produktnews_1m',
       'clicked_anzahl_produktnews_3m', 'clicked_anzahl_produktnews_6m',
       'unsubscribed_anzahl_produktnews_1w',
       'unsubscribed_anzahl_produktnews_1m',
       'unsubscribed_anzahl_produktnews_3m',
       'unsubscribed_anzahl_produktnews_6m']]   

# newsletter region hamburg without rates
df_nl_hamburg_1 = df[['received_anzahl_hamburg_1w', 'received_anzahl_hamburg_1m',
       'received_anzahl_hamburg_3m', 'received_anzahl_hamburg_6m',
       'opened_anzahl_hamburg_1w', 'opened_anzahl_hamburg_1m',
       'opened_anzahl_hamburg_3m', 'openedanzahl_hamburg_6m',
       'clicked_anzahl_hamburg_1w', 'clicked_anzahl_hamburg_1m',
       'clicked_anzahl_hamburg_3m', 'clicked_anzahl_hamburg_6m',
       'unsubscribed_anzahl_hamburg_1w', 'unsubscribed_anzahl_hamburg_1m',
       'unsubscribed_anzahl_hamburg_3m', 'unsubscribed_anzahl_hamburg_6m']] 

# newsletter zeitbrief without rates
df_zb_1 = df[['received_anzahl_zeitbrief_1w', 'received_anzahl_zeitbrief_1m',
       'received_anzahl_zeitbrief_3m', 'received_anzahl_zeitbrief_6m',
       'opened_anzahl_zeitbrief_1w', 'opened_anzahl_zeitbrief_1m',
       'opened_anzahl_zeitbrief_3m', 'openedanzahl_zeitbrief_6m',
       'clicked_anzahl_zeitbrief_1w', 'clicked_anzahl_zeitbrief_1m',
       'clicked_anzahl_zeitbrief_3m', 'clicked_anzahl_zeitbrief_6m',
       'unsubscribed_anzahl_zeitbrief_1w', 'unsubscribed_anzahl_zeitbrief_1m',
       'unsubscribed_anzahl_zeitbrief_3m', 'unsubscribed_anzahl_zeitbrief_6m']]                



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
for i in df_nl_bestandskunden_1:
    df[i] = df[i].apply(flatten_greater_0)

for i in df_nl_produktnews_1:
    df[i] = df[i].apply(flatten_greater_0)

for i in df_nl_hamburg_1:
    df[i] = df[i].apply(flatten_greater_0)

for i in df_zb_1:
    df[i] = df[i].apply(flatten_greater_0)

# rename
df.rename(columns={'openedanzahl_bestandskunden_6m': 'opened_anzahl_bestandskunden_6m',
                   'openedanzahl_produktnews_6m': 'opened_anzahl_produktnews_6m',
                   'openedanzahl_hamburg_6m': 'opened_anzahl_hamburg_6m',
                   'openedanzahl_zeitbrief_6m': 'opened_anzahl_zeitbrief_6m'}, inplace=True)

name = ['received_anzahl', 'opened_anzahl', 'clicked_anzahl', 'unsubscribed_anzahl']
art = ['bestandskunden','produktnews','hamburg','zeitbrief']
zeitraum = ['1w', '1m', '3m', '6m']
titel = ['nl_received_1w', 'nl_received_1m', 'nl_received_3m', 'nl_received_6m', 'nl_opened_1w', 'nl_opened_1m', 'nl_opened_3m',
        'nl_opened_6m','nl_clicked_1w', 'nl_clicked_1m', 'nl_clicked_3m', 'nl_clicked_6m', 'nl_unsubscribed_1w', 'nl_unsubscribed_1m',
        'nl_unsubscribed_3m', 'nl_unsubscribed_6m']
links = []
for n in name:
    for z in zeitraum:
        for a in art:
            links.append(n + '_' + a + '_' + z)

for t in titel:
    df[t] = df[links[0]] + df[links[1]] + df[links[2]] + df[links[3]]
    links = links[3:]


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

# dataframe ready for prediction

print(df.columns)

# Min Max Scaler on the initial training set
scaler = load(open('trained_models/scaler.pckl', 'rb'))

df = df[important_features]
df = scale.transform(df)

print('Shape of df',df.shape)
# load model
# votingcf = joblib.load("/trained_models/votingcf.joblib")
#stackingcf = joblib.load("trained_models/stacking_CVC.joblib")

print('Loading Stacking Classifier')
model = load(open('trained_models/stacking_CVC.pckl', 'rb'))

print('Doing the Predictions')
predictions = model.predict(df)
predictions_proba = model.predict_proba(df)[:,1]
print('Prediction Results (0=No Churn, 1=Churn): ',predictions)

predictions_df = pd.DataFrame()
predictions_df["prediction"] = predictions
predictions_df["probability"] = predictions_proba
# save to csv

predictions_df.to_csv("predictions.csv")

_ = plt.hist(predictions, bins='auto')  # arguments are passed to np.histogram
plt.title("Prediction Results")
plt.show()
