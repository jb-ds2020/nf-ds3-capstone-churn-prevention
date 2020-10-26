"""
use this as im ported libary
import eda_methods as eda
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def meta(df, transpose=True):
    """
    This function returns a dataframe that lists:
    - column names
    - nulls abs
    - nulls rel
    - dtype
    - duplicates
    - number of diffrent values (nunique)
    """
    metadata = []
    dublicates = sum([])
    for elem in df.columns:

        # Counting null values and percantage
        null = df[elem].isnull().sum()
        rel_null = round(null/df.shape[0]*100, 2)

        # Defining the data type
        dtype = df[elem].dtype

        # Check dublicates
        duplicates = df[elem].duplicated().any()

        # Check number of nunique vales
        nuniques = df[elem].nunique()


        # Creating a Dict that contains all the metadata for the variable
        elem_dict = {
            'varname': elem,
            'nulls': null,
            'percent': rel_null,
            'dtype': dtype,
            'dup': duplicates,
            'nuniques': nuniques
        }
        metadata.append(elem_dict)

    meta = pd.DataFrame(metadata, columns=['varname', 'nulls', 'percent', 'dtype', 'dup', 'nuniques'])
    meta.set_index('varname', inplace=True)
    meta = meta.sort_values(by=['nulls'], ascending=False)
    if transpose:
        return meta.transpose()
    print(f"Shape: {df.shape}")

    return meta

def data_loss(df_clean, df_raw):
    """
    This function returns the data loss in percent.
    """
    return f"{round((df_clean.shape[0]/df_raw.shape[0])*100,3)}% data loss"

def describe_plus(df, transpose=True):
    """
    This function returns a dataframe based on describ() function added:
    - skew()
    - kurtosis()
    - variance
    """
    statistics = pd.DataFrame(df.describe())
    skew       = pd.Series(df.skew())
    kurtosis   = pd.Series(df.kurtosis())
    variance   = pd.Series(df.var())

    statistics.loc['skew'] = skew
    statistics.loc['kurtosis'] = kurtosis
    statistics.loc['variance'] = variance

    if transpose:
        return round(statistics.transpose(), 2)
    return round(statistics, 2)

def correlogram(df):
    """
    This function plots a correlogram.
    """
    #Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    mask = np.triu(df.corr())
    ax = sns.heatmap(round(df.corr()*100, 0),
                     annot=True,
                     mask=mask, cmap="coolwarm")
    return df.corr()

def plot_train_test_split(y, y_train, y_test):
    """
    This function plots the the sizes of training and test set.
    Also you will get a dataframe with the number of values and the relative distribution.
    """
    # plot
    y.plot.hist()
    y_train.plot.hist()
    y_test.plot.hist()

    # dataframe with relative and absolut values
    plt.legend(['all', 'train', 'test'])
    storage = pd.DataFrame()
    storage['train abs'] = round(y_train.value_counts(), 2)
    storage['train %']   = round((y_train.value_counts()/y_train.shape[0]), 2)
    storage['test abs']  = round(y_test.value_counts(), 2)
    storage['test %']    = round((y_test.value_counts()/y_test.shape[0]), 2)
    storage['all abs']   = round(y.value_counts(), 2)
    storage['all %']     = round((y.value_counts()/y.shape[0]), 2)

    # prints informations about splits
    print ("Training set has {} samples.".format(y_train.shape[0]))
    print ("Testing set has {} samples.".format(y_test.shape[0]))
    return storage
