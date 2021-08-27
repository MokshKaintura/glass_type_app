
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache
def prediction(model,ri,na,mg,al,si,k,ca,ba,fe):
  glass_type = model.predict([[ri,na,mg,al,si,k,ca,ba,fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return "building windows float processed".upper()
  elif glass_type == 2:
    return "building windows non float processed".upper()
  elif glass_type == 3:
    return "vehicle windows float processed".upper()
  elif glass_type == 4:
    return "vehicle windows non float processed".upper()
  elif glass_type == 5:
    return "containers".upper()
  elif glass_type == 6:
    return "TABLEWARE"
  else:
    return 'HEADLAMPS'
st.title('Glass type predictor')
st.sidebar.title('Exploratory Data Analysis')
if st.sidebar.checkbox('Show raw data'):
  st.subheader('Full Dataset')
  st.dataframe(glass_df)
st.sidebar.subheader("scatter plot")
features_list = st.sidebar.multiselect('Select the x axis values:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse',False)
for i in features_list:
  st.subheader(f"scatter plot between {i} and glass_type")
  plt.figure(figsize=(12,6),dpi=100)
  sns.scatterplot(x=i,y='GlassType',data=glass_df)
  st.pyplot()
st.sidebar.subheader('histogram')
hist_feateres = st.sidebar.multiselect('select features to create histograms:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in hist_feateres:
    st.subheader(f"histogram for {i}")
    plt.figure(figsize=(15,5),dpi=100)
    plt.hist(glass_df[i],bins='sturges',edgecolor='black')
    st.pyplot()
st.sidebar.subheader('boxplot')
box_feateres = st.sidebar.multiselect('select features to create boxplot:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in box_feateres:
    st.subheader(f"boxplot for {i}")
    plt.figure(figsize=(15,5),dpi=100)
    sns.boxplot(glass_df[i])
    st.pyplot()