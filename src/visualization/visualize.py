import re
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#% matplotlib inline
# Plot the data:distribution of categories
def plot_categories(counts ,labels):
    # counts=resume_data['Category'].value_counts()
    # labels=resume_data['Category'].unique()
    plt.figure(figsize=(15, 7))
    plt.bar(labels, counts)
    plt.xlabel('Category')
    plt.ylabel('number of resume')
    plt.xticks(rotation=45)
    plt.title('Distribution of categories')
    plt.show()
#PIe Plot the data :Resume Categories
def piePlot_categories(counts,labels):
    # counts=resume_data['Category'].value_counts()
    # labels=resume_data['Category'].unique()
    # Create a color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    # Create pie plot
    plt.figure(figsize=(12, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=colors, startangle=180)
    # Add a circle at the center to simulate depth perception
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    plt.title("Resume Categories", fontsize=16)
    plt.show()
def wordfreq(data ,df):
    count = data['Resume'].str.split(expand=True).stack().value_counts().reset_index()
    count.columns = ['Word', 'Frequency']

    return count.head(10)
def plot_wordFreq_by_categories(data):
    categories = np.sort(data['Category'].unique())
    df_categories = [data[data['Category'] == category].loc[:, ['Resume', 'Category']] for category in categories]
    fig = plt.figure(figsize=(32, 64))
    for i, category in enumerate(categories):
        wf = wordfreq(data,df_categories[i])

        fig.add_subplot(6, 4, i + 1).set_title(category)
        plt.bar(wf['Word'], wf['Frequency'])
        plt.ylim(0, 3500)

    plt.show()
    plt.close()
def piPlot_wordFreq_by_categories(data):
    categories = np.sort(data['Category'].unique())
    df_categories = [data[data['Category'] == category].loc[:, ['Resume', 'Category']] for category in categories]
    for i, category in enumerate(categories):
        wf = wordfreq(data,df_categories[i])
        fig = plt.figure(figsize=(15, 17))
        fig.add_subplot(5, 5, i + 1)
        plt.title(category)
        plt.pie(wf['Frequency'],labels= wf['Word'], autopct='%1.1f%%',shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
        plt.axis('equal')  
        plt.show()

# Visualize the confusion matrix using plot_confusion_matrix
def confusion_matrix_model(y_test,y_pred,model):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)

    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix ')
    plt.show()

# resume=pd.read_csv('data/interim/intermediate_Resume.csv')#data/interim/intermediate_Resume.csv
# counts=resume['Category'].value_counts()
# labels=resume['Category'].unique()
# plot_categories(counts,labels)