import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from purifytext import clean_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

def predict(resume):
    try:
        if(os.path.exists('UpdatedResumeDataSet.csv')):
            print("File found!")
            df=pd.read_csv('UpdatedResumeDataSet.csv')
            # print(df.shape)
            # Corrected bar plot
            # category_counts = df['Category'].value_counts()
            # sns.barplot(x=category_counts.index, y=category_counts.values)
            # plt.xticks(rotation=90)
            # plt.title("Category Distribution")
            # plt.xlabel("Categories")
            # plt.ylabel("Count")
            # plt.show()
            
            df=clean_text(dataframe=df,column_name='Resume') #cleans the dataframe i.e cleaning the excel data.
            CORPUS = np.array(df['Resume']) #stores the Column 'Resume' in an array
            # print(CORPUS.size)
            y=df['Category']
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(CORPUS, y, test_size=0.2, random_state=42)
            pipeline_lr = Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("model", LogisticRegression())])
            pipeline_lr.fit(X_train, y_train)
            # y_train_pred = pipeline_lr.predict(X_train)
            # y_test_pred = pipeline_lr.predict(X_test)
            # def prediction(resume):
            resume_df = pd.DataFrame([resume], columns=["Resume"])
            clean_df = clean_text(dataframe=resume_df, column_name="Resume")
            CORPUS = np.array(clean_df["Resume"])
            pred = pipeline_lr.predict(CORPUS)
            return label_encoder.inverse_transform(pred)[0]

        else:
            print("File not found.")
            
    except Exception as e:
        print(str(e))
    
