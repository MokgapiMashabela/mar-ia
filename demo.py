import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns #allows us to plot graphs

#Connection to database
heart_database = sqlite3.connect("assignment.db")

cursor = heart_database.cursor()

data = """
    CREATE TABLE heart(
        age INTEGER,
        sex INTEGER,
        cp INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        fbs INTEGER,
        restecg INTEGER,
        thalach INTEGER,
        exang INTEGER,
        oldpeak FLOAT,
        slope INTEGER,
        ca INTEGER,
        thal INTEGER,
        target INTEGER

        )"""

cursor.execute(data)
print("Database has been created")

#Load data file
df = pd.read_csv("heart.csv",header = 0,sep=';')

print(df.head(5))

#Load data file to SQLite
df.to_sql('heart', heart_database, if_exists='replace')

for index in range(len(df)):
        cursor.execute("INSERT INTO heart(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(df['age'].loc[index],df['sex'].loc[index],df['cp'].loc[index],df['trestbps'].loc[index],df['chol'].loc[index],df['fbs'].loc[index],df['restecg'].loc[index],df['thalach'].loc[index],df['exang'].loc[index],df['oldpeak'].loc[index],df['slope'].loc[index],df['ca'].loc[index],df['thal'].loc[index],df['target'].loc[index]))

#Close connection
cursor.close()

#Q2.1A
#data cleaning
print(df.head(5))
print(df.tail(5))
print(df.columns)

print(df.shape)
print(df.dtypes)

print(df.isna())
print(df.isna().sum())

print(df.hist())

#Q2.1B
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15, 10))
axes = axes.flatten()
for i, column in enumerate(categorical_columns):
        sns.countplot(x=column, hue='target', data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {column} variable by Target variable')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('count')

plt.tight_layout()
plt.show()

#Q2.1C
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15, 10))
axes = axes.flatten()
for i, column in enumerate(numerical_columns):
        sns.boxplot(x='target', y=column, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {column} variable by Target variable')
        axes[i].set_xlabel('Target')
        axes[i].set_ylabel(column)

plt.tight_layout()
plt.show()

#Q3
#This project will choose between RandomForestClassifier, KNeighborsClassifier, and LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df= pd.read_csv("heart.csv",header = 0,sep=';')
print(df.info())
print(df.describe())

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')

sns.set_style('whitegrid')
sns.countplot(x='target', data=df, palette='RdBu_r')

#Data prepocessing
#converting the categorical variables into dummy variables for scaling
dataset = pd.get_dummies(df, columns= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #standard scaler is used since there are features with alot of variations and have diff units, will help us be able to scale up or down
StandardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = StandardScaler.fit_transform(dataset[columns_to_scale])

print(dataset.head())
#splitting the features and target
Y=df['target']
X=df.drop(['target'], axis = 1)
print(X)
print(Y)

from sklearn.model_selection import cross_val_score
#k Nearest Neighbour 
knn_scores = []
for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        score=cross_val_score(knn_classifier, X,Y,cv=10)
        knn_scores.append(score.mean())

plt.plot([k for k in range(1,21)], knn_scores, color = 'green')
for i in range(1,21):
        plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.title('K Neighbors Classifier scores for the diff K values')

knn_classifier = KNeighborsClassifier(n_neighbors=12)
score=cross_val_score(knn_classifier, X,Y,cv=10)

score.mean()

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier, X,Y, cv=10)

score.mean()

from sklearn.model_selection import train_test_split

#Splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

#model evaluation
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data: ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data: ', test_data_accuracy)

print(df['target'].value_counts())

print(df.head())
print(df.shape)


input_data = (57,0,1,130,236,0,0,174,0,0,1,1,2)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
        print("The person does not have any heart disease")
else:
        print("The person has a heart disease")


#saving trained model
import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (57,0,1,130,236,0,0,174,0,0,1,1,2)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
        print("The person does not have any heart disease")
else:
        print("The person has a heart disease")
