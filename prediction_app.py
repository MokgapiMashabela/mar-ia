import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import pandas as pd

loaded_model = pickle.load(open('C:/VS Code\VS code for DAA assigmnent/trained_model.sav', 'rb'))

with st.sidebar:
    
    selected = option_menu(       
                           ["Heart Prediction System"],
                           ["Predict to See Graphs"],
                           icons=["heart"],
                           default_index=0)

#Heart Disease Prediction Page

if (selected == "Prediction"):

        st.title('Heart Disease Prediction Web Application')
# page title
st.title('Heart Disease Prediction using ML')

#getting input from user
col1, col2, col3 = st.columns(3)
with col1:
        age = st.text_input('Age')

with col2:
        sex = st.text_input('Sex')

with col3:
        cp = st.text_input('Chest Pain types')

with col1:
        trestbps = st.text_input('Resting Blood Pressure')

with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')

with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')

with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')

with col3:
        exang = st.text_input('Exercise Induced Angina')

with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

with col1:
        thal = st.text_input('Status of heart (thal): 0 = normal; 1 = fixed defect; 2 = reversable defect')
      
# code for Prediction
heart_diagnosis = ''

#button for Prediction
if st.button('Prediction Result'):

                user_input= [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
   
                user_input = [float(x) for x in user_input]
        
                if loaded_model:
                        heart_prediction = loaded_model.predict([user_input])

                if heart_prediction[0] == 0:
                        heart_diagnosis = 'The person does not have any heart disease'
                else:
                        heart_diagnosis = 'The person is having a heart disease'
        
st.success(heart_diagnosis)

#Explore Page
df = pd.read_csv("C:/VS Code/VS code for DAA assigmnent\heart.csv",header = 0,sep=';')

st.title('Graphs')
# page title
st.title('Categorical and Numerical Variables')

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


#Numerical Variables
st.write("Distribution Of Numerical Variables By Target Variable:")

st.pyplot(fig)

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

st.write("Distribution Of Numerical Variables By Target Variable:")

st.pyplot(fig)
