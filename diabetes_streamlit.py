import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('D:/environments/Streamlit_diabities/train_model.sav','rb'))


def diabetes_prediction(input_data):
    scaled = StandardScaler()
    #input_data = (4,110,95,0,0,37,0.191,30)
    input_data_as_array = np.array(input_data)
    input_data_reshape = input_data_as_array.reshape(1,-1)
    std_scaled = scaled.fit_transform(input_data_reshape)
    prediction = model.predict(std_scaled)

    if (prediction[0] == 0):
        return 'The Prediction is true the person is "Diabetic"'
    else:
       return 'The Prediction is true the person is "non Diabetic"'

#input_data = (4,110,95,0,0,37,0.191,30)  
#diabetes_prediction(input_data=input_data)


def main():
    st.title("Diabetes Prediction using Classifier Model")
    
    Pregnancies = st.text_input("Number of Pregancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness Values')
    Insulin = st.text_input('Insulin values')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction values')
    AGE = st.text_input('Age values')

    if st.button("Predict"):
        input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,
                      Insulin,BMI,DiabetesPedigreeFunction,AGE)
        result = diabetes_prediction(input_data)
        st.success(result)

if __name__ =='__main__':
    main()