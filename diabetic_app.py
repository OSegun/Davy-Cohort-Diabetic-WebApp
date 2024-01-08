import streamlit as st
import pickle
import numpy as np
import pandas as pd


loaded_model = pickle.load(open('diabetics_model.sav', 'rb'))

#Input_data = (1,163,72,0,0,39,1.222,33)

def predictions(Input_data):

    Input_data_numpy =  np.asarray(Input_data)

    Input_data_reshaped = Input_data_numpy.reshape(1, -1)

    data_prediction = loaded_model.predict(Input_data_reshaped)
    print(data_prediction)

    if (data_prediction[0]== 0):
        return 'The patient is non diabetic'
    else:
        return 'The patient is diabetic'
    



def main():

    st.title("Davey GoMyCode Diabetic Recommendation System")

    Pregnancies = st.number_input("Number of Pregnancy? ")
    Glucose  = st.number_input("Glucose Level? ")
    BloodPressure = st.number_input("Blood Pre4ssure? ")
    SkinThickness = st.number_input("Skin Thickness? ")
    Insulin = st.number_input("Insulin Level? ")
    BMI = st.number_input("Body Mass Index? ")
    DiabetesPedigreeFunction = st.number_input("Diabetic Pedigree Function? ")
    Age = st.number_input("Patient Age? ")
    
    
    Outcome = " "


    if st.button("Result"):
        Outcome = predictions([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age])
        
    st.success(Outcome)


if __name__ == "__main__":
    main()