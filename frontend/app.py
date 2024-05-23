import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process(model, age, gender, smoking, hx_smoking, hx_radiothreapy, thyroid_function, physical_examination, adenopathy, pathology, focality, risk, t, n, m, stage, response):

    thyroid_function_mapping = {'Euthyroid': 2, 'Clinical Hyperthyroidism': 0, 'Clinical Hypothyroidism': 1,
       'Subclinical Hyperthyroidism': 3, 'Subclinical Hypothyroidism': 4}
    physical_examination_mapping = {'Single nodular goiter-left': 3, 'Multinodular goiter': 1,
       'Single nodular goiter-right': 4, 'Normal': 2, 'Diffuse goiter': 0}
    pathology_mapping = {'Micropapillary': 2, 'Papillary': 3, 'Follicular': 0, 'Hurthel cell': 1}
    focality_mapping = {'Uni-Focal': 1, 'Multi-Focal': 0}
    risk_mapping = {'Low': 2, 'Intermediate': 1, 'High': 0}
    t_mapping = {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T4a': 5, 'T4b': 6}
    n_mapping = {'N0': 0, 'N1b': 2, 'N1a': 1}
    m_mapping = {'M0': 0, 'M1': 1}
    stage_mapping = {'I': 0, 'II': 1, 'IVB': 4, 'III': 2, 'IVA': 3}
    response_mapping = {'Indeterminate': 2, 'Excellent': 1, 'Structural Incomplete': 3,
 'Biochemical Incomplete': 0}
    
    gender = 1 if (gender == "Male") else 0
    smoking = 1 if (smoking == "Yes") else 0
    hx_smoking = 1 if (hx_smoking == "Yes") else 0
    hx_radiothreapy = 1 if (hx_radiothreapy == "Yes") else 0
    adenopathy = 1 if (adenopathy == "Yes") else 0

    thyroid_function = thyroid_function_mapping[thyroid_function]
    physical_examination = physical_examination_mapping[physical_examination]
    pathology = pathology_mapping[pathology]
    focality = focality_mapping[focality]
    risk = risk_mapping[risk]
    t = t_mapping[t]
    n = n_mapping[n]
    m = m_mapping[m]
    stage = stage_mapping[stage]
    response = response_mapping[response]

    input = [age, gender, smoking, hx_smoking, hx_radiothreapy, thyroid_function, physical_examination, adenopathy, pathology, focality, risk, t, n, m, stage, response]

    result = model.predict(np.array(input).reshape(1, -1))

    return result

def main():
    st.write("## Thyroid Disease Prediction")

    #Sidebar
    selected_option = st.sidebar.radio(label = "Menu", options = ["Models", "Diagnose"])

    if selected_option == "Models":
        st.write("#### Model Analysis") 

        results = None 

        with open("../data/results.pkl", "rb") as file:
            results = pickle.load(file)
        file.close()
    

        model_names = ["lr", "knn", "svm", "dt", "rf", "gb"]
        accuracies = []

        for model_name in model_names:
            accuracies.append([results[model_name]["acc"]])
        
        data = pd.DataFrame(accuracies, index = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest", "Gradient Boosting"])
        st.bar_chart(data)

    else:

        #Input Form
        row1 = st.columns(3)
        age = row1[0].text_input(label = "Age")
        gender = row1[1].selectbox("Gender", ["Male", "Female"])
        smoking = row1[2].selectbox("Smoking", ["Yes", "No"])

        st.write("")

        row2 = st.columns([1,2.5,2.5,1])
        hx_smoking = row2[1].selectbox("Hx Smoking", ["Yes", "No"])
        hx_radiothreapy = row2[2].selectbox("Hx Radiothreapy", ["Yes", "No"])

        st.write("")

        row3 = st.columns(3)
        thyroid_function = row3[0].selectbox("Thyroid Function", ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism',
       'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'])
        physical_examination = row3[1].selectbox("Physical Examination", ['Single nodular goiter-left', 'Multinodular goiter',
       'Single nodular goiter-right', 'Normal', 'Diffuse goiter'])
        adenopathy = row3[2].selectbox("Adenopathy", ["Yes", "No"])
        
        st.write("")

        row4 = st.columns(4)
        pathology = row4[0].selectbox("Pathology", ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
        focality = row4[1].selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
        risk = row4[2].selectbox("Risk", ['Low', 'Intermediate', 'High'])
        t = row4[3].selectbox("T", ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
        
        st.write("")

        row5 = st.columns(4)
        n = row5[0].selectbox("N", ['N0', 'N1b', 'N1a'])
        m = row5[1].selectbox("M", ['M0', 'M1'])
        stage = row5[2].selectbox("Stage", ['I', 'II', 'IVB', 'III', 'IVA'])
        response = row5[3].selectbox("Response", ['Indeterminate', 'Excellent', 'Structural Incomplete',
       'Biochemical Incomplete'])
        
        st.write("")
        st.write("")

        tmp = st.columns(3)

        #Model Selection
        temp = st.columns([2,4])
        selected_box = tmp[1].selectbox("Select Model for Prediction", ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest", "Gradient Boosting"])

        mapping = {"Logistic Regression": "lr", "KNN": "knn", "SVM": "svm", "Decision Tree": "dt", "Random Forest": "rf", "Gradient Boosting": "gb"}

        model_name = mapping[selected_box]
        model = None 

        with open(f"../data/{model_name}.pkl", "rb") as file:
            model = pickle.load(file)
        file.close()

        with st.columns(7)[3]:
            btn = st.button("Predict")

        if btn:
            result = process(model, age, gender, smoking, hx_smoking, hx_radiothreapy, thyroid_function, physical_examination, adenopathy, pathology, focality, risk, t, n, m, stage, response)

            st.markdown(f"<h1 style='text-align: center; font-size: 33px;'>Result: {'Yes' if result == 1 else 'No'}</h1>", unsafe_allow_html = True)

if __name__ == "__main__":
    main()