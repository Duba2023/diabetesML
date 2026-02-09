import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model_and_preprocessing_tools():
    with open('diabetes_prediction_model.pkl', 'rb') as file:
        components = pickle.load(file)
    return (components['imputation_medians'],
            components['transformation_lambdas'],
            components['scaler'],
            components['model'])

# Load the components
imputation_medians, transformation_lambdas, scaler, model = load_model_and_preprocessing_tools()

st.title('Diabetes Prediction App')
st.write('Enter the patient’s details to predict the likelihood of diabetes.')

with st.sidebar:
    st.header('Input Patient Data')
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('BloodPressure', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('SkinThickness', min_value=0, max_value=99, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree_function = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, value=0.3, format='%.3f')
    age = st.number_input('Age', min_value=21, max_value=100, value=30)

# When the user clicks the 'Predict' button
if st.sidebar.button('Predict Diabetes'):
    # Create a dictionary of user inputs
    user_input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # Convert user input to a DataFrame
    input_df = pd.DataFrame([user_input_dict])

    # Apply preprocessing steps
    processed_input_df = input_df.copy()

    # 1. Handle 0s and create missing indicators (for columns originally with 0s that were imputed)
    columns_to_process = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']
    for col in columns_to_process:
        processed_input_df[f'{col}_Missing'] = 0 # Initialize as not missing
        if col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and processed_input_df[col].iloc[0] == 0:
            processed_input_df.loc[0, f'{col}_Missing'] = 1
            processed_input_df.loc[0, col] = imputation_medians[col]

    # 2. Apply transformations
    from scipy.stats import boxcox, yeojohnson

    # Box-Cox transformation (requires positive values)
    columns_for_boxcox = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
    for col in columns_for_boxcox:
        if transformation_lambdas[col] is not None: # Apply only if lambda was found
            processed_input_df[col] = boxcox(processed_input_df[col], lmbda=transformation_lambdas[col])

    # Yeo-Johnson transformation (handles zero and negative values)
    columns_for_yeojohnson = ['Pregnancies', 'BloodPressure']
    for col in columns_for_yeojohnson:
        if transformation_lambdas[col] is not None: # Apply only if lambda was found
            processed_input_df[col] = yeojohnson(processed_input_df[col], lmbda=transformation_lambdas[col])

    # Ensure all expected feature columns are present and in the correct order
    # The original feature_columns included missing indicators, so we must add them to processed_input_df
    # The order of columns must match the training data used for the scaler and model.
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                       'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
                       'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing']

    # Reorder columns to match the training data
    processed_input_df = processed_input_df[feature_columns]

    # 3. Apply scaling
    scaled_input = scaler.transform(processed_input_df)
    scaled_input_df = pd.DataFrame(scaled_input, columns=feature_columns)

    # Make prediction
    prediction = model.predict(scaled_input_df)
    prediction_proba = model.predict_proba(scaled_input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The patient is likely to have diabetes with a probability of {prediction_proba[0]:.2f}.')
    else:
        st.success(f'The patient is likely NOT to have diabetes with a probability of {prediction_proba[0]:.2f}.')
