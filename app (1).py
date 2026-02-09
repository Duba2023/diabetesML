
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
        if processed_input_df[col].iloc[0] == 0:
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

    # Define the full list of feature columns, including missing indicators
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                       'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
                       'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing']

    # Reorder all columns (both original and missing indicators) to match the training data
    processed_input_df = processed_input_df[feature_columns] # Ensure all columns are present and ordered

    # Identify numerical columns to be scaled (these are the ones the scaler was fitted on)
    columns_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                        'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Separate numerical columns from missing indicator columns
    numerical_input_for_scaling = processed_input_df[columns_to_scale]
    missing_indicators = processed_input_df[[col for col in feature_columns if col.endswith('_Missing')]]

    # Apply scaling only to the numerical columns
    scaled_numerical_input = scaler.transform(numerical_input_for_scaling)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=columns_to_scale)

    # Recombine scaled numerical columns and original missing indicators
    # Create a new DataFrame with the correct column order for the model
    final_input_df_for_prediction = pd.DataFrame(index=[0], columns=feature_columns)
    for col in columns_to_scale:
        final_input_df_for_prediction[col] = scaled_numerical_df[col]
    for col in [c for c in feature_columns if c.endswith('_Missing')]:
        final_input_df_for_prediction[col] = missing_indicators[col]

    # Make prediction
    prediction = model.predict(final_input_df_for_prediction)
    prediction_proba = model.predict_proba(final_input_df_for_prediction)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The patient is likely to have diabetes with a probability of {prediction_proba[0]:.2f}.')
    else:
        st.success(f'The patient is likely NOT to have diabetes with a probability of {prediction_proba[0]:.2f}.')
