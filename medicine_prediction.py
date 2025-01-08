import pandas as pd
import numpy as np
import pickle
from sklearn import svm
import google.generativeai as genai

# Directly pass the Gemini API key
genai.configure(api_key="AIzaSyDjuZHbm2C9gH3dywU1Z9bJlE9PkUb-IY0")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}

# Function to get suggestions from Gemini
def get_gemini_suggestions(disease):
    prompt = f"Create a detailed meal plan, workout, medication, and precautions for {disease}."
    model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)
    try:
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error fetching Gemini suggestions: {str(e)}"

# Load additional datasets
precaution = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')
description = pd.read_csv('description.csv')
medication = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')
Training = pd.read_csv('Training.csv')


# Load symptoms data from file
symptoms_df = pd.read_csv('symptoms_df.csv')
symptoms_columns = ['symptoms_1', 'symptoms_2', 'symptoms_3', 'symptoms_4']
all_symptoms = pd.melt(symptoms_df[symptoms_columns], value_name='symptom')['symptom'].dropna().unique()

# Create a dictionary for mapping symptoms to indices
symptoms_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Load the diseases list
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 
    14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 39: 'Varicose veins',
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis',
    5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Training the Support Vector Classifier (SVC)
X_train = Training.drop('prognosis',axis=1) # Your training data here
y_train = Training['prognosis']
 # To check the distribution of labels
# Your training labels here

# Train and save the model
svc = svm.SVC()  # Initialize the SVC model
svc.fit(X_train, y_train)  # Train the model

# Save the trained model to a file
with open('svc.pkl', 'wb') as model_file:
    pickle.dump(svc, model_file)

# Load the saved SVC model
svc = pickle.load(open('svc.pkl', 'rb'))

# Helper function to retrieve data from CSVs and Gemini API
def helper(dis):
    # Description
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc)

    # Precautions
    pre = precaution[precaution['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    pre = pre[0] if len(pre) > 0 else ["No precautions found"]

    # Medications
    med = medication[medication['Disease'] == dis]['Medication'].values
    med = med if len(med) > 0 else ["No medications found"]

    # Diet
    die = diets[diets['Disease'] == dis]['Diet'].values
    die = die if len(die) > 0 else ["No diet found"]

    # Workout
    wrkout = workout[workout['disease'] == dis]['workout'].values
    wrkout = wrkout if len(wrkout) > 0 else ["No workout found"]

    # Call Gemini API for additional suggestions
    gemini_response = get_gemini_suggestions(dis)

    return desc, pre, med, die, wrkout, gemini_response

# Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))  # Initialize zeros

    # Map patient symptoms to input vector
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            print(f"Warning: symptoms '{item}' not recognized.")

    return diseases_list[svc.predict([input_vector])[0]]

# Get user input for symptoms
symptoms = input("Enter your symptoms (comma separated): ")
user_symptoms = [s.strip() for s in symptoms.split(',')]
user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]

# Predict the disease
predicted_disease = get_predicted_value(user_symptoms)

# Retrieve all relevant information
desc, pre, med, die, wrkout, gemini_suggestions = helper(predicted_disease)

# Display the results
print("\nPredicted Disease: ", predicted_disease)
print("Description: ", desc)

print("\nPrecaution:")
for i, p_i in enumerate(pre, 1):
    print(f"{i}: {p_i}")

print("\nMedication:")
for i, m_i in enumerate(med, 1):
    print(f"{i}: {m_i}")

print("\nWorkout:")
for i, w_i in enumerate(wrkout, 1):
    print(f"{i}: {w_i}")

print("\nDiet:")
for i, d_i in enumerate(die, 1):
    print(f"{i}: {d_i}")

# Print additional suggestions from Gemini API
print("\nAdditional Suggestions from Gemini API:\n", gemini_suggestions)
