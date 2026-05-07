import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv("/content/diabetics_recommendation_cleaned4.csv")

# Handle missing values
meal_columns = ['Breakfast', 'Morning_Snack', 'Lunch', 'Afternoon_Snack', 'Dinner', 'Evening_Snack']
data[meal_columns] = data[meal_columns].fillna('')
data.fillna(data.mean(numeric_only=True), inplace=True)

# Feature Engineering
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 100], 
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
data['BP_Category'] = pd.cut(data['Systolic_BP'], bins=[0, 120, 140, 1000], 
                            labels=['Normal', 'Prehypertension', 'Hypertension'])
data['Cholesterol_Ratio'] = data['LDL'] / data['HDL']
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 30, 50, 100], 
                           labels=['Young', 'Middle', 'Senior'])
data['Sugar_BMI_Interaction'] = data['FastingBloodSugar'] * data['BMI']

# Define target variable
bins = [0, 100, 126, 1000]
labels = ['Low', 'Normal', 'High']
data['Sugar_Level'] = pd.cut(data['FastingBloodSugar'], bins=bins, labels=labels)

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Sugar_Level'])

# Numerical and categorical features
numerical_cols = ['FastingBloodSugar', 'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 
                  'Triglycerides', 'BMI', 'Age', 'Cholesterol_Ratio', 'Sugar_BMI_Interaction']
categorical_cols = ['Gender', 'BMI_Category', 'BP_Category', 'Age_Group']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[numerical_cols])
X_categorical = pd.get_dummies(data[categorical_cols], drop_first=True).values

# Combine numerical and categorical features
X = np.hstack((X_numerical, X_categorical))

# Handle class imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Logistic Regression
logreg = LogisticRegression(
    solver='lbfgs',
    penalty='l2',
    C=1.0,
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)
print(f"\nLogistic Regression Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Save model and tools
joblib.dump(logreg, 'logistic_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

# Enhanced recommendation system with flexible filtering
def recommend_meals_health_based(new_patient_df, top_n=1, predicted_sugar_level=None):
    matched = data[
        (data['FastingBloodSugar'] == new_patient_df['FastingBloodSugar'].iloc[0]) &
        (data['Systolic_BP'] == new_patient_df['Systolic_BP'].iloc[0]) &
        (data['Diastolic_BP'] == new_patient_df['Diastolic_BP'].iloc[0]) &
        (data['LDL'] == new_patient_df['LDL'].iloc[0]) &
        (data['HDL'] == new_patient_df['HDL'].iloc[0]) &
        (data['Triglycerides'] == new_patient_df['Triglycerides'].iloc[0]) &
        (data['BMI'] == new_patient_df['BMI'].iloc[0]) &
        (data['Age'] == new_patient_df['Age'].iloc[0]) &
        (data['Gender'] == new_patient_df['Gender'].iloc[0])
    ]
    
    if not matched.empty:
        return matched[meal_columns + ['Sugar_Level'] + numerical_cols + ['Gender']].head(top_n)
    
    filtered_data = data[data['Sugar_Level'] == predicted_sugar_level].copy()
    
    if filtered_data.empty:
        print(f"c'{predicted_sugar_level}'.")
        return pd.DataFrame()
    
   
    
    # Extract health features
    bmi = new_patient_df['BMI'].iloc[0]
    triglycerides = new_patient_df['Triglycerides'].iloc[0]
    systolic_bp = new_patient_df['Systolic_BP'].iloc[0]
    diastolic_bp = new_patient_df['Diastolic_BP'].iloc[0]
    ldl = new_patient_df['LDL'].iloc[0]
    hdl = new_patient_df['HDL'].iloc[0]
    
    # Primary filtering with relaxed ranges
    filtered_data = filtered_data[
        (filtered_data['BMI'].between(bmi - 7, bmi + 7)) &
        (filtered_data['LDL'].between(ldl - 30, ldl + 30)) &
        (filtered_data['Systolic_BP'].between(systolic_bp * 0.75, systolic_bp * 1.25)) &
        (filtered_data['Diastolic_BP'].between(diastolic_bp * 0.75, diastolic_bp * 1.25)) &
        (filtered_data['Triglycerides'].between(triglycerides * 0.6, triglycerides * 1.4))
    ]
    
   
    
    filtered_data = filtered_data.reset_index(drop=True)
    
    weights = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    X_train_numerical = scaler.transform(filtered_data[numerical_cols]) * weights
    X_new_numerical = scaler.transform(new_patient_df[numerical_cols]) * weights
    cosine_sim = cosine_similarity(X_new_numerical, X_train_numerical)
    cosine_sim_normalized = (cosine_sim + 1) / 2
    sim_scores = list(enumerate(cosine_sim_normalized[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    meal_indices = [i[0] for i in sim_scores]
    
    print("\nIndividual similarity scores for recommended meals:")
    for idx, score in sim_scores:
        print(f"Meal {idx}: {score * 100:.2f}%")
    
    return_cols = meal_columns + ['Sugar_Level'] + numerical_cols + ['Gender']
    return filtered_data.iloc[meal_indices][return_cols]

# Collect new patient data
def get_numeric_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if value < min_val or value > max_val:
                print(f"Please enter a value between {min_val} and {max_val}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

def get_binary_input(prompt):
    while True:
        value = input(prompt)
        if value in ['0', '1']:
            return int(value)
        print("Please enter 0 or 1.")

print("\n Enter new patient data:")
new_patient = {
    'FastingBloodSugar': get_numeric_input("Fasting Blood Sugar (mg/dL, example: 80-130): ", 80, 130),
    'Systolic_BP': get_numeric_input("Systolic Blood Pressure (mmHg, example: 90-139): ", 90, 139),
    'Diastolic_BP': get_numeric_input("Diastolic Blood Pressure (mmHg, example: 60-90): ", 60, 90),
    'LDL': get_numeric_input("LDL Cholesterol (mg/dL, example: 40-70): ", 40, 70),
    'HDL': get_numeric_input("HDL Cholesterol (mg/dL, example: 50-70): ", 50, 70),
    'Triglycerides': get_numeric_input("Triglycerides (mg/dL, example: 90-150): ", 90, 150),
    'Gender': get_binary_input("Gender (0 = Female, 1 = Male): "),
    'BMI': get_numeric_input("Body Mass Index BMI (example: 18-36): ", 18, 36),
    'Age': get_numeric_input("Age (years, example: 10-90): ", 10, 90),
}

# Process new patient data
new_patient_df = pd.DataFrame([new_patient])
new_patient_df['BMI_Category'] = pd.cut(new_patient_df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
new_patient_df['BP_Category'] = pd.cut(new_patient_df['Systolic_BP'], bins=[0, 120, 140, 1000], 
                                      labels=['Normal', 'Prehypertension', 'Hypertension'])
new_patient_df['Cholesterol_Ratio'] = new_patient_df['LDL'] / new_patient_df['HDL']
new_patient_df['Age_Group'] = pd.cut(new_patient_df['Age'], bins=[0, 30, 50, 100], 
                                     labels=['Young', 'Middle', 'Senior'])
new_patient_df['Sugar_BMI_Interaction'] = new_patient_df['FastingBloodSugar'] * new_patient_df['BMI']

# Transform new patient data
X_numerical_new = scaler.transform(new_patient_df[numerical_cols])
X_categorical_new = pd.get_dummies(new_patient_df[categorical_cols], drop_first=True)
train_categorical_cols = pd.get_dummies(data[categorical_cols], drop_first=True).columns
X_categorical_new = X_categorical_new.reindex(columns=train_categorical_cols, fill_value=0).values
X_new = np.hstack((X_numerical_new, X_categorical_new))

# Predict
y_pred_new = logreg.predict(X_new)
y_pred_label = label_encoder.inverse_transform(y_pred_new)[0]
print(f"\n Predicting blood sugar levels for a new patient: {y_pred_label}")

# Recommend meals
print("\n Recommended meals for a new patient:")
recommended_meals = recommend_meals_health_based(new_patient_df, top_n=1, predicted_sugar_level=y_pred_label)
if not recommended_meals.empty:
    # Display meal details and key health metrics
    display_cols = meal_columns + ['Sugar_Level', 'BMI', 'Triglycerides', 'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 'Gender']
    print(recommended_meals[display_cols])
    # Calculate efficiency score
    weights = np.array([2.0, 1.0, 1.0, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0, 1.0])
    X_recommended = scaler.transform(recommended_meals[numerical_cols]) * weights
    X_new_scaled = scaler.transform(new_patient_df[numerical_cols]) * weights
    similarities = cosine_similarity(X_new_scaled, X_recommended)[0]
    similarities_normalized = (similarities + 1) / 2
    efficiency_score = np.mean(similarities_normalized) * 100
    print(f"\n Efficiency score for current recommendations: {efficiency_score:.2f}%")
else:
    print("There are no meal recommendations available.")

# Analyze data distribution for debugging
print("\n Statistics Triglycerides and Gender in class 'Normal':")
print(data[data['Sugar_Level'] == 'Normal'][['Triglycerides', 'Gender']].describe())
print("Gender distribution in class 'Normal':")
print(data[data['Sugar_Level'] == 'Normal']['Gender'].value_counts())