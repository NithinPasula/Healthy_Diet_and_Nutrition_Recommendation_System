import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout
import random
import os
import pickle

# For reproducibility with controlled randomness
np.random.seed(random.randint(1, 10000))
tf.random.set_seed(random.randint(1, 10000))

# Load and preprocess dataset
df = pd.read_csv('Database/patient.csv')

# Define features and targets
categorical_features = ['Gender', 'Chronic_Disease', 'Genetic_Risk_Factor',
                        'Allergies', 'Alcohol_Consumption', 'Smoking_Habit',
                        'Dietary_Habits', 'Preferred_Cuisine', 'Food_Aversions']
numerical_features = ['Age', 'Height_cm', 'Weight_kg', 'BMI',
                      'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
                      'Cholesterol_Level', 'Blood_Sugar_Level', 'Daily_Steps',
                      'Exercise_Frequency', 'Sleep_Hours', 'Caloric_Intake',
                      'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake']
target_calories = 'Recommended_Calories'
target_protein = 'Recommended_Protein'
target_carbs = 'Recommended_Carbs'
target_fats = 'Recommended_Fats'
target_meal_plan = 'Recommended_Meal_Plan'

# Handle missing data
for col in categorical_features:
    df[col] = df[col].fillna('None')
for col in numerical_features:
    df[col] = df[col].fillna(df[col].median())

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Encode meal plans
label_encoder = LabelEncoder()
df['Meal_Plan_Encoded'] = label_encoder.fit_transform(df[target_meal_plan])
meal_plan_classes = label_encoder.classes_

# Split data
X = df[numerical_features + categorical_features]
y_calories = df[target_calories]
y_protein = df[target_protein]
y_carbs = df[target_carbs]
y_fats = df[target_fats]
y_meal_plan = df['Meal_Plan_Encoded']
X_train, X_test, y_calories_train, y_calories_test, y_protein_train, y_protein_test, \
y_carbs_train, y_carbs_test, y_fats_train, y_fats_test, \
y_meal_plan_train, y_meal_plan_test = train_test_split(
    X, y_calories, y_protein, y_carbs, y_fats, y_meal_plan, test_size=0.2, random_state=42)

# Preprocess features
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
input_shape = X_train_preprocessed.shape[1]

# Neural network model
def build_nn_model():
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    shared = Dense(32, activation='relu')(x)

    meal_plan_branch = Dense(16, activation='relu')(shared)
    meal_plan_noise = tf.keras.layers.GaussianNoise(0.2)(meal_plan_branch)
    meal_plan_output = Dense(len(meal_plan_classes), activation='softmax', name='meal_plan')(meal_plan_noise)

    calories_branch = Dense(16, activation='relu')(shared)
    calories_noise = tf.keras.layers.GaussianNoise(0.15)(calories_branch)
    calories_output = Dense(1, name='calories')(calories_noise)

    protein_branch = Dense(16, activation='relu')(shared)
    protein_noise = tf.keras.layers.GaussianNoise(0.15)(protein_branch)
    protein_output = Dense(1, name='protein')(protein_noise)

    carbs_branch = Dense(16, activation='relu')(shared)
    carbs_noise = tf.keras.layers.GaussianNoise(0.15)(carbs_branch)
    carbs_output = Dense(1, name='carbs')(carbs_noise)

    fats_branch = Dense(16, activation='relu')(shared)
    fats_noise = tf.keras.layers.GaussianNoise(0.15)(fats_branch)
    fats_output = Dense(1, name='fats')(fats_noise)

    model = Model(inputs=inputs, outputs=[
        meal_plan_output, calories_output, protein_output, carbs_output, fats_output
    ])
    model.compile(
        optimizer='adam',
        loss={
            'meal_plan': 'sparse_categorical_crossentropy',
            'calories': 'mean_squared_error',
            'protein': 'mean_squared_error',
            'carbs': 'mean_squared_error',
            'fats': 'mean_squared_error'
        },
        metrics={
            'meal_plan': 'accuracy',
            'calories': 'mae',
            'protein': 'mae',
            'carbs': 'mae',
            'fats': 'mae'
        }
    )
    return model

# Define meal plans (aligned with dataset categories)
meal_plans = {
    'High-Protein Diet': [
        {
            'breakfast': 'Egg white omelette with spinach and turkey, whole grain toast',
            'lunch': 'Grilled chicken breast, quinoa, steamed broccoli',
            'dinner': 'Baked salmon, sweet potato, asparagus',
            'snacks': ['Greek yogurt with berries', 'Protein shake', 'Handful of almonds']
        },
        {
            'breakfast': 'Protein pancakes with fresh berries, turkey bacon',
            'lunch': 'Tuna salad with mixed greens, olive oil dressing',
            'dinner': 'Lean beef stir fry with bell peppers and brown rice',
            'snacks': ['Cottage cheese with pineapple', 'Hard-boiled eggs', 'Edamame']
        },
        {
            'breakfast': 'Greek yogurt with nuts, seeds, and honey',
            'lunch': 'Grilled shrimp bowl with black beans and corn',
            'dinner': 'Turkey meatballs with zucchini noodles',
            'snacks': ['Protein bar', 'Smoked salmon on cucumber slices', 'Beef jerky']
        }
    ],
    'Low-Carb Diet': [
        {
            'breakfast': 'Avocado and bacon omelette',
            'lunch': 'Cobb salad with grilled chicken and blue cheese dressing',
            'dinner': 'Baked cod with roasted cauliflower',
            'snacks': ['Celery with almond butter', 'String cheese', 'Pork rinds']
        },
        {
            'breakfast': 'Chia seed pudding with coconut milk and berries',
            'lunch': 'Lettuce-wrapped burger with side salad',
            'dinner': 'Grilled steak with garlic butter and asparagus',
            'snacks': ['Deviled eggs', 'Olives', 'Keto fat bombs']
        },
        {
            'breakfast': 'Crustless spinach and feta quiche',
            'lunch': 'Chicken caesar salad (no croutons)',
            'dinner': 'Zucchini boats stuffed with ground turkey and cheese',
            'snacks': ['Guacamole with bell pepper slices', 'Macadamia nuts', 'Pepperoni slices']
        }
    ],
    'Balanced Diet': [
        {
            'breakfast': 'Oatmeal with banana, cinnamon, and walnuts',
            'lunch': 'Quinoa bowl with roasted vegetables and chickpeas',
            'dinner': 'Grilled fish with wild rice and roasted asparagus',
            'snacks': ['Fresh fruit', 'Yogurt', 'Trail mix']
        },
        {
            'breakfast': 'Whole grain toast with avocado and poached egg',
            'lunch': 'Mediterranean wrap with hummus and vegetables',
            'dinner': 'Baked chicken with sweet potato and green beans',
            'snacks': ['Apple with peanut butter', 'Whole grain crackers with cheese', 'Carrot sticks with hummus']
        },
        {
            'breakfast': 'Smoothie bowl with mixed berries, banana, and granola',
            'lunch': 'Lentil soup with whole grain bread',
            'dinner': 'Stir-fried tofu with mixed vegetables and brown rice',
            'snacks': ['Orange slices', 'Mixed nuts', 'Rice cakes with avocado']
        }
    ],
    'Low-Fat Diet': [
        {
            'breakfast': 'Greek yogurt with honey, walnuts, and fresh figs',
            'lunch': 'Lentil salad with feta, cucumber, and tomatoes',
            'dinner': 'Grilled sea bass with olive oil, lemon, and herbs, side of roasted vegetables',
            'snacks': ['Hummus with pita', 'Handful of olives', 'Fresh grapes']
        },
        {
            'breakfast': 'Whole grain toast with tomato, olive oil, and herbs',
            'lunch': 'Chickpea and vegetable soup with a small piece of bread',
            'dinner': 'Eggplant moussaka with Greek salad',
            'snacks': ['Tzatziki with cucumber slices', 'Dates stuffed with almond butter', 'Orange slices']
        },
        {
            'breakfast': 'Frittata with spinach, tomatoes, and feta cheese',
            'lunch': 'Tabbouleh salad with grilled chicken',
            'dinner': 'Baked salmon with quinoa and roasted Mediterranean vegetables',
            'snacks': ['Almonds and dried apricots', 'Greek yogurt with honey', 'Whole grain crackers with hummus']
        }
    ]
}

# Generate recommendations
def make_varied_recommendations(patient_data, nn_model, preprocessor, meal_plans, label_encoder, temperature=0.8):
    patient_df = pd.DataFrame([patient_data])
    patient_preprocessed = preprocessor.transform(patient_df)
    predictions = nn_model.predict(patient_preprocessed, verbose=0)
    meal_plan_probs, calories, protein, carbs, fats = predictions

    meal_plan_probs = meal_plan_probs[0]
    meal_plan_probs = np.log(meal_plan_probs + 1e-10) / temperature
    meal_plan_probs = np.exp(meal_plan_probs)
    meal_plan_probs = meal_plan_probs / np.sum(meal_plan_probs)

    meal_plan_idx = np.random.choice(len(meal_plan_probs), p=meal_plan_probs)
    meal_plan = label_encoder.classes_[meal_plan_idx]

    noise_factor = 0.05
    calories = float(calories[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    protein = float(protein[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    carbs = float(carbs[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    fats = float(fats[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))

    calories = max(1500, min(3500, calories))  # Adjusted minimum to 1500 kcal
    protein = max(50, min(200, protein))
    carbs = max(50, min(400, carbs))
    fats = max(20, min(150, fats))

    if meal_plan not in meal_plans:
        print(f"Warning: Predicted meal plan '{meal_plan}' not in meal_plans. Defaulting to Balanced Diet.")
        meal_plan = 'Balanced Diet'
    detailed_plan = random.choice(meal_plans[meal_plan])

    return {
        'Meal Plan Type': meal_plan,
        'Recommended Calories': round(calories),
        'Recommended Protein (g)': round(protein),
        'Recommended Carbs (g)': round(carbs),
        'Recommended Fats (g)': round(fats),
        'Detailed Meal Plan': detailed_plan
    }

# Validate and get user input
# Validate and get user input
def get_user_input():
    print("\n== Enter Patient Information ==\n(Enter values within the specified ranges)")
    default_patient = {
        'Age': 45, 'Gender': 'Male', 'Height_cm': 178, 'Weight_kg': 85, 'BMI': 26.8,
        'Chronic_Disease': 'None', 'Blood_Pressure_Systolic': 130, 'Blood_Pressure_Diastolic': 85,
        'Cholesterol_Level': 195, 'Blood_Sugar_Level': 105, 'Genetic_Risk_Factor': 'No',
        'Allergies': 'None', 'Daily_Steps': 8000, 'Exercise_Frequency': 3, 'Sleep_Hours': 7,
        'Alcohol_Consumption': 'Yes', 'Smoking_Habit': 'No', 'Dietary_Habits': 'Regular',
        'Caloric_Intake': 2300, 'Protein_Intake': 95, 'Carbohydrate_Intake': 250, 'Fat_Intake': 80,
        'Preferred_Cuisine': 'Mediterranean', 'Food_Aversions': 'None'
    }
    valid_categorical = {
        'Gender': ['Male', 'Female', 'Other'],
        'Chronic_Disease': ['None', 'Diabetes', 'Hypertension', 'Heart Disease'],
        'Genetic_Risk_Factor': ['Yes', 'No'],
        'Allergies': ['None', 'Nuts', 'Dairy', 'Gluten'],
        'Alcohol_Consumption': ['Yes', 'No'],
        'Smoking_Habit': ['Yes', 'No'],
        'Dietary_Habits': ['Regular', 'Vegetarian', 'Vegan'],
        'Preferred_Cuisine': ['Mediterranean', 'Italian', 'Indian', 'American'],
        'Food_Aversions': ['None', 'Spicy', 'Sweet', 'Sour']
    }
    # Define acceptable ranges for all numerical features
    numeric_ranges = {
        'Age': (18, 100),
        'Height_cm': (100, 250),
        'Weight_kg': (30, 200),
        'BMI': (15, 40),
        'Blood_Pressure_Systolic': (90, 200),
        'Blood_Pressure_Diastolic': (60, 140),
        'Cholesterol_Level': (100, 300),
        'Blood_Sugar_Level': (70, 200),
        'Daily_Steps': (0, 20000),
        'Exercise_Frequency': (0, 7),
        'Sleep_Hours': (0, 12),
        'Caloric_Intake': (1000, 4000),
        'Protein_Intake': (20, 300),
        'Carbohydrate_Intake': (0, 500),
        'Fat_Intake': (0, 200)
    }
    patient = {}
    for key, default in default_patient.items():
        while True:
            # Build prompt without default values
            if key in numeric_ranges:
                low, high = numeric_ranges[key]
                prompt = f"{key} ({low}-{high}): "
            elif key in valid_categorical:
                options = ", ".join(valid_categorical[key])
                prompt = f"{key} ({options}): "
            else:
                prompt = f"{key}: "

            value = input(prompt).strip()
            # Use default if empty
            if not value:
                patient[key] = default
                break

            # Validate numerical inputs
            if key in numerical_features:
                try:
                    value = float(value)
                    low, high = numeric_ranges.get(key, (None, None))
                    if low is not None and not (low <= value <= high):
                        print(f"{key} must be between {low} and {high}.")
                        continue
                    # Additional specific checks
                    if key == 'Age' and not (18 <= value <= 100):
                        print("Age must be between 18 and 100.")
                        continue
                    patient[key] = value
                    break
                except ValueError:
                    print(f"{key} must be a number.")
            # Validate categorical inputs
            else:
                if value in valid_categorical.get(key, [value]):
                    patient[key] = value
                    break
                print(f"Invalid {key}. Choose from: {valid_categorical[key]}")
    return patient

# Load or train model
model_file = 'nutrition_model_v2.pkl'
old_model_file = 'nutrition_model.pkl'
use_pretrained = False

# Force retraining if nutrition_model_v2.pkl doesn't exist
if os.path.exists(model_file):
    choice = input("\nUse pretrained model (p) or retrain (r)? [p/r]: ").lower()
    if choice == 'p':
        use_pretrained = True
elif os.path.exists(old_model_file):
    print("\nOld model file 'nutrition_model.pkl' found.")
    choice = input("Use old pretrained model (p) or retrain (r)? [p/r]: ").lower()
    if choice == 'p':
        use_pretrained = True
        model_file = old_model_file
else:
    print("\nNo pretrained model found. Training new model...")
    use_pretrained = False

if use_pretrained:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
        nn_model = model_from_json(model_data['architecture'])
        nn_model.set_weights(model_data['weights'])
        # Handle old .pkl format lacking preprocessor and label_encoder
        if 'preprocessor' in model_data and 'label_encoder' in model_data:
            preprocessor = model_data['preprocessor']
            label_encoder = model_data['label_encoder']
        else:
            print("Old .pkl format detected. Regenerating preprocessor and label_encoder.")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            preprocessor.fit(X_train)
            label_encoder = LabelEncoder()
            label_encoder.fit(df[target_meal_plan])
        nn_model.compile(
            optimizer='adam',
            loss={
                'meal_plan': 'sparse_categorical_crossentropy',
                'calories': 'mean_squared_error',
                'protein': 'mean_squared_error',
                'carbs': 'mean_squared_error',
                'fats': 'mean_squared_error'
            },
            metrics={
                'meal_plan': 'accuracy',
                'calories': 'mae',
                'protein': 'mae',
                'carbs': 'mae',
                'fats': 'mae'
            }
        )
else:
    print("Training model...")
    nn_model = build_nn_model()
    history = nn_model.fit(
        X_train_preprocessed,
        {
            'meal_plan': y_meal_plan_train,
            'calories': y_calories_train,
            'protein': y_protein_train,
            'carbs': y_carbs_train,
            'fats': y_fats_train
        },
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    print("Saving model to nutrition_model_v2.pkl...")
    model_data = {
        'architecture': nn_model.to_json(),
        'weights': nn_model.get_weights(),
        'preprocessor': preprocessor,
        'label_encoder': label_encoder
    }
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)

# Interactive recommendation loop
while True:
    try_another = input("\nGenerate recommendation for a patient? (y/n): ").lower()
    if try_another != 'y':
        break
    patient = get_user_input()
    recommendations = make_varied_recommendations(
        patient, nn_model, preprocessor, meal_plans, label_encoder
    )
    print("\n=== Personalized Nutrition Recommendations ===")
    print(f"Recommended Meal Plan: {recommendations['Meal Plan Type']}")
    print(f"Daily Caloric Target: {recommendations['Recommended Calories']} kcal")
    print(f"Protein: {recommendations['Recommended Protein (g)']} g")
    print(f"Carbohydrates: {recommendations['Recommended Carbs (g)']} g")
    print(f"Fats: {recommendations['Recommended Fats (g)']} g")
    detailed_plan = recommendations['Detailed Meal Plan']
    print("\nDetailed Meal Plan:")
    print(f"Breakfast: {detailed_plan['breakfast']}")
    print(f"Lunch: {detailed_plan['lunch']}")
    print(f"Dinner: {detailed_plan['dinner']}")
    print("Snacks:")
    for snack in detailed_plan['snacks']:
        print(f"- {snack}")
