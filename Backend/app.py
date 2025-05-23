import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from keras.models import model_from_json
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "nutrition_recommender")

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    users_collection = db["users"]
    recommendations_collection = db["recommendations"]
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None

# Load the saved model
model_file = 'nutrition_model_v2.pkl'
try:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
        nn_model = model_from_json(model_data['architecture'])
        nn_model.set_weights(model_data['weights'])
        preprocessor = model_data['preprocessor']
        label_encoder = model_data['label_encoder']
        
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
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

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
        },
        {
            'breakfast': 'Paneer and spinach scramble with whole wheat toast',
            'lunch': 'Lentil and chickpea salad with quinoa and olive oil dressing',
            'dinner': 'Grilled tofu steak with sautéed veggies and brown rice',
            'snacks': ['Greek yogurt', 'Protein smoothie', 'Roasted chickpeas']
        },
        {
            'breakfast': 'Oats with almond milk, chia seeds, and banana',
            'lunch': 'Quinoa and black bean salad with avocado dressing',
            'dinner': 'Grilled tofu and vegetable stir-fry with brown rice',
            'snacks': ['Fruit smoothie with plant-based protein', 'Roasted chickpeas', 'Carrot sticks with hummus']
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
        },
        {
            'breakfast': 'Tofu scramble with avocado and cherry tomatoes',
            'lunch': 'Zucchini noodles with pesto and grilled paneer',
            'dinner': 'Cauliflower rice stir-fry with broccoli and bell peppers',
            'snacks': ['Boiled eggs', 'Roasted chickpeas', 'Greek yogurt with cucumber']
        },
        {
            'breakfast': 'Oats with almond milk, chia seeds, and banana',
            'lunch': 'Quinoa and black bean salad with avocado dressing',
            'dinner': 'Grilled tofu and vegetable stir-fry with brown rice',
            'snacks': ['Fruit smoothie with plant-based protein', 'Roasted chickpeas', 'Carrot sticks with hummus']
        }
    ],
    'Balanced Diet': [
        {
            'breakfast': 'Scrambled eggs with whole grain toast and orange juice',
            'lunch': 'Grilled chicken wrap with mixed greens and hummus',
            'dinner': 'Fish curry with brown rice and sautéed vegetables',
            'snacks': ['Boiled egg', 'Cheese cubes', 'Yogurt with fruit']
        },
        {
            'breakfast': 'Omelette with mushrooms and turkey, side of fruit salad',
            'lunch': 'Beef stew with vegetables and quinoa',
            'dinner': 'Baked tilapia with roasted sweet potatoes',
            'snacks': ['Protein bar', 'Greek yogurt', 'Nuts']
        },
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
        },
        {
            'breakfast': 'Oats with almond milk, chia seeds, and banana',
            'lunch': 'Quinoa and black bean salad with avocado dressing',
            'dinner': 'Grilled tofu and vegetable stir-fry with brown rice',
            'snacks': ['Fruit smoothie with plant-based protein', 'Roasted chickpeas', 'Carrot sticks with hummus']
        }
    ],
    'Low-Fat Diet': [
        {
            'breakfast': 'Scrambled egg whites with tomatoes and whole grain toast',
            'lunch': 'Grilled chicken salad with lemon vinaigrette',
            'dinner': 'Steamed cod with mixed vegetables and brown rice',
            'snacks': ['Low-fat string cheese', 'Boiled egg', 'Apple slices']
        },
        {
            'breakfast': 'Oatmeal with skim milk and berries',
            'lunch': 'Turkey breast sandwich on whole wheat with lettuce and tomato',
            'dinner': 'Baked salmon with steamed broccoli and sweet potato',
            'snacks': ['Greek yogurt', 'Rice cakes with almond butter', 'Hard-boiled eggs']
        },
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
        },
        {
            'breakfast': 'Oats with almond milk, chia seeds, and banana',
            'lunch': 'Quinoa and black bean salad with avocado dressing',
            'dinner': 'Grilled tofu and vegetable stir-fry with brown rice',
            'snacks': ['Fruit smoothie with plant-based protein', 'Roasted chickpeas', 'Carrot sticks with hummus']
        }
    ]
}

#Remove this
def filter_meals(meals, dietary_habits, allergies, aversions):
    filtered = []
    prefer_nonveg = (dietary_habits.lower() == 'regular')

    for meal in meals:
        text = f"{meal['breakfast']} {meal['lunch']} {meal['dinner']} {' '.join(meal['snacks'])}".lower()

        if prefer_nonveg:
            if all(veg_word in text for veg_word in ['lentil', 'chickpea', 'vegetable', 'salad', 'tofu']) and \
               not any(nonveg_word in text for nonveg_word in ['chicken', 'fish', 'turkey', 'egg', 'beef', 'shrimp']):
                continue

        if dietary_habits.lower() == "vegetarian":
            if any(non_veg in text for non_veg in ['chicken', 'fish', 'salmon', 'beef', 'turkey', 'shrimp', 'pork']):
                continue
        elif dietary_habits.lower() == "vegan":
            if any(animal in text for animal in ['egg', 'milk', 'cheese', 'yogurt', 'chicken', 'beef', 'fish', 'butter', 'honey', 'paneer','turkey', 'shrimp', 'bacon', 'salmon', 'lamb']):
                continue

        if allergies.lower() == "nuts":
            if any(nut in text for nut in ['almond', 'walnut', 'nuts', 'cashew', 'peanut']):
                continue
        elif allergies.lower() == "dairy":
            if any(dairy in text for dairy in ['milk', 'cheese', 'yogurt', 'butter']):
                continue
        elif allergies.lower() == "gluten":
            if any(gluten in text for gluten in ['bread', 'pasta', 'cracker', 'toast', 'grain']):
                continue

        if aversions.lower() == "spicy" and "spicy" in text:
            continue
        elif aversions.lower() == "sweet" and any(sugar in text for sugar in ['honey', 'syrup', 'sweet']):
            continue
        elif aversions.lower() == "sour" and "sour" in text:
            continue

        filtered.append(meal)

    return filtered

# Generate recommendations
def make_varied_recommendations(patient_data, temperature=0.8):
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

    # 🔁 Apply dietary filtering logic
    filtered_meals = filter_meals(
        meal_plans[meal_plan],
        patient_data['Dietary_Habits'],
        patient_data['Allergies'],
        patient_data['Food_Aversions']
    )

    if not filtered_meals:
        print("No filtered meals matched preferences. Falling back to unfiltered list.")
        detailed_plan = random.choice(meal_plans[meal_plan])
    else:
        detailed_plan = random.choice(filtered_meals)

    return {
        'mealPlanType': meal_plan,
        'recommendedCalories': round(calories),
        'recommendedProtein': round(protein),
        'recommendedCarbs': round(carbs),
        'recommendedFats': round(fats),
        'detailedMealPlan': detailed_plan
    }


# Save user data and recommendations to MongoDB
def save_to_mongodb(user_data, recommendations):
    if db is None:
        print("MongoDB connection not available. Data not saved.")
        return None
    
    try:
        # Generate a unique user ID if not provided
        user_id = user_data.get('user_id', str(uuid.uuid4()))
        
        # Save user data
        user_document = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'demographics': {
                'age': user_data.get('age'),
                'gender': user_data.get('gender'),
                'height_cm': user_data.get('height_cm'),
                'weight_kg': user_data.get('weight_kg'),
                'bmi': user_data.get('bmi')
            },
            'health_info': {
                'chronic_disease': user_data.get('chronic_disease'),
                'blood_pressure_systolic': user_data.get('blood_pressure_systolic'),
                'blood_pressure_diastolic': user_data.get('blood_pressure_diastolic'),
                'cholesterol_level': user_data.get('cholesterol_level'),
                'blood_sugar_level': user_data.get('blood_sugar_level'),
                'genetic_risk_factor': user_data.get('genetic_risk_factor'),
                'allergies': user_data.get('allergies')
            },
            'lifestyle': {
                'daily_steps': user_data.get('daily_steps'),
                'exercise_frequency': user_data.get('exercise_frequency'),
                'sleep_hours': user_data.get('sleep_hours'),
                'alcohol_consumption': user_data.get('alcohol_consumption'),
                'smoking_habit': user_data.get('smoking_habit')
            },
            'nutrition': {
                'dietary_habits': user_data.get('dietary_habits'),
                'caloric_intake': user_data.get('caloric_intake'),
                'protein_intake': user_data.get('protein_intake'),
                'carbohydrate_intake': user_data.get('carbohydrate_intake'),
                'fat_intake': user_data.get('fat_intake'),
                'preferred_cuisine': user_data.get('preferred_cuisine'),
                'food_aversions': user_data.get('food_aversions')
            }
        }
        
        # Insert or update user data
        users_collection.update_one(
            {'user_id': user_id},
            {'$set': user_document},
            upsert=True
        )
        
        # Save recommendation
        recommendation_document = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'recommendations': recommendations
        }
        
        result = recommendations_collection.insert_one(recommendation_document)
        return result.inserted_id
    
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        
        # Calculate BMI
        height_m = data['height_cm'] / 100
        weight_kg = data['weight_kg']
        bmi = weight_kg / (height_m * height_m)
        data['bmi'] = round(bmi, 1)
        
        # Create patient data in the format expected by the model
        patient_data = {
            'Age': data['age'],
            'Gender': data['gender'],
            'Height_cm': data['height_cm'],
            'Weight_kg': data['weight_kg'],
            'BMI': data['bmi'],
            'Chronic_Disease': data['chronic_disease'],
            'Blood_Pressure_Systolic': data['blood_pressure_systolic'],
            'Blood_Pressure_Diastolic': data['blood_pressure_diastolic'],
            'Cholesterol_Level': data['cholesterol_level'],
            'Blood_Sugar_Level': data['blood_sugar_level'],
            'Genetic_Risk_Factor': data['genetic_risk_factor'],
            'Allergies': data['allergies'],
            'Daily_Steps': data['daily_steps'],
            'Exercise_Frequency': data['exercise_frequency'],
            'Sleep_Hours': data['sleep_hours'],
            'Alcohol_Consumption': data['alcohol_consumption'],
            'Smoking_Habit': data['smoking_habit'],
            'Dietary_Habits': data['dietary_habits'],
            'Caloric_Intake': data['caloric_intake'],
            'Protein_Intake': data['protein_intake'],
            'Carbohydrate_Intake': data['carbohydrate_intake'],
            'Fat_Intake': data['fat_intake'],
            'Preferred_Cuisine': data['preferred_cuisine'],
            'Food_Aversions': data['food_aversions']
        }
        
        recommendations = make_varied_recommendations(patient_data)
        
        # Save data to MongoDB
        save_to_mongodb(data, recommendations)
        
        return jsonify({"success": True, "recommendations": recommendations})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/user-history', methods=['GET'])
def get_user_history():
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({"success": False, "error": "User ID is required"})
        
        if db is None:
            return jsonify({"success": False, "error": "Database connection not available"})
        
        # Get user recommendations history
        recommendations = list(recommendations_collection.find(
            {"user_id": user_id},
            {"_id": 0} # Exclude MongoDB _id field from results
        ).sort("timestamp", -1))  # Sort by newest first
        
        # Convert datetime objects to strings for JSON serialization
        for rec in recommendations:
            rec["timestamp"] = rec["timestamp"].isoformat()
        
        return jsonify({"success": True, "history": recommendations})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/form-data', methods=['GET'])
def get_form_data():
    # Return valid options for form dropdowns
    return jsonify({
        "gender": ["Male", "Female", "Other"],
        "chronic_disease": ["None", "Diabetes", "Hypertension", "Heart Disease"],
        "genetic_risk_factor": ["Yes", "No"],
        "allergies": ["None", "Nuts", "Dairy", "Gluten"],
        "alcohol_consumption": ["Yes", "No"],
        "smoking_habit": ["Yes", "No"],
        "dietary_habits": ["Regular", "Vegetarian", "Vegan"],
        "preferred_cuisine": ["Mediterranean", "Italian", "Indian", "American"],
        "food_aversions": ["None", "Spicy", "Sweet", "Sour"],
        "numeric_ranges": {
            "age": [18, 100],
            "height_cm": [100, 250],
            "weight_kg": [30, 200],
            "BMI" : [10, 50],
            "blood_pressure_systolic": [90, 200],
            "blood_pressure_diastolic": [60, 140],
            "cholesterol_level": [100, 300],
            "blood_sugar_level": [70, 200],
            "daily_steps": [0, 20000],
            "exercise_frequency": [0, 7],
            "sleep_hours": [0, 12],
            "caloric_intake": [1000, 4000],
            "protein_intake": [20, 300],
            "carbohydrate_intake": [0, 500],
            "fat_intake": [0, 200]
        }
    })

if __name__ == '__main__':
    # app.run(debug=True, port=5000)
    app.run(host='0.0.0.0', debug=True, port=5000)
