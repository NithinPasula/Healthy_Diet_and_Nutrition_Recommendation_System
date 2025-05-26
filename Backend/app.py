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


load_dotenv()

app = Flask(__name__)
CORS(app)  


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "nutrition_recommender")


try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    users_collection = db["users"]
    recommendations_collection = db["recommendations"]
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None


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
        },
        {
            'breakfast': 'Cottage cheese pancakes with fresh strawberries',
            'lunch': 'Grilled chicken thighs with wild rice and roasted Brussels sprouts',
            'dinner': 'Pan-seared halibut with quinoa pilaf and green beans',
            'snacks': ['Turkey roll-ups with cheese', 'Pumpkin seeds', 'Chocolate protein pudding']
        },
        {
            'breakfast': 'Hemp seed smoothie bowl with banana and coconut flakes',
            'lunch': 'Tempeh and vegetable curry with brown rice',
            'dinner': 'Grilled portobello mushrooms stuffed with quinoa and herbs',
            'snacks': ['Pea protein smoothie', 'Sunflower seed butter on celery', 'Nutritional yeast popcorn']
        },
        {
            'breakfast': 'Chia pudding with almond milk and fresh mango',
            'lunch': 'Black bean and sweet potato bowl with tahini dressing',
            'dinner': 'Lentil walnut bolognese with zucchini noodles',
            'snacks': ['Plant-based protein bar', 'Spirulina smoothie', 'Roasted pumpkin seeds']
        },
        {
            'breakfast': 'Quinoa breakfast bowl with nuts and dried fruit',
            'lunch': 'Three-bean chili with cornbread',
            'dinner': 'Stuffed bell peppers with lentils and brown rice',
            'snacks': ['Almond butter energy balls', 'Green smoothie with spinach', 'Trail mix with seeds']
        },
        {
            'breakfast': 'Protein-rich overnight oats with chia and hemp hearts',
            'lunch': 'Chickpea tikka masala with basmati rice',
            'dinner': 'Grilled eggplant with tahini sauce and quinoa tabbouleh',
            'snacks': ['Homemade granola with nuts', 'Coconut yogurt with berries', 'Roasted edamame']
        },
        {
            'breakfast': 'Tofu scramble with spinach and whole grain toast',
            'lunch': 'Lentil loaf with mashed cauliflower and kale',
            'dinner': 'Tempeh stir-fry with broccoli and brown rice',
            'snacks': ['Roasted edamame', 'Chickpea protein balls', 'Vegan protein smoothie']
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
        },
        {
            'breakfast': 'Smoked salmon with cream cheese and cucumber slices',
            'lunch': 'Greek salad with grilled lamb and olive oil',
            'dinner': 'Pork tenderloin with roasted radishes and herbs',
            'snacks': ['Brie cheese with olives', 'Avocado with sea salt', 'Prosciutto-wrapped asparagus']
        },
        {
            'breakfast': 'Coconut flour pancakes with sugar-free syrup',
            'lunch': 'Tuna-stuffed bell peppers with mixed greens',
            'dinner': 'Herb-crusted chicken thighs with cauliflower mash',
            'snacks': ['Cheese crisps', 'Cucumber boats with tuna salad', 'Hard-boiled egg with everything seasoning']
        },
        {
            'breakfast': 'Almond flour muffins with cream cheese frosting',
            'lunch': 'Cauliflower rice bowl with grilled vegetables and tahini',
            'dinner': 'Stuffed mushrooms with spinach and cashew cream',
            'snacks': ['Kale chips', 'Avocado chocolate mousse', 'Coconut butter fat bombs']
        },
        {
            'breakfast': 'Green smoothie with avocado and coconut milk',
            'lunch': 'Zucchini lasagna with cashew ricotta',
            'dinner': 'Cauliflower steaks with herb oil and roasted vegetables',
            'snacks': ['Coconut yogurt with nuts', 'Cucumber with almond butter', 'Seaweed snacks']
        },
        {
            'breakfast': 'Chia seed breakfast pudding with coconut cream',
            'lunch': 'Shiitake mushroom and spinach salad with hemp seeds',
            'dinner': 'Grilled tempeh with sautéed bok choy and sesame oil',
            'snacks': ['Macadamia nut butter on celery', 'Coconut chips', 'Avocado and lime smoothie']
        },
        {
            'breakfast': 'Chia pudding with unsweetened almond milk and flaxseeds',
            'lunch': 'Zucchini noodles with tofu and cashew pesto',
            'dinner': 'Stuffed bell peppers with cauliflower rice and lentils',
            'snacks': ['Avocado slices with lime', 'Seaweed salad', 'Celery sticks with almond butter']
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
        },
        {
            'breakfast': 'French toast with fresh berries and maple syrup',
            'lunch': 'Grilled portobello burger with sweet potato fries',
            'dinner': 'Herb-roasted chicken with wild rice and roasted carrots',
            'snacks': ['Banana with almond butter', 'Whole grain muffin', 'Cherry tomatoes with mozzarella']
        },
        {
            'breakfast': 'Breakfast burrito with scrambled eggs and vegetables',
            'lunch': 'Buddha bowl with quinoa, roasted vegetables, and tahini',
            'dinner': 'Baked cod with herb-roasted potatoes and steamed broccoli',
            'snacks': ['Fruit and nut bar', 'Veggie chips with guacamole', 'Greek yogurt parfait']
        },
        {
            'breakfast': 'Buckwheat pancakes with fresh fruit compote',
            'lunch': 'Falafel wrap with cucumber yogurt sauce',
            'dinner': 'Mushroom and barley risotto with side salad',
            'snacks': ['Date and oat energy balls', 'Vegetable smoothie', 'Whole grain crackers with cheese']
        },
        {
            'breakfast': 'Acai bowl with granola and coconut flakes',
            'lunch': 'Quinoa-stuffed bell peppers with black beans',
            'dinner': 'Grilled vegetables with polenta and fresh herbs',
            'snacks': ['Fresh fruit salad', 'Hummus with veggie sticks', 'Coconut yogurt with granola']
        },
        {
            'breakfast': 'Overnight oats with almond milk, chia seeds, and fresh berries',
            'lunch': 'Quinoa salad with black beans, corn, and avocado',
            'dinner': 'Stuffed eggplant with couscous and herbs',
            'snacks': ['Fruit salad', 'Carrot sticks with hummus', 'Nuts and seeds mix']
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
        },
        {
            'breakfast': 'Steel-cut oats with cinnamon and fresh apple',
            'lunch': 'Grilled chicken breast with quinoa and steamed vegetables',
            'dinner': 'Baked white fish with herbs and roasted root vegetables',
            'snacks': ['Fresh berries', 'Rice cakes with tomato', 'Fat-free yogurt with fruit']
        },
        {
            'breakfast': 'Egg white frittata with vegetables and herbs',
            'lunch': 'Turkey and vegetable soup with whole grain roll',
            'dinner': 'Steamed salmon with lemon and dill, quinoa, and asparagus',
            'snacks': ['Air-popped popcorn', 'Sliced cucumber with herbs', 'Fresh orange segments']
        },
        {
            'breakfast': 'Smoothie with banana, berries, and fat-free yogurt',
            'lunch': 'Vegetable barley soup with crusty bread',
            'dinner': 'Grilled vegetables with herbs and balsamic reduction',
            'snacks': ['Baked apple with cinnamon', 'Vegetable juice', 'Rice cakes with cucumber']
        },
        {
            'breakfast': 'Overnight oats with fresh fruit and cinnamon',
            'lunch': 'Lentil and vegetable curry with brown rice',
            'dinner': 'Roasted vegetable medley with quinoa and fresh herbs',
            'snacks': ['Fresh fruit smoothie', 'Vegetable broth with herbs', 'Baked sweet potato chips']
        },
        {
            'breakfast': 'Fruit smoothie with spinach, banana, and oat milk',
            'lunch': 'Vegetable soup with lentils and herbs',
            'dinner': 'Steamed tofu with sautéed greens and brown rice',
            'snacks': ['Apple slices', 'Cucumber sticks', 'Air-popped popcorn']
        }
    ]
}


def filter_meals(meals, dietary_habits, allergies, aversions):
    filtered = []
    prefer_nonveg = (dietary_habits.lower() == 'regular')

    for meal in meals:
        text = f"{meal['breakfast']} {meal['lunch']} {meal['dinner']} {' '.join(meal['snacks'])}".lower()

        if prefer_nonveg:
            pass

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


def make_varied_recommendations(patient_data, temperature=0.8):
    patient_df = pd.DataFrame([patient_data])
    patient_preprocessed = preprocessor.transform(patient_df)
    predictions = nn_model.predict(patient_preprocessed, verbose=0)
    meal_plan_probs, calories, protein, carbs, fats = predictions

    meal_plan_probs = meal_plan_probs[0]
    meal_plan_probs = np.log(meal_plan_probs + 1e-10) / temperature
    meal_plan_probs = np.exp(meal_plan_probs)
    meal_plan_probs = meal_plan_probs / np.sum(meal_plan_probs)

    allowed_meal_plans = {}
    for plan_name, meals in meal_plans.items():
        filtered = filter_meals(meals, patient_data['Dietary_Habits'], 'None', 'None')
        if filtered:
            allowed_meal_plans[plan_name] = filtered

    if not allowed_meal_plans:
        print("No allowed meal plans found based on diet. Defaulting to Balanced Diet.")
        allowed_meal_plans = {'Balanced Diet': meal_plans['Balanced Diet']}

    available_labels = [plan for plan in label_encoder.classes_ if plan in allowed_meal_plans]
    if not available_labels:
        available_labels = list(allowed_meal_plans.keys())

    available_indices = [i for i, cls in enumerate(label_encoder.classes_) if cls in available_labels]
    available_probs = [meal_plan_probs[i] for i in available_indices]

    if not available_indices or not available_probs or len(available_probs) != len(available_indices):
        print("No valid indices or mismatched probabilities. Defaulting to Balanced Diet.")
        meal_plan = 'Balanced Diet'
    else:
        total_prob = sum(available_probs)
        if total_prob == 0:
            available_probs = [1 / len(available_probs)] * len(available_probs)
        else:
            available_probs = [p / total_prob for p in available_probs]

        try:
            meal_plan_idx = np.random.choice(available_indices, p=available_probs)
            meal_plan = label_encoder.classes_[meal_plan_idx]
        except Exception as e:
            print(f"Random choice failed: {e}. Defaulting to Balanced Diet.")
            meal_plan = 'Balanced Diet'

    noise_factor = 0.05
    calories = float(calories[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    protein = float(protein[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    carbs = float(carbs[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))
    fats = float(fats[0][0]) * (1 + np.random.uniform(-noise_factor, noise_factor))

    calories = max(1500, min(3500, calories))
    protein = max(50, min(200, protein))
    carbs = max(50, min(400, carbs))
    fats = max(20, min(150, fats))
    
    if meal_plan not in allowed_meal_plans:
        print(f"Meal plan '{meal_plan}' not found in allowed plans. Falling back to Balanced Diet.")
        meal_plan = 'Balanced Diet'

    filtered_meals = filter_meals(
        allowed_meal_plans[meal_plan],
        patient_data['Dietary_Habits'],
        patient_data['Allergies'],
        patient_data['Food_Aversions']
    )

    if not filtered_meals:
        print("No filtered meals matched preferences. Falling back to unfiltered list.")
        detailed_plan = random.choice(allowed_meal_plans[meal_plan])
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


def save_to_mongodb(user_data, recommendations):
    if db is None:
        print("MongoDB connection not available. Data not saved.")
        return None
    
    try:
        
        user_id = user_data.get('user_id', str(uuid.uuid4()))
        
        
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
        
        
        users_collection.update_one(
            {'user_id': user_id},
            {'$set': user_document},
            upsert=True
        )
        
        
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
        
        
        height_m = data['height_cm'] / 100
        weight_kg = data['weight_kg']
        bmi = weight_kg / (height_m * height_m)
        data['bmi'] = round(bmi, 1)
        
        
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
        
        
        save_to_mongodb(data, recommendations)
        
        return jsonify({"success": True, "recommendations": recommendations})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/user-history', methods=['GET'])
def get_user_history():
    try:
        if db is None:
            return jsonify({"success": False, "error": "Database connection not available"})
        
        
        recommendations = list(recommendations_collection.find(
            {},  
            {"_id": 0}
        ).sort("timestamp", -1))
        
        for rec in recommendations:
            rec["timestamp"] = rec["timestamp"].isoformat()
        
        return jsonify({"success": True, "history": recommendations})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/form-data', methods=['GET'])
def get_form_data():
    return jsonify({
        "gender": ["Male", "Female", "Other"],
        "chronic_disease": ["None", "Diabetes", "Hypertension", "Heart Disease"],
        "genetic_risk_factor": ["Yes", "No"],
        "allergies": ["None", "Nuts", "Dairy", "Gluten"],
        "alcohol_consumption": ["Yes", "No"],
        "smoking_habit": ["Yes", "No"],
        "dietary_habits": ["Regular", "Vegetarian", "Vegan"],
        "preferred_cuisine": ["Mediterranean", "Asian", "Indian", "Western"],
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
    app.run(host='0.0.0.0', debug=True, port=5000)
