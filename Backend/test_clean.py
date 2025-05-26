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

np.random.seed(random.randint(1, 10000))
tf.random.set_seed(random.randint(1, 10000))

df = pd.read_csv('../Database/patient.csv')

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

for col in categorical_features:
    df[col] = df[col].fillna('None')
for col in numerical_features:
    df[col] = df[col].fillna(df[col].median())

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


label_encoder = LabelEncoder()
df['Meal_Plan_Encoded'] = label_encoder.fit_transform(df[target_meal_plan])
meal_plan_classes = label_encoder.classes_


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

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
input_shape = X_train_preprocessed.shape[1]

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
            if any(non_veg in text for non_veg in ['chicken', 'fish', 'salmon', 'beef', 'turkey', 'shrimp', 'pork', 'bacon', 'pepperoni']):
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

        if aversions.lower() == "spicy":
            if "spicy" in text:
                continue
        elif aversions.lower() == "sweet":
            if any(sugar in text for sugar in ['honey', 'syrup', 'sweet']):
                continue
        elif aversions.lower() == "sour":
            if "sour" in text:
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
            if key in numeric_ranges:
                low, high = numeric_ranges[key]
                prompt = f"{key} ({low}-{high}): "
            elif key in valid_categorical:
                options = ", ".join(valid_categorical[key])
                prompt = f"{key} ({options}): "
            else:
                prompt = f"{key}: "

            value = input(prompt).strip()
            if not value:
                patient[key] = default
                break

            
            if key in numerical_features:
                try:
                    value = float(value)
                    low, high = numeric_ranges.get(key, (None, None))
                    if low is not None and not (low <= value <= high):
                        print(f"{key} must be between {low} and {high}.")
                        continue
                    
                    if key == 'Age' and not (18 <= value <= 100):
                        print("Age must be between 18 and 100.")
                        continue
                    patient[key] = value
                    break
                except ValueError:
                    print(f"{key} must be a number.")

            else:
                if value in valid_categorical.get(key, [value]):
                    patient[key] = value
                    break
                print(f"Invalid {key}. Choose from: {valid_categorical[key]}")
    return patient


model_file = 'nutrition_model_v2.pkl'
old_model_file = 'nutrition_model.pkl'
use_pretrained = False


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
