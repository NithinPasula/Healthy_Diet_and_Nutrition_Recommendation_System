# **Personalized Nutrition & Diet Recommendation System**

This repository hosts the official implementation of the **Personalized Nutrition & Diet Recommendation System** that provides customized meal plans and nutritional recommendations using advanced machine learning techniques.

Created by **[Pasula Nithin]** 📧 [22e55a0510@hitam.org]

## **📌 Project Overview**

This project provides a personalized nutrition experience using a **Multi-output Feedforward Neural Network** to dynamically match user health profiles with optimal dietary recommendations. It is designed to deliver real-time, adaptive meal plans and macronutrient targets based on individual physiological attributes, medical history, and lifestyle factors.

## **📁 Repository Structure**

* `frontend/` – Contains the HTML5 + JavaScript frontend application with Bootstrap styling.
* `backend/` – Contains the FastAPI backend server logic and neural network recommendation engine.
* `database/` – Holds configuration related to MongoDB and data storage.

## **⚙️ Installation & User Manual**

Follow the steps below to install and run the project locally:

1. **Clone the Repository**
   * Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/NithinPasula/Healthy_Diet_and_Nutrition_Recommendation_System.git
cd Healthy_Diet_and_Nutrition_Recommendation_System
```

2. **Create a `.env` File in Backend**
   * Navigate to the `backend` directory:

```bash
cd backend
```

   * Create a file named `.env` with the following configuration:

```
MONGO_URI=mongodb://mongo:27017
DB_NAME=nutrition_recommender
```

3. **Return to Root Directory**
   * Navigate back to the root directory of the project:

```bash
cd ..
```

4. **Start the Application**
   * Run the containerized application using Docker Compose:

```bash
docker compose up --build
```

5. **Access the Application**
   * Visit the following URL in your browser to start using the system:

```
http://localhost:3000
```

## **📦 Tech Stack**

* **Frontend:** HTML5 + Vanilla JavaScript + Bootstrap 5
* **Backend:** Python (FastAPI)
* **Database:** MongoDB
* **Machine Learning:** TensorFlow/Keras, Multi-output Neural Network
* **Containerization:** Docker & Docker Compose
* **Dataset Source:** Personalized Medical Diet Recommendations Dataset

## **🧠 Machine Learning Model**

* **Model Type:** Multi-output Feedforward Neural Network
* **Architecture:** Shared hidden layers with separate output branches
* **Outputs:** Meal plan classification + Nutritional targets (calories, protein, carbs, fats)
* **Input Features:** 20+ health and lifestyle parameters including demographics, health metrics, lifestyle factors, and dietary preferences

## **🚧 Project Status**

This project is a complete, containerized solution ready for deployment. The system successfully generates personalized nutrition recommendations based on comprehensive user health profiles.

## **📬 Contact**

For any queries or further information, feel free to reach out:

**[Pasula Nithin]** 📧 [22e55a0510@hitam.org]
