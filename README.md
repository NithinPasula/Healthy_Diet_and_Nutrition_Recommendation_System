<h1 align="center">🍽️ Healthy Diet and Nutrition Recommendation System</h1>
<p align="center">
 This repository contains the official implementation of the Healthy Diet and Nutrition Recommendation System, a machine learning-based web application that provides personalized meal plans and nutritional advice. Developed as part of the recruitment process at <strong>JTP Co. LTD.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20JS%20%7C%20Bootstrap-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Database-MongoDB-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Container-Docker-blue?style=flat-square" />
</p>

---

## 📋 Table of Contents

* [🔍 Overview](#-overview)
* [✨ Key Functionalities](#-key-functionalities)
* [🧠 Machine Learning Model](#-machine-learning-model)
* [📊 Architecture Diagram](#-architecture-diagram)
* [📂 File Structure](#-file-structure)
* [🧰 Tech Stack](#-tech-stack)
* [🛠️ Prerequisites](#️-prerequisites)
* [🚀 Installation Guide and User Manual](#-installation-guide-and-user-manual)
* [⚡ Quick Test Inputs](#-quick-test-inputs)
* [🖼️ UI Snapshots](#-ui-snapshots)
* [📨 Contact](#-contact)

---

## 🔍 Overview

The **Healthy Diet and Nutrition Recommendation System** processes user inputs such as medical history, daily habits, and current nutrition profile to output personalized macro-nutrient goals and meal plans across all meals of the day.

* Analyzes the user's **medical history**, **daily routine**, and **current diet**.
* Outputs **personalized macro-nutrient goals** and **categorized meal plans** (breakfast to snacks).
* Adjusts recommendations in real-time for both known and **first-time users**.

It is built to adapt to individuals with **chronic diseases**, **genetic risk factors**, or **dietary restrictions**, enabling inclusive and dynamic diet recommendations.

---

## ✨ Key Functionalities

* 🔐 **Custom User Input Forms** – Collect detailed demographic, lifestyle, and dietary inputs.
* 🧮 **Daily Nutritional Goals** – Suggest daily targets for calories, protein, fats, and carbohydrates.
* 🍱 **Complete Meal Plans** – Generate tailored breakfast, lunch, dinner, and snack options.
* 🔁 **Dynamicity & Adaptability** – Ensure variability in recommendations using Gaussian noise layers.
* ⚠️ **Health Condition Support** – Address users with chronic conditions or allergies.
* 📦 **Containerized Design** – Modular and deployable using Docker.

---

## 🧠 Machine Learning Model

The predictive system leverages a **multi-output feedforward neural network** with dedicated heads for regression and classification tasks:

* 🧱 **Shared Representation Layers**: Capture generalized health and lifestyle patterns.
* 🌿 **Separate Output Branches**: Handle macro-nutrient predictions and meal plan classification.
* 🌫️ **Gaussian Noise Layers**: Introduce small variability for better generalization across user profiles.

**Model Outputs:**

* 📊 Nutrient Goals: Calories, Protein, Carbs, Fats
* 🍽️ Meal Plan Category: Breakfast, Lunch, Dinner, Snacks

---

### 📊 Architecture Diagram

Below is the architecture diagram of the multi-output neural network used in the system. It showcases shared layers, dedicated branches for macro-nutrient regression and meal category classification, and the use of Gaussian noise layers for variability.

![Neural_Network Architecture](https://github.com/user-attachments/assets/ccd0e268-24ac-43c5-ad2b-7cb2e9da3d57)


---

## 📁 File Structure

```bash
📦 Healthy_Diet_and_Nutrition_Recommendation_System
├── Frontend/         # UI built with HTML5, JavaScript & Bootstrap
├── Backend/          # FastAPI logic + ML model implementation
├── Database/         # Dataset, MongoDB config & backend DB routes
```


---

## 🧰 Tech Stack

| Component      | Technology                                           |
| -------------- | ---------------------------------------------------- |
| **Frontend**   | HTML5, JS, Bootstrap                                 |
| **Backend**    | Python, FastAPI/Flask, TensorFlow/Keras              |
| **Modeling**   | Multi-output Neural Network                          |
| **Deployment** | Docker                                               |
| **Dataset**    | https://www.kaggle.com/datasets/ziya07/personalized-medical-diet-recommendations-dataset |

---

## 🛠️ Prerequisites

Ensure the following tools are installed on your system:

- ✅ [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- ✅ Docker Compose (bundled with Docker Desktop)
- ✅ [Git](https://git-scm.com/)
- ✅ [MongoDB Compass](https://www.mongodb.com/try/download/compass) – for database inspection

---

## 🚀 Installation Guide and User Manual

### 🔧 Installation Steps

1. Ensure Docker Desktop is installed and running.
2. Clone the repository:

   ```bash
   git clone https://github.com/NithinPasula/Healthy_Diet_and_Nutrition_Recommendation_System.git
   cd Healthy_Diet_and_Nutrition_Recommendation_System
   ```
3. Build and run the containers:

   ```bash
   docker compose up --build
   ```
4. Visit the application in your browser at:

   ```
   http://localhost:3000
   ```
   This URL will load the Healthy Diet and Nutrition Recommendation System interface.

5. Database Access

- To examine the data, open **MongoDB Compass** and connect using the following URI:

  ```bash
  mongodb://localhost:27018

---

| Name                       | Description                                       |
| -------------------------- | ------------------------------------------------- |
| **nutrition\_recommender** | Name of the database                              |
| **recommendations**        | Collection to store the generated recommendations |
| **users**                  | Collection to store user input data               |

**Images for Reference:**
![Screenshot 2025-05-29 194743](https://github.com/user-attachments/assets/7311adef-54bf-44da-91d1-0126381cfc34)
---
![Screenshot 2025-05-29 194826](https://github.com/user-attachments/assets/67592f72-ebb4-4a2f-ae38-18ab4ddea841)
---
![Screenshot 2025-05-29 194808](https://github.com/user-attachments/assets/5b2bed49-493b-42d8-8f56-fb3757ba236a)


---

## ⚡ Quick Test Inputs

Use the following sample user input to quickly test the system:

| Parameter                     | Value             |
|------------------------------|-------------------|
| Age                          | 32                |
| Gender                       | Male              |
| Height (cm)                  | 181               |
| Weight (kg)                  | 95                |
| Chronic Disease              | Diabetes          |
| Blood Pressure (Systolic)    | 120               |
| Blood Pressure (Diastolic)   | 80                |
| Cholesterol Level            | 180               |
| Blood Sugar Level            | 95                |
| Genetic Risk Factor          | No                |
| Allergies                    | None              |
| Daily Steps                  | 8000              |
| Exercise Frequency (days/week)| 3                |
| Sleep Hours                  | 7                 |
| Alcohol Consumption          | Yes               |
| Smoking Habit                | No                |
| Dietary Habits               | Regular           |
| Current Caloric Intake       | 3000              |
| Current Protein Intake (g)   | 20                |
| Current Carbohydrate Intake (g) | 350           |
| Current Fat Intake (g)       | 150               |
| Preferred Cuisine            | Indian            |
| Food Aversions               | None              |

---

### 🧑‍🏫 User Manual

1. **Open Application**: Go to `http://localhost:3000`
2. **Fill the Form**: Provide age, gender, health conditions, current diet, activity level, etc.
3. **Submit**: Click the "Get Recommendation" button.
4. **View Output**: The result page shows:

   * Recommended macro-nutrients (calories, protein, carbs, fats)
   * A full day’s meal plan (Breakfast, Lunch, Dinner, Snacks)

---

## 🖼️ UI Snapshots

### 🔹 User Input Page

![User Input Page](https://github.com/user-attachments/assets/13e0f52b-b0ad-4eaf-8655-6d8b2e306433)

---

### 🔹 Prediction Result Page

![Prediction Result 1](https://github.com/user-attachments/assets/1d2b8d91-be3e-4107-801c-3362cc524bd1)

---

![Prediction Result 2](https://github.com/user-attachments/assets/0d99412b-148b-41c7-81da-7c1fc8328fba)

---

![Prediction Result 3](https://github.com/user-attachments/assets/c9c2a52a-e620-4f4c-9020-b0d9297059dd)

---

### 🔹 View Past Recommendations

![Past Recommendations 1](https://github.com/user-attachments/assets/0be2cb1b-c5db-4d86-b051-3f6bb4588286)

---

![Past Recommendations 2](https://github.com/user-attachments/assets/98e2e920-20a0-4c22-aa66-60ecb46d007f)

---

## 📨 Contact

For suggestions or technical queries, please reach out to:

**Developer:** Pasula Nithin <br>
**Email:**
[22e55a0510@hitam.org](mailto:22e55a0510@hitam.org)

---
