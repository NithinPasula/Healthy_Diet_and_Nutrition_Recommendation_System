
<h1 align="center">🍽️ Personalized Nutrition & Diet Recommendation System</h1>
<p align="center">
  This repository hosts the official implementation of the Personalized Nutrition & Diet Recommendation System that provides customized meal plans and nutritional recommendations using advanced machine learning techniques.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20JS%20%7C%20Bootstrap-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Database-MongoDB-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Container-Docker-blue?style=flat-square" />
</p>

---

## 📖 Overview

**Personalized Nutrition & Diet Recommendation System** is an intelligent dietary assistant that dynamically generates personalized meal plans and nutritional targets based on a user's physiological data, medical history, and lifestyle habits. 

It leverages a **multi-output feedforward neural network** to deliver real-time, adaptive recommendations.

> 👨‍💻 Developed by **Pasula Nithin**  
> 📧 [22e55a0510@hitam.org](mailto:22e55a0510@hitam.org)

---

## 🧠 Key Features

- 🎯 Tailored meal plans & macro recommendations
- 🧬 Machine learning–based prediction engine
- ⚡ FastAPI backend integrated with TensorFlow
- 🖥️ Responsive HTML5 + Bootstrap frontend
- 🐳 Fully containerized using Docker

---

## 📁 Project Structure

```bash
📦 Healthy_Diet_and_Nutrition_Recommendation_System
├── Frontend/         # UI built with HTML5, JavaScript & Bootstrap
├── Backend/          # FastAPI logic + ML model implementation
├── Database/         # Dataset, MongoDB config & backend DB routes
```

---

## 🛠️ Prerequisites

Ensure the following tools are installed on your system:

- ✅ [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- ✅ Docker Compose (bundled with Docker Desktop)
- ✅ [Git](https://git-scm.com/)

---

## 🚀 Getting Started

### 🔁 Step 1: Clone the Repository

```bash
git clone https://github.com/NithinPasula/Healthy_Diet_and_Nutrition_Recommendation_System.git
cd Healthy_Diet_and_Nutrition_Recommendation_System
```

### 🛠 Step 2: Start the Application

```bash
docker compose up --build
```

### 🌐 Step 3: Access the System

Visit the following URL in your browser:

```
http://localhost:3000
```

---

## 🧬 Machine Learning Model

- **Model Type:** Multi-output Feedforward Neural Network  
- **Architecture:** Shared hidden layers with branched outputs  
- **Predicted Outputs:**  
  - 🥗 Recommended Meal Category  
  - 🔢 Macronutrient Targets: Calories, Carbs, Protein, Fats  
- **Input Parameters:**  
  - 20+ features: age, gender, health conditions, allergies, preferences, activity level, etc.

---

## 💻 Tech Stack

| Layer        | Technologies Used                            |
|--------------|-----------------------------------------------|
| 🎨 Frontend  | HTML5, JavaScript, Bootstrap 5                |
| ⚙️ Backend   | Python, FastAPI                               |
| 🧠 ML Engine | TensorFlow, Keras                             |
| 🗃 Database   | MongoDB                                       |
| 📦 DevOps    | Docker & Docker Compose                       |

---

## 🗃 MongoDB Access

To view personalized diet recommendations directly from the MongoDB database, connect with the following URI:

```
mongodb://localhost:27018/
```

You can use tools like MongoDB Compass or Mongo shell to inspect the data.

---

## 📌 Dataset Source

Dataset used for training is publicly available at:  
🔗 [Kaggle - Personalized Medical Diet Dataset](https://www.kaggle.com/datasets/ziya07/personalized-medical-diet-recommendations-dataset)

---

## 📈 Project Status

✅ MVP Complete  
✅ ML Model Integrated  
✅ Containerized for easy deployment  
✅ Ready for production & future upgrades

---

## 📬 Contact

Have questions or want to collaborate?

**Pasula Nithin**  
📧 [22e55a0510@hitam.org](mailto:22e55a0510@hitam.org)  
🔗 [GitHub Profile](https://github.com/NithinPasula)

---
