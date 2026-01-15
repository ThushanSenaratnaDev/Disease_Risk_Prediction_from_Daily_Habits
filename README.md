# Disease Risk Prediction from Daily Habits ⚕️

## Project Overview

This repository hosts a machine learning-powered application designed to predict an individual's **disease risk** based on their daily habits and lifestyle factors. By leveraging a trained model, the application provides users with an accessible interface to assess their risk level and gain insights into the factors influencing their health outcomes.

The project is structured into three main components:
1.  **Model Training:** A Jupyter notebook for data processing, model selection, and training (using XGBoost).
2.  **Backend API:** A lightweight server (FastAPI/Uvicorn) to host the prediction model.
3.  **Frontend Application:** An interactive web interface (Streamlit) for user input and result visualization.

***

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Machine Learning** | **XGBoost** | The primary algorithm used for disease risk classification/prediction. |
| **Backend API** | **FastAPI** / **Uvicorn** | Provides a fast, modern API endpoint to serve the machine learning model predictions. |
| **Frontend UI** | **Streamlit** | Creates the interactive, user-friendly web application for input collection and result display. |
| **Data/Training** | **Python** / **Jupyter Notebook** | Used for model development, data analysis, and preparation. |

***

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

You need **Python 3.7+** installed on your system. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
