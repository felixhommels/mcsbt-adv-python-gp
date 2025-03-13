# Food Delivery Prediction Project

## Overview

This project is a Flask web application that predicts food delivery times based on various input parameters. It utilizes a machine learning model trained on historical delivery data to provide accurate predictions. The application also includes statistical analysis features and a webhook for continuous integration and deployment.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Machine Learning Model](#machine-learning-model)
- [Webhook for CI/CD](#webhook-for-cicd)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- Python 3.12.4
- Flask
- Pandas
- NumPy
- Scikit-learn
- Git (for version control)
- PythonAnywhere (for deployment)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/felixhommels/mcsbt-adv-python-gp.git
   cd mcsbt-adv-python-gp
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:

   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

2. Open your web browser and navigate to `http://127.0.0.1:8000` to access the application.

## Endpoints

### Home Route

- **GET** `/`
  - Renders the welcome page.

### Prediction Route

- **GET** `/predict`
  - Renders the prediction form.
- **POST** `/predict`
  - Accepts input parameters in the request body and returns the predicted delivery time.

### Statistics Route

- **GET** `/statistics`
  - Accepts query parameters for weather, traffic, time of day, vehicle type, and courier experience to return statistical data.

### Order ID Route

- **GET** `/data/<int:order_id>`
  - Fetches information for a specific order based on the provided Order ID.

### Webhook Route

- **POST** `/webhook`
  - Handles incoming webhook notifications from GitHub to automatically pull the latest changes from the repository.

## Machine Learning Model

The machine learning model is built using scikit-learn's Linear Regression. The model is trained on historical delivery data, and the following steps are performed:

1. Load the dataset and preprocess it by dropping unnecessary columns and handling missing values.
2. Convert categorical variables into numerical format using one-hot encoding.
3. Split the data into training and testing sets.
4. Train the Linear Regression model and evaluate its performance using metrics like Mean Squared Error and R² Score.
5. Save the trained model to a pickle file for later use in the Flask application.

## Webhook for CI/CD

The webhook route allows for continuous integration and deployment. It listens for incoming webhook notifications from GitHub and automatically pulls the latest changes from the repository when a new commit is made. This ensures that the application is always up-to-date with the latest code changes.
