# 🚦 Kanakapura Road Traffic Forecaster

Live Web App: https://traffictimepredictionmodel-x9geeggzqnrlmgawhuqqcy.streamlit.app/

## 📌 Project Overview
The Kanakapura Road Traffic Forecaster is a machine learning web application designed to predict evening peak hour commute times. Built as a Mathematics and Statistics academic project, the model utilizes real world street friction data to simultaneously forecast the Estimated Time of Arrival (ETA) for both 500 meter and 1 kilometer milestones.

## 🔬 Mathematical Methodology & Data Processing
This project implements Multi Output Multiple Linear Regression to predict continuous time variables. During development, several real world data science challenges were addressed to ensure mathematical validity:

Multicollinearity Resolution: High correlation between Light Vehicles (LVs) and Heavy Vehicles (HVs) skewed initial coefficients. This was resolved by engineering a unified Total Vehicles feature, allowing the model to accurately weigh the impact of overall traffic density.

Outlier Mitigation: Extreme anomalies (e.g., a 20 minute localized gridlock event) were filtered out to prevent severe skewing of the regression plane and stabilize the R squared accuracy score.

Zero Intercept Regression: To prevent the dummy variable trap from producing negative base times for certain vehicle classes, the model was trained with fit_intercept=False. This forced the algorithm to calculate true, positive baseline seconds for every vehicle type.

Algorithmic Constraints (The "Physics Floor"): Hardcoded application logic ensures that the 1km ETA can never be mathematically lower than the 500m ETA plus the bottleneck delay, preventing extrapolation into negative time.

## ⚙️ Tech Stack
Backend: Python, Scikit Learn, Pandas, NumPy, Joblib
Frontend: Streamlit
Deployment: Streamlit Community Cloud

## 🚀 How to Run Locally

1. Clone the repository:
git clone https://github.com/yourusername/kanakapura-traffic-ai.git
cd kanakapura-traffic-ai

2. Create a virtual environment & install dependencies:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run the Streamlit application:
streamlit run app.py

## 👨‍💻 Author
Rudransh Choubey
B.Tech Artificial Intelligence & Machine Learning
📧 Contact: rudranshchoubey@outlook.com
