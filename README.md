# Customer Churn Prediction

This repository contains the dataset and codebase for the Customer Churn Prediction project.

## Project Structure
- `data/raw/`: Contains the raw CSV datasets (`train.csv`, `test.csv`, `sample_submission.csv`). Note: The raw data might be excluded from Git history to manage repository size.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and model training experiments.
- `src/`: Python source code containing custom modules for data processing and modeling.
- `app.py`: Streamlit application to host the final predictive model.
- `requirements.txt`: Required Python dependencies.

## Setup & Running the Streamlit App
1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
