Machine Learning Project

📌 Project Overview

This project presents a complete end-to-end machine learning pipeline built with CatBoost, a high-performance gradient boosting library for tabular data. The goal is to create a reliable, easy-to-follow workflow for predicting Airbnb rental prices based on various listing features.

The dataset used is the Airbnb Rent Prediction 2025
 dataset from Kaggle, which includes details such as property location, type, number of rooms, host information, and other features that influence rental pricing.
This makes it an ideal dataset for regression modeling and for showcasing CatBoost’s ability to handle mixed data types with minimal preprocessing.

CatBoost was chosen because it:

 - Handles categorical variables seamlessly

 - Requires minimal data cleaning

 - Reduces overfitting through built-in regularization

 - Offers interpretability and visualization tools

 - Performs efficiently even on large tabular datasets

The included Jupyter Notebook guides you through every step — from data exploration and preprocessing to model training, evaluation, and deployment.

🎯 Objectives

Build a dependable CatBoost regression model for rent prediction

Create a transparent and reproducible machine learning workflow

Generate meaningful visualizations and performance metrics

Ensure the model is reusable and easy to extend in future projects

📂 Repository Structure
.
├── notebook.ipynb          # Main notebook with the full ML workflow
├── data/                   # Dataset storage
├── models/                 # Saved CatBoost models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🧠 Workflow Summary
1. Data Loading & Exploration

Import Airbnb data from Kaggle

Explore dataset structure and summary statistics

Identify categorical and numerical features

Handle missing values and check target distribution

2. Preprocessing

Even though CatBoost needs minimal preprocessing, this step ensures data quality through:

Cleaning and formatting columns

Handling missing or inconsistent entries

Splitting data into training and validation sets

3. Model Development

A CatBoostRegressor is used for rental price prediction with tunable parameters such as:

iterations, learning_rate, depth, loss_function, and eval_metric

Optional GPU acceleration and early stopping

4. Model Training

Includes:

Iterative logging of performance

Real-time visualization of learning curves

Automatic handling of categorical variables

5. Model Evaluation & Validation

Evaluation covers:

Metrics: RMSE, MAE, and R² score

Feature importance visualization

Training vs. validation loss analysis

Prediction vs. actual comparisons

6. Model Saving & Reuse

Save your model:

model.save_model("models/catboost_model.cbm")


Load it later:

from catboost import CatBoost
model = CatBoost()
model.load_model("models/catboost_model.cbm")

⚙️ Installation & Setup
1. Clone the Repository
git clone <your-repo-url>
cd <project-folder>

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

3. Install Dependencies
pip install -r requirements.txt


Or install manually:

pip install catboost numpy pandas scikit-learn matplotlib

▶️ Using the Notebook

Launch Jupyter Notebook:

jupyter notebook


Open notebook.ipynb

Adjust dataset paths if needed

Run the cells step by step — each section includes explanations and outputs

📊 Outputs & Visualizations

The notebook generates:

Feature importance plots

Training and validation loss curves

Evaluation metric summaries

Sample predictions and error visualizations

Saved model files ready for reuse

🔧 Customization Options

You can easily extend the project by adding:

Hyperparameter tuning (GridSearch, RandomizedSearch, Optuna)

Cross-validation using CatBoost tools

Integration with scikit-learn pipelines

Streamlit or FastAPI deployment for web use

Automated feature engineering steps

📈 Future Enhancements

Potential improvements include:

Comparing CatBoost with XGBoost or LightGBM

Generating automated EDA reports

Building a training script for CLI usage

Adding Docker support for reproducible environments

👥 Team Contributions

This project was built through great teamwork and collaboration:

Sakshitha – Handled data preprocessing, ensuring data consistency, cleaning, and formatting.

Noura – Designed and fine-tuned the CatBoost model, focusing on hyperparameter optimization and performance analysis.

Tom – Developed the Streamlit interface, making model predictions and visualizations interactive and user-friendly.

Together, the team created a seamless workflow that combines data preparation, model building, and deployment in one coherent system.

🤝 Contributing

Contributions are welcome!
You can:

Improve documentation or add new features

Enhance the model or introduce new visualizations

Extend deployment options or EDA capabilities

Please submit a pull request or open an issue to collaborate.

📜 License

This project is released for personal and educational use.
Feel free to modify, extend, and build upon it — with credit to the original contributors
