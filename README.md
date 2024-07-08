# Assignment-Predictive-Modeling-for-Digital-Marketing-Campaign-Performance

# Methodology
Developed and implemented a comprehensive data analysis and machine learning pipeline to predict and improve the performance of digital marketing campaigns. Utilized advanced data processing, visualization, and modeling techniques to derive actionable insights and recommendations.

Objective: To develop a data analysis and machine learning pipeline for predicting and improving digital marketing campaign performance.
Data Loading and Basic Information: Loaded the dataset and performed initial checks.
Exploratory Data Analysis (EDA): Visualized data distributions and relationships between variables using various plots.
Data Preprocessing: Handled missing values, encoded categorical variables, and standardized numeric features.
Feature Engineering: Created new features based on existing ones to enhance predictive power.
Target Identification: Selected the performance metric with the highest correlation for prediction.
Model Training and Evaluation: Split data into training and testing sets, trained a Gradient Boosting Regressor model, and evaluated performance using MSE and R² scores.
Feature Importance Analysis: Identified key factors influencing campaign performance.
Visualization of Results: Created various plots to present findings effectively.
Recommendations and Insights: Provided actionable recommendations based on feature importance and EDA findings.
Tools and Technologies: Python, Pandas, Seaborn, Matplotlib, Scikit-learn, Gradient Boosting Regressor, GridSearchCV.


Data Loading and Inspection: The dataset was loaded and inspected for basic information such as data structure, summary statistics, and missing values.

Exploratory Data Analysis (EDA): Visualizations including histograms, box plots, and a correlation matrix were generated to understand the distribution and relationships among variables.

![Screenshot 2024-07-07 135933](https://github.com/RATHISHBARATH/Assignment-Predictive-Modeling-for-Digital-Marketing-Campaign-Performance/assets/94107495/fecd95ce-fc19-4392-910e-f705dfc49d93)

Feature Engineering: New features were created based on interactions between numeric variables to potentially enhance predictive power.

Model Training: A Gradient Boosting Regressor model was trained using GridSearchCV for hyperparameter tuning to predict the target variable.

![Screenshot 2024-07-07 140258](https://github.com/RATHISHBARATH/Assignment-Predictive-Modeling-for-Digital-Marketing-Campaign-Performance/assets/94107495/ef15a646-7207-4417-a4e7-92a0d627c772)


Model Evaluation: The model was evaluated using cross-validation scores (MSE), test mean squared error, and R2 score to assess performance.
Findings:

Data Distribution: Numeric variables showed skewed distributions, and some outliers were identified through box plots.

Correlation Analysis: Certain numeric variables exhibited moderate to strong correlations, suggesting potential multicollinearity.

Categorical Insights: Categorical variables showed varying distributions, which were analyzed against the target variable to derive insights.

Model Performance: The trained model demonstrated robust performance with a cross-validation MSE score, indicating good predictive capability. The R2 score further confirmed the model’s explanatory power.

Recommendations: Feature Importance: Focus on enhancing features identified as crucial by the model, such as Feature A and Feature B, which significantly influence the target variable.
Optimization Strategies: Explore strategies to handle outliers in numeric variables to potentially improve model robustness.
Further Analysis: Conduct deeper analysis into categorical variables to uncover hidden patterns that could enhance segmentation strategies.
