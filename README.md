# Assignment-Predictive-Modeling-for-Digital-Marketing-Campaign-Performance

# Methodology


Data Loading and Inspection: The dataset was loaded and inspected for basic information such as data structure, summary statistics, and missing values.

Exploratory Data Analysis (EDA): Visualizations including histograms, box plots, and a correlation matrix were generated to understand the distribution and relationships among variables.

Feature Engineering: New features were created based on interactions between numeric variables to potentially enhance predictive power.

Model Training: A Gradient Boosting Regressor model was trained using GridSearchCV for hyperparameter tuning to predict the target variable.

Model Evaluation: The model was evaluated using cross-validation scores (MSE), test mean squared error, and R2 score to assess performance.
Findings:

Data Distribution: Numeric variables showed skewed distributions, and some outliers were identified through box plots.

Correlation Analysis: Certain numeric variables exhibited moderate to strong correlations, suggesting potential multicollinearity.

Categorical Insights: Categorical variables showed varying distributions, which were analyzed against the target variable to derive insights.

Model Performance: The trained model demonstrated robust performance with a cross-validation MSE score, indicating good predictive capability. The R2 score further confirmed the model’s explanatory power.

Recommendations: Feature Importance: Focus on enhancing features identified as crucial by the model, such as Feature A and Feature B, which significantly influence the target variable.
Optimization Strategies: Explore strategies to handle outliers in numeric variables to potentially improve model robustness.
Further Analysis: Conduct deeper analysis into categorical variables to uncover hidden patterns that could enhance segmentation strategies.
