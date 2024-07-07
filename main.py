import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from google.colab import files
import io  # Required for file handling in Colab

# Define custom functions

# EDA functions
def load_data():
    try:
        uploaded = files.upload()
        file_name = next(iter(uploaded))
        df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_info(df):
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())

def correlation_matrix(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_distributions(df, numeric_columns):
    for col in numeric_columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_boxplots(df, numeric_columns):
    for col in numeric_columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

def plot_categorical_attributes(df, categorical_attributes):
    for col in categorical_attributes:
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_categorical_vs_performance(df, categorical_attributes, target):
    for cat_col in categorical_attributes:
        sns.boxplot(x=df[cat_col], y=df[target])
        plt.title(f'{target} vs {cat_col}')
        plt.show()

# Preprocessing functions
def handle_missing_values(df, numeric_columns):
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def encode_categorical_variables(df, categorical_attributes):
    label_encoders = {}
    for col in categorical_attributes:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def standardize_features(df, numeric_columns):
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler

def feature_engineering(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) >= 2:
        df['NewFeature'] = df[numeric_columns[0]] * df[numeric_columns[1]]
    return df

# Modeling functions
def split_data(df, features, target, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return -np.mean(cv_scores), mse, r2

def feature_importance(model, features):
    importances = model.feature_importances_
    return dict(zip(features, importances))

# Additional Plot Functions
def scatter_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def bar_graph(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[x_col], y=df[y_col])
    plt.title(f'Bar Graph: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.show()

def line_plot(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
    plt.title(f'Line Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

def pie_chart(df, column):
    plt.figure(figsize=(8, 8))
    counts = df[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart: Distribution of {column}')
    plt.show()

def heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Heatmap of Correlation Matrix')
    plt.show()

def density_plot(df, column):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[column], shade=True)
    plt.title(f'Density Plot: {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

def violin_plot(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[x_col], y=df[y_col])
    plt.title(f'Violin Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.show()

def bubble_chart(df, x_col, y_col, bubble_size_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], size=df[bubble_size_col], sizes=(20, 200), legend=False)
    plt.title(f'Bubble Chart: {x_col} vs {y_col} (Bubble Size: {bubble_size_col})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def contour_plot(df, x_col, y_col):
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=df[x_col], y=df[y_col], cmap='viridis', shade=True, bw_method='silverman')
    plt.title(f'Contour Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def dot_plot(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=df[x_col], y=df[y_col], jitter=True, alpha=0.6)
    plt.title(f'Dot Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def create_presentation(methodology_content, findings_content, recommendations_content, img_paths):
    """
    Function to create a presentation deck highlighting key points from analysis and model results.
    """
    # Example implementation to create presentation deck
    pass

def identify_target_column(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric columns found to identify target.")

    correlation_matrix = df[numeric_columns].corr()
    target_candidates = correlation_matrix.abs().sum().sort_values(ascending=False).index
    # Assuming the column with the highest sum of correlations with other columns as target
    potential_target = target_candidates[0]
    return potential_target

# Main script

# Load the dataset
df = load_data()

if df is not None:
    # Perform basic information checks
    basic_info(df)

    # Define performance metrics and categorical attributes dynamically
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_attributes = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(numeric_columns) > 0:
        plot_distributions(df, numeric_columns)
        plot_boxplots(df, numeric_columns)
    if len(categorical_attributes) > 0:
        plot_categorical_attributes(df, categorical_attributes)
    if len(numeric_columns) > 0:
        correlation_matrix(df)

    # Identify the target column dynamically
    target = identify_target_column(df)
    print(f"Identified target column: {target}")

    # Handle missing values and encode categorical variables
    df = handle_missing_values(df, numeric_columns)
    df, label_encoders = encode_categorical_variables(df, categorical_attributes)

    # Standardize features and perform feature engineering
    df, scaler = standardize_features(df, numeric_columns)
    df = feature_engineering(df)

    # Plot categorical attributes against performance target
    if len(categorical_attributes) > 0:
        plot_categorical_vs_performance(df, categorical_attributes, target)

    # Split the data into training and testing sets
    features = df.columns[df.columns != target].tolist()
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    cv_score, mse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"Cross-Validation Score (MSE): {cv_score}")
    print(f"Test Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Determine feature importance
    importances = feature_importance(model, features)
    print('Feature Importances:')
    for feature, importance in importances.items():
        print(f'{feature}: {importance}')

    # Generate recommendations and insights
    print("\nRecommendations for Future Digital Marketing Campaigns:")
    sorted_insights = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    for feature, importance in sorted_insights[:5]:
        print(f"- Focus on improving {feature}, as it has a high impact on campaign performance (importance score: {importance:.2f}).")
else:
    print("Failed to load data.")
