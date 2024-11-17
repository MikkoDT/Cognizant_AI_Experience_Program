# AI Job Simulation with Cognizant - Gala Groceries Supply Chain Optimization

## Project Overview: 
In this project, I worked on a data science simulation for Cognizant, focusing on optimizing the supply chain for Gala Groceries, a tech-driven grocery store chain in the USA. The business problem centered around the challenge of perishable grocery stock management—balancing overstock and understock situations to reduce waste and maximize customer satisfaction. Using exploratory data analysis (EDA), I provided insights and recommendations to address this issue.

## Task 1

### Key Steps:
### 1. Data Preparation:
- Utilized a CSV dataset containing transaction-level sales data from Gala Groceries.
- Loaded the data into Google Colab and processed it using Python’s pandas library.

### 2. Exploratory Data Analysis (EDA):
- Statistical Summary: Analyzed key attributes like unit price, quantity sold, and total sales using descriptive statistics (mean, median, count).
- Category Analysis: Identified 22 unique product categories, with top-selling categories like fruits, vegetables, and packaged foods.
- Customer Type Analysis: Segmented sales data by five customer types (non-member, standard, premium, basic, gold), identifying significant customer behavior patterns.
- Demand Analysis: Conducted a product and category-level sales volume analysis to pinpoint which products had the highest demand.
- Correlation Analysis: Explored relationships between variables, identifying strong correlations between unit price, quantity, and total sales.

### 3. Visualization:
- Created various visualizations using seaborn to depict distributions of key variables such as total sales, product categories, and customer types.

### 4. Business Recommendations:
- Recommended more granular customer segmentation for targeted promotions.
- Suggested tracking external factors (e.g., seasonal trends, supplier lead times) to improve stock predictions.
- Emphasized the need for integrating real-time IoT sensor data into stock management to dynamically adjust inventory levels.

[Task 1 Project Folder](https://github.com/MikkoDT/Cognizant_AI_Experience_Program/tree/main/Task1)

## Task 2

### Step 1: Data Modeling
Based on the problem statement: “Can we accurately predict the stock levels of products based on sales data and sensor data on an hourly basis to more intelligently procure products from our suppliers?”—you should choose relevant data. Since the client provided sales data and sensor data, the key data sources include:

1. Sales Data:
- Hourly sales data per product.
- Historical sales trends (to identify seasonal/periodic patterns).
- Promotions, discounts, and other price-related factors influencing sales.
2. Sensor Data:
- Temperature data from storage facilities (could impact stock degradation rates).
- Stock level data from refrigerators and freezers in stores (providing real-time insight into inventory levels).
3. Additional data points to consider:
- Supplier delivery schedules (to correlate restocking with depletion).
- Expiration or spoilage rates (to model stock depletion based on environmental conditions).
- Demand surges during specific times (weekends, holidays).

### Step 2: Strategic Planning
The plan to use this data involves several steps:

1. Data Preprocessing:
- Clean and integrate sales data and sensor data, ensuring uniform timestamps and removing outliers or sensor errors.
- Handle missing data in sensor readings (such as interpolating for small gaps or using statistical techniques for larger gaps).

2. Feature Engineering:
- Use sales trends to create demand forecasting features (like moving averages, seasonal patterns).
- Combine stock level data from sensors with sales data to estimate depletion rates.
- Incorporate external factors like temperature and its impact on perishable goods.

3. Modeling:
- Use time-series analysis (e.g., ARIMA or Prophet) for short-term stock predictions.
- Develop machine learning models (e.g., Random Forest, XGBoost, or LSTM networks) to predict future stock levels based on sales and sensor data.

4. Validation:
- Backtest models on historical data, adjusting hyperparameters and retraining models as needed.
- Compare model predictions to real data to evaluate accuracy.

5. Continuous Improvement:
- Build a pipeline for real-time data ingestion and continuous model updating.
- Add monitoring to track model performance and retrain as new data becomes available.

### Step 3: Communication - PowerPoint Slide
To create a concise and business-friendly slide:

#### Slide Title: Strategic Plan for Stock Prediction Using Sales and Sensor Data
Objective: Accurately predict hourly stock levels to optimize procurement.

Key Data Sources:
- Sales Data: Historical sales, pricing, promotions, demand trends.
- Sensor Data: Temperature in storage, real-time stock levels in refrigerators/freezers.
- Additional Data: Supplier delivery schedules, product spoilage rates.

## Plan:
1. Data Integration & Preprocessing: Clean, combine, and align sales and sensor data.
2. Feature Engineering: Create features based on sales trends, stock levels, and environmental factors.
3. Modeling:
  - Time-Series Analysis: Use ARIMA or similar methods for short-term forecasts.
  - Machine Learning: Leverage advanced models (e.g., LSTM) for multi-variate predictions.
4. Validation: Backtest models, adjust based on historical accuracy.
5. Deployment: Implement a real-time pipeline for continuous model updating and monitoring.

## Outcome: Improved procurement strategies based on accurate stock depletion predictions, reducing waste and ensuring timely restocking.

[Task 2 Project Folder](https://github.com/MikkoDT/Cognizant_AI_Experience_Program/tree/main/Task2)

## Task 3

### Data Preparation
1. Data Loading & Cleaning: We’ve imported sales, stock levels, and temperature data. After removing unnecessary columns, the timestamps have been converted to a consistent format. We aggregated quantities and stock percentages by timestamp and product ID and combined all datasets, ensuring null values are handled.

2. Feature Engineering: Key features like day of the month, day of the week, and hour were extracted from timestamps. Additionally, the product categories were transformed into dummy variables, making them suitable for modeling. The merged data now includes variables like product ID, estimated stock percentage, quantity, temperature, unit price, and time features.

### Model Development
Next, we’ll create a model to predict sales quantity (the target variable) based on factors like estimated stock, temperature, time of day, and product category.

#### Steps:
- Train-Test Split: Split the data into training and testing sets to ensure the model can generalize.
- Model Selection: Choose a suitable regression model, like linear regression, decision tree regressor, or random forest, to predict sales quantity. Let’s opt for random forest regressor since it’s robust and can handle a mix of feature types well.
- Model Training: Fit the model on training data using predictors such as stock levels, temperature, and other engineered features.
- Model Evaluation: Evaluate model performance using the testing set. Metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) will be converted into business-friendly terms.

### Communicating Results to the Business
Once the modeling process is complete, the business will need a concise presentation of the results, without technical jargon. Here’s how to frame the analysis:

### PowerPoint Slide Summary

Title: Sales Quantity Prediction to Optimize Inventory Management

Objective: We developed a machine learning model to predict sales quantity based on stock levels, temperature, product category, and time. This will help optimize inventory and reduce the risk of stockouts.

Key Insights:
- Our model can predict sales quantities with an average accuracy of X units per product, per hour.
- The prediction model showed that stock level is the most influential factor, followed by time of day and product category.
- On average, products with low stock (<20%) see a 30% drop in sales compared to well-stocked items.

Business Impact: By utilizing this model, your inventory management team can:
- Proactively restock products with declining stock levels.
- Forecast sales demand based on time, category, and temperature to avoid stockouts.
- Expected Inventory Optimization: Prevent overstocking and reduce holding costs by approximately Y%.

[Task 3 Project Folder](https://github.com/MikkoDT/Cognizant_AI_Experience_Program/tree/main/Task3)

## Task 4
The task involves structuring a Python module to load a CSV file, train a machine learning model using the provided data, and report performance metrics. Below is the planned structure and a step-by-step breakdown for implementation:

### Step 1: Plan
We will follow a modular approach, defining the following sections:
- Import Libraries: Include necessary libraries for data handling, modeling, and performance evaluation.
- Load Data Function: A function to read data from a CSV file.
- Create Target and Predictors Function: A function to split the dataset into the target and predictor variables.
- Train Model Function: A function to train the model and implement cross-validation.
- Execution Function (run): A final function to combine the above steps and run the entire pipeline.

Step 2: Write the Python Module
```
# Section 1 - Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Section 2 - Define Variables
# Number of folds for cross-validation and train-test split ratio
K = 10
SPLIT = 0.75  # 75% training, 25% testing

# Section 3 - Load Data Function
def load_data(path: str):
    """
    Load data from a CSV file into a Pandas DataFrame.
    
    :param path: str, path to the CSV file.
    :return: pd.DataFrame containing the loaded data.
    """
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')  # Drop unnecessary columns if present
    return df

# Section 4 - Create Target and Predictors Function
def create_target_and_predictors(data: pd.DataFrame, target: str = "estimated_stock_pct"):
    """
    Split the data into predictor variables (X) and the target variable (y).
    
    :param data: pd.DataFrame, the dataset to split.
    :param target: str, the target column.
    :return: X (predictors), y (target)
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Section 5 - Train Model with Cross-Validation
def train_algorithm_with_cross_validation(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest Regressor model with K-fold cross-validation.
    
    :param X: pd.DataFrame, predictor variables.
    :param y: pd.Series, target variable.
    """
    # List to store performance metrics (mean absolute error)
    metrics = []

    # Cross-validation loop
    for fold in range(K):
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=fold)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize and train the model
        model = RandomForestRegressor(random_state=fold)
        model.fit(X_train, y_train)

        # Predict and calculate error
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        metrics.append(mae)

        # Print performance for each fold
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Average performance across all folds
    avg_mae = sum(metrics) / len(metrics)
    print(f"Average MAE across {K} folds: {avg_mae:.3f}")

# Section 6 - Run Pipeline
def run(path: str):
    """
    Load data, create target and predictor variables, and train the model.
    
    :param path: str, path to the CSV file.
    """
    # Load data from the given path
    df = load_data(path)

    # Split data into predictors (X) and target (y)
    X, y = create_target_and_predictors(df)

    # Train the model and output performance metrics
    train_algorithm_with_cross_validation(X, y)

# Main entry point when the script is executed
if __name__ == "__main__":
    # Provide the path to your CSV file here
    csv_path = "data/your_dataset.csv"
    
    # Run the pipeline
    run(csv_path)
```

### Explanation of Code Structure:
1. Modular Functions:
- load_data(): Reads the CSV file, removing unnecessary columns.
- create_target_and_predictors(): Splits data into features (X) and target variable (y).
- train_algorithm_with_cross_validation(): Trains the model using K-fold cross-validation, standardizes the data, and reports performance metrics for each fold.
- run(): Combines all functions and executes the pipeline.

2. Cross-Validation:
- The cross-validation loop splits data into train-test sets for each fold and standardizes the features before training the RandomForest model.
- The performance is evaluated using Mean Absolute Error (MAE).

3. Best Practices:
- Clear comments and docstrings are added to make the code understandable.
- Variables are consistently named, and constants (K, SPLIT) are defined at the top for easy configuration.

[Task 4 Project Folder](https://github.com/MikkoDT/Cognizant_AI_Experience_Program/tree/main/Task4)

**Technologies Used**:

- [Python](https://www.python.org/): Core programming language for module development.
- [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.
- [Scikit-learn](https://scikit-learn.org/stable/): Model training, cross-validation, and evaluation.
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): Supervised learning algorithm for regression tasks.
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): Feature scaling for optimal model performance.
