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


