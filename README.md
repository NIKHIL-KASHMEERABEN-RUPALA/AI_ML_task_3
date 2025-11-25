# AI_ML_task_3

California Housing Prices - Advanced Linear Regression Showcase
This project showcases the application of advanced linear regression techniques to predict housing prices in California. It walks through a comprehensive machine learning pipeline, including data preprocessing, model training, evaluation, and visualization, all aimed at predicting house prices based on various features like the number of rooms, population density, and house age.

What’s Covered in This Project

1. Data Loading and Exploration
The dataset used in this project comes from the fetch_california_housing function in Scikit-learn. It contains various features about housing in California. The target variable, which is the median house value, is scaled by multiplying by 100,000 to get actual dollar amounts.
The dataset includes features such as:
The average number of rooms
The population size
The age of the house
I also created some new features to improve the model:
Average rooms per household (ratio of rooms to house age)
Bedroom ratio (ratio of bedrooms to rooms)
Population density (ratio of population to average occupancy)



2. Feature Engineering
I added a few derived features to help the model capture relationships between the data better. These features were designed to improve the model’s predictive power by transforming existing data into more informative variables.

3. Model Setup
After splitting the data into training and testing sets (80% for training, 20% for testing), I set up a preprocessing pipeline to standardize the features using StandardScaler.
The models used include:
Linear Regression
Ridge Regression (L2 regularization)
Lasso Regression (L1 regularization)
ElasticNet (a combination of L1 and L2 regularization)
Each model was built in a pipeline that also includes polynomial feature expansion (degree 2) to capture non-linear relationships in the data.


4. Model Training and Evaluation
For each model, I evaluated its performance using several metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score (both test and cross-validation)
Cross-validation (CV) was used to check how well each model generalized to unseen data. This helped ensure the models weren’t overfitting or underfitting.


5. Model Selection
After training, I selected the best-performing model based on its R² score (how well it predicts housing prices) and its performance in cross-validation.



6. Model Diagnostics and Visualization
A series of visualizations were created to analyze the model’s performance and interpretability:
Actual vs Predicted: A scatter plot to compare actual vs predicted house prices.
Residuals vs Predicted: A plot to check for homoscedasticity (whether residuals have constant variance).
Residual Distribution: A histogram of residuals to check if they follow a normal distribution.
Top Feature Coefficients: A bar chart showing which features have the most influence on predictions.
Partial Dependence Plots: These plots show how different features affect house prices.
Learning Curve: A plot showing how the model’s performance improves as more data is used for training.


7. Key Insights
    
Here are some of the key takeaways:
Median Income is the most significant predictor of house prices in California.
Houses closer to the ocean tend to have a non-linear premium value, which the polynomial features help capture.
Older houses in denser areas might have a higher value due to location.
Regularization techniques (like Ridge and Lasso) help prevent overfitting, especially when polynomial features are used.
The residuals are mostly random, indicating the model does a good job capturing the data patterns.



8. Final Model Interpretation
The final selected model had the highest R² score, showing it explained a significant amount of the variance in house prices. The Mean Absolute Error (MAE) gives us an estimate of how much the model’s predictions are off on average, while the R² score indicates how well the model fits the data.
Libraries Used

Numpy: For numerical operations and calculations.

Pandas: For data manipulation and analysis.

Matplotlib/Seaborn: For creating visualizations.

Scikit-learn: For building the machine learning models and handling preprocessing tasks.

Warnings: To suppress unnecessary warnings during execution.

Conclusion

This project demonstrates several important techniques in machine learning:
Building a production-grade ML pipeline.
Advanced feature engineering for improved predictive accuracy.
Regularization techniques to avoid overfitting.
Comprehensive model evaluation and diagnostic plots.
Visualizing and interpreting model outputs to gain meaningful insights.
The result is a predictive model that can estimate house prices in California based on various features, along with a set of visualizations that provide clear, actionable insights.

How to Run It ---
Install the required libraries:
pip install numpy pandas scikit-learn matplotlib seaborn
Run the script in a Python environment to see the analysis in action.
