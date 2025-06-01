# crop yields prediction using machine learning:

## the problem:
Agricultural productivity is critical to ensuring food security, economic stability, and sustainable resource use, particularly in regions heavily dependent on farming. However, crop yields are influenced by a complex interplay of factors, including weather conditions, soil quality, crop management practices, and climate variability. Traditional methods of yield estimation are often reactive, labor-intensive, and unable to effectively capture nonlinear relationships within agricultural data.

This project aims to develop a machine learning-based system that can accurately predict crop yields based on historical data such as weather patterns, soil properties, and farming practices. The objective is to enable proactive decision-making for farmers, policymakers, and agribusinesses, allowing them to optimize resource allocation, anticipate food production levels, and mitigate the risks associated with climate change and unpredictable environmental conditions.

By leveraging supervised learning algorithms and real-world datasets, the system will model the relationships between key agricultural inputs and yield outcomes, offering interpretable and actionable insights that support precision agriculture and sustainable farming.

## the objectives:
1. to preprocess the data to make it ready for mcahine learning model
2. to train the model on the existing data
3. to test the model on new unseen data

## requirments
1. pandas
2. scikit-learn
3. stramlit
4. pickle
5. numpy
6. seaborn
7. matplotlib
## data preprocessing
- Handled missing values (NaNs) by either removing or filling them.
- Separated the features (X) from the target variable (y).
- Encoded categorical variables into numbers using LabelEncoder.
- Scaled the data using MinMaxScaler to make sure all features were on the same scale, which helps the model learn better.

## data splitting
I used train_test_split to divide the dataset into three parts:
- Training set: used to train the model.
- Validation set: used to fine-tune the model and check for overfitting.
- Test set: used to evaluate final model performance on unseen data.
## model selection
For this project, I chose the Linear Regression model. It’s a simple and effective algorithm for predicting continuous values like crop yield. Since the project is not too complex, using a more advanced model would add unnecessary complications.

To evaluate how well the model performed, I used the following metrics:
- Mean Absolute Error (MAE): average size of the errors.
- Mean Squared Error (MSE): similar to MAE but gives more weight to big errors.
-R² Score: shows how well the model explains the variation in the data. A score close to 1 means excellent performance.

## streamlit application
The model and its features were wrapped into a Streamlit app for easy use. Users can:
- Upload a dataset.
- Choose which columns to encode.
- Train and test the model.
- View predictions


