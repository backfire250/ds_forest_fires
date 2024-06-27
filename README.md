# Optimizing Models For Predicting Forest Fires
Repo for my data science study for predicting forest fires


**Project Goal**:  Predict the extent of fire damage to a forest using data from the UCI Machine Learning Repository.
* Aim to determine how specific characteristics in our dataset can be used to accurately predict future forest fire damage
* Use multiple machine learning techniques to predict outcomes based on provided data
* Build regularized and non-linear models in addition to linear models, then evaluate using k-fold cross validation

## Project Steps
1. Data Collection
2. Data Cleaning
3. Outlier Detection
4. Data Standardization
5. Exploratory Data Analysis
6. Feature Selection
7. Model Building
8. Model Evaluation

## Code and Resources Used  
**Python Version:** 3.7   
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn 
**Jupyter Notebooks**  

## Data Collection
I collected the data from the UCI Machine Learning Repository. The dataset we're using contains information on fires, the damage caused, and associated meteorological data.

## Data Cleaning
After taking an initial look at the data, I needed to clean it up so that it was usable for our model. I made the following changes after some initial evaluation:

*	Converted string values to numerical values for use in our models 
*	Dropped columns that did not seem relevant to predicting the amount of area damaged by forest fires
*	Found that our target column was skewed towards 0, so we applied a log transform to that particular column
*	There were 8 columns in our dataset that were missing data, so we used KNN imputation to fill them in

## Outlier Detection
Using boxplots, I was able to identify the number of outliers in each of our feature columns. The 'FFMC' column had the most amount of outliers with 53, but this is relatively small considering we have 517 observations in this data set. After looking at boxplots of the outliers for each feature column, the outliers that did appear in our feature columns seemed relatively minor and I was satisfied with keeping them included in our dataset.

## Data Standardization
I wanted to standardize our data so that the feature columns would have a mean of 0 and a standard deviation of 1. I took our imputed data from one of our previous steps and applied the StandardScaler from sklearn.preprocessing. After applying the scaler, I created a new DataFrame from the standardized, scaled data and concatenated this with our columns that weren't missing any values.

## EDA (Exploratory Data Analysis)
I looked at the correlation between my continuous variables and my target variable. Below are a couple highlights from my EDA, as well as a map of the residuals after my model evaluation was complete.

![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/fires%20eda.png)
![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/fires%20correlation.png))
![](https://github.com/backfire250/Ernie_Portfolio/blob/main/images/fires%20residuals.png)

## Feature Selection
After cleaning our dataset and doing some initial data analysis, we could move on to selecting features for our model. I began by trying sequential feature selection, both forward and backward, to see what results it could come up with. I was expecting that this should give us an efficient way of evaluating the performance of different feature combinations and let us choose the one with the best results. I looked at both forward and backward feature selection and experimented with 2, 4, 6 and 8 features for each to see which features would be selected.

## Model Building
I split the data into train and test sets with a test size of 20%.

I tried three different models and evaluated them using Mean Squared Error (the average variance between actual values and projected values in the dataset).

I tried four different models:
*    **Forward/Backward Linear Regression** - Used as the baseline for my model.
*    **Lasso Regression** - I wanted to account for any non-linear components in the dataset
*    **Ridge Regression** - For the same reason I used Lasso regression, I thought that this would be a good fit.
*    **Spline** - I tried this model mostly just out of curiosity.

## Model Evaluation
The forward/backward linear regression model far outperformed the other approaches on the test and validation sets.
*    **Forward/Backward Linear Regression** : MSE = 1.86
*    **Lasso Regression** : MSE = 2.29
*    **Ridge Regression** : MSE = 2.28
*    **Spline**: MSE = 7.78

## Model Evaluation
In this step, I built a Flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the resource section above. The API endpoint takes in a request with a list of values from a job posting and returns an estimated salary.

## Conclusion
With this project, we demonstrated the use of many different techniques in our machine learning library and used these techniques to build models that could accurately predict the outcomes based on our provided data. We used imputation to fill in missing data from our data set, looked at outliers and examined the correlation between our feature columns and our target column. We also built several regularized and non-linear models and used k-fold cross validation to evaluate our models.
