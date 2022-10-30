# Machine Learning Regression and Time Series
# Machine Learning Workflow
- Data Profiling
- Data Cleansing
- Data Exploration
- Feature Engineering
- Modelling
- Evaluation and Deployment
### Requirements of Machine Learning
There are 3 Requirements of Machine Learning: 
- There must be data 
- The data must have a pattern 
- The algorithm is hard to be processed
# Data Profiling
Data profiling is the process of reviewing source data, understanding structure, content and interrelationships, and identifying potential for data projects. 
# Data Cleansing
Data cleansing is the process of detecting and correcting (or removing) corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.
# Exploratory Data Analysis
Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
### Multicollinearity
Multicollinearity occurs when one independent variable in a regression model is linearly correlated with another independent variable.
The way to detect multicollinearity in the regression model is by looking at the strength of the correlation between the independent variables. If there is a correlation between independent variables > 0.5, it can be indicated that there is multicollinearity.
# Feature Engineering
Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used for creating a predictive model using Machine learning or statistical Modelling.
### One-hot Encoding
-	One-hot encoding is a process of converting categorical data variables so they can be provided to machine learning algorithms to improve predictions.
-	The Python library Pandas provides a function called __get_dummies__ to enable one-hot encoding
### map() function
-	To convert numerical data variables can use the map() function
### Scaling
Feature scaling is about transforming the values of different numerical features to fall within a similar range like each other. The feature scaling is used to prevent the supervised learning models from getting biased toward a specific range of values.
-	StandardScaler for Standardization
    - StandardScaler is a class from sklearn.preprocessing which is used for standardization.
    - Standardization is used to center the feature columns at mean 0 with a standard deviation of 1 so that the feature columns have the same parameters as a standard normal distribution.

-	MinMaxScaler for Normalization
    - MinMaxScaler is a class from sklearn.preprocessing which is used for normalization.
    - Normalization refers to the rescaling of the features to a range of [0, 1], which is a special case of min-max scaling.
    
# Preprocessing Modeling
Preprocessing is the most important aspect of data processing. When data is acquired as an output of an experiment, the next step is modeling the data to extract useful information.
###	Feature Selection
Feature selection is the process of selecting the features that contribute the most to the prediction variable or output that you are interested in, either automatically or manually.
###	Feature Importance
Feature importance assigns a score to each of your data’s features; the higher the score, the more important or relevant the feature is to your output variable.
- Extra Trees Classifier
     - Extra Trees Classifier is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result.
###	Train Test Split
- The train-test split is a technique for evaluating the performance of a machine learning algorithm.
# Modeling
In this topic, the modeling that will be explained is Simple Linear Regression, Multiple Linear Regression, and Time Series.

**Step 1**: Fitting into training

**Step 2**: Predic the result

**Step 3**: Plot the result

# Machine Learning Regression
# Regression
The simplest type of relationship these vatiables is a linear relationship
### Situation:
-	There is a single response variable Y, also called the dependent variable
-	Depends on the vakue of a set of input, aslo called independent varaibles (X1, X2, ..., Xr)
### Linear Regression Assumption
-	Relation between dependant variable and independent variable is linear.
-	Error or residual of the model need to be normally distributed.
-	There is no multicolineraity.

# Simple Linear Regression
Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.
-	One Variable X (independent variable)
-	One Variable y (dependent variable)

# Multiple Linear Regression 
Multiple linear regression is a statistical method to estimate the relationship between two or more independent variables and one dependent variable.
- Two or more variable X (indepedendent variable)
- One variable y (dependet variable)

# Time Series
- A Time Series is typically defined as a series of values that one or more variables take over successive time periods. For example, sales volume over a period of successive years, average temperature in a city over months etc.
- Therefore, Time Series forecast is about forecasting a variable’s value in future, based on it’s own past values.
- To modelling time series data, we used prophet module. Prophet models time series as a generalized additive model (GAM) combining the trend function, seasonality function, holiday effects, and an error term in one model
- In machine learning, train/test split splits the data randomly. But, in time series data the values at the rear of the dataset if for testing and everything else for training.
- The better the model is the model with the more similar the data from the model obtained to the original data. One of the indicators is to look at the MAPE value (the smaller one is the better one).

# Cross Validation
Cross-validation (CV) is a technique used to assess a machine learning model and test its performance (or accuracy). It involves reserving a specific sample of a dataset on which the model isn't trained. Later on, the model is tested on this sample to evaluate it. Cross-validation is used to protect a model from overfitting, especially if the amount of data available is limited. It's also known as rotation estimation or out-of-sample testing and is mainly used in settings where the model's target is prediction.

# Hyperparameter Tuning
There is a list of different machine learning models. They all are different in some way or the other, but what makes them different is nothing but input parameters for the model. These input parameters are named as Hyperparameters. These hyperparameters will define the architecture of the model, and the best part about these is that you get a choice to select these for your model. Of course, you must select from a specific list of hyperparameters for a given model as it varies from model to model. 

Often, we are not aware of optimal values for hyperparameters which would generate the best model output. So, what we tell the model is to explore and select the optimal model architecture automatically. This selection procedure for hyperparameter is known as Hyperparameter Tuning.

# Evaluate Model
-	One way to evaluate models is to use MAPE
-	The Mean Absolute Percentage Error (MAPE) can be used in machine learning to measure the accuracy of a model.
-	Interpretasion of MAPE
    -	*> 50%		: The accuracy of model is **Poor**
    -	20% - 50%	: The accuracy of model is **Relatively good**
    -	10% – 20%	: The accuracy of model is **Good**
    -	< 10%		  : The accuracy of model is **Great**
