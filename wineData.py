# Dependencies
# Step 0 - Set up your environment
import numpy as np # Efficient numerical computation
import pandas as pd # Working with data frames
from sklearn.model_selection import train_test_split # Helps choose between models
from sklearn import preprocessing # Preprocessing
from sklearn.ensemble import RandomForestRegressor # Import random forest model family
from sklearn.pipeline import make_pipeline # Cross-validation tool
from sklearn.model_selection import GridSearchCV # Cross-validation tool
from sklearn.metrics import mean_squared_error, r2_score # Metrics to evaluate model performance
from sklearn.externals import joblib # Persist model - alternative to Pickle
import matplotlib.pyplot as plt # I like pretty graphs

# STEP 1 - Import wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';') # Separate data so it doesn't suck

# Let's take a look at this data, and see if it'll cause any problems when we train our model (if it's skewed or something)
histData = data.quality # Get target metric data
n, bins, patches = plt.hist(histData, 'auto', align='left')
plt.title('Distribution of Red Wine Quality')
plt.xticks(np.arange(1, 10, 1))
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

# Step 2 - Splitting into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.2,
    random_state=123,
    stratify=y
)

# Standardize & preprocess data
# Standardization is commonly required in ML as many algorithms assume data is centered at zero and have approximately the same variance. 
# Need to invoke the Transformer API to fit a standardization using the training data. 
# Allows you to insert preprocessing steps into a cross-validation pipeline. 
# This line accomplishes both step 3 - data prep - and step 4 - choosing a model. 
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100)) # Alternate cross-validation

# Step 5: Training/Set hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model - CV
clf.fit(X_train, y_train)

# Test model
y_pred = clf.predict(X_test)

# Step 6: Evaluate model performance
print(r2_score(y_test, y_pred))

print(mean_squared_error(y_test, y_pred))

# Plot data and see how good the model is
plt.plot(y_test, y_pred, 'ro')
plt.xlabel('Real Data')
plt.ylabel('Predicted Data')
plt.xticks(np.arange(1, 10, 1))
plt.yticks(np.arange(1, 10, 1))
plt.show()

# Save model
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')




