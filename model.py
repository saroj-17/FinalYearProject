import numpy as np
import pandas as pd
import numpy as np 
import pandas as pd 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import seaborn as sns #seaborn is used for the staic data visualization 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


########################################################################################
#
#EXPLANTION OF THE CODE
#The code is used to create a simple decision tree regressor and a simple random forest regressor from scratch.
#The SimpleDecisionTreeRegressor class contains the following methods:

# __init__(self):
# Initializes the decision tree and feature importances attributes to None.

# mean_squared_error(y):
# Static method that calculates the mean squared error for a set of values y.
# Example: If y = [3, 4, 5, 6, 7], the mean squared error would be calculated as follows:
# Mean: (3 + 4 + 5 + 6 + 7) / 5 = 5
# Mean Squared Error: ((3-5)^2 + (4-5)^2 + (5-5)^2 + (6-5)^2 + (7-5)^2) / 5 = 2

# get_mse(self, y):
# Calculates the mean squared error for a given set of y values using the mean_squared_error method.

# best_split(self, X, y):
# Finds the best feature and value to split the data based on minimizing
# the weighted mean squared error.
# Example: If the dataset has features like 'airlines', 'duration_hrs',  'destination', 
# this method determines the best feature and value to split the dataset 
# into subsets based on these features.

# grow_tree(self, X, y, depth, min_samples_split):
# Recursively grows the decision tree.
# Stops growing when the depth is 0 or the number of samples is less than 
# the minimum required for a split.
# Example: It recursively splits the dataset into subsets based on the best 
# feature and value until the stopping conditions are met.

# fit(self, X, y, max_depth=None, min_samples_split=None):
# Fits the decision tree to the training data.
# Calls the grow_tree method to build the decision tree.
# Calculates and stores feature importances.
# Example: fit(X_train, y_train, max_depth=10, min_samples_split=10) 
# trains the decision tree with a maximum depth of 10 and a minimum number 
# of samples required to split a node of 10.

# calculate_feature_importances(self, X, y):
# Calculates feature importances based on mean squared error.
# Example: It calculates the importance of each feature in predicting
# the target variable by evaluating how much the mean squared error 
# decreases when a feature is chosen for splitting.

# predict_obs(self, x, tree):
# Predicts the output for a single observation using the trained tree.
# Example: Given an observation x, it traverses the decision tree and 
# predicts the output based on the learned split conditions.

# predict(self, X):
# Predicts the outputs for a set of observations using the trained tree.
# Example: predict(X_test) predicts the outputs for the test set X_test.

# k_fold_cross_validation_DT(self, X, y, k=5):
# Performs k-fold cross-validation to evaluate the decision tree regressor.
# Splits the data into k folds, trains the model on k-1 folds, and evaluates 
# on the remaining fold.
# Example: k_fold_cross_validation_DT(X, y, k=10) evaluates the decision tree
# regressor using 10-fold cross-validation on the dataset X with target variable y.






# Definition of the SimpleDecisionTreeRegressor class
class SimpleDecisionTreeRegressor:
    # Initialization method to set initial values
    def __init__(self):
        # Initializing the tree attribute to None
        self.tree = None
        # Initializing the feature_importances_ attribute to None
        self.feature_importances_ = None

    # Static method to calculate the mean squared error of a set of values y
    @staticmethod
    def mean_squared_error(y):
        # If the length of y is 0, return 0.0 to avoid division by zero
        if len(y) == 0:
            return 0.0
        # Calculate the mean of y
        mean = np.mean(y)
        # Calculate the mean squared error
        return np.mean((y - mean) ** 2)

    # Method to calculate the mean squared error for a given set of y values
    def get_mse(self, y):
        return self.mean_squared_error(y)

    # Method to find the best feature and value to split the data based on minimizing weighted mean squared error
    def best_split(self, X, y):
        # Get the list of feature names
        features = list(X.columns)
        # Initialize variables to store the best feature, value, and mean squared error
        best_feature, best_value, best_mse = None, None, float('inf')

        # Iterate over features
        for feature in features:
            # Get unique values for the current feature and sort them
            values = sorted(X[feature].unique())
            # Iterate over values
            for value in values:
                # Create boolean masks for left and right subsets of the data
                left_mask = X[feature] <= value
                right_mask = ~left_mask

                # Get y values for left and right subsets
                y_left = y[left_mask]
                y_right = y[right_mask]

                # Calculate mean squared errors for left and right subsets
                mse_left = self.get_mse(y_left)
                mse_right = self.get_mse(y_right)

                # Calculate the number of samples in left and right subsets
                n_left = len(y_left)
                n_right = len(y_right)

                # Calculate weights for left and right subsets
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculate the weighted mean squared error
                w_mse = w_left * mse_left + w_right * mse_right

                # Update best feature, value, and mean squared error if current split is better
                if w_mse < best_mse:
                    best_feature, best_value, best_mse = feature, value, w_mse

        # Return the best feature, value, and mean squared error
        return best_feature, best_value

    # Method to recursively grow the decision tree
    def grow_tree(self, X, y, depth, min_samples_split):
        # Check stopping conditions: if depth is 0 or the number of samples is less than the minimum required for a split
        if depth == 0 or len(y) < min_samples_split:
            # Return the mean of y if it is not empty, otherwise return None
            return np.mean(y) if not y.empty else None

        # Find the best feature and value to split the data
        best_feature, best_value = self.best_split(X, y)

        # If no valid split is found, return the mean of y if it is not empty, otherwise return None
        if best_feature is None:
            return np.mean(y) if not y.empty else None

        # Create boolean masks for left and right subsets of the data
        left_mask = X[best_feature] <= best_value
        right_mask = ~left_mask

        # Recursively grow the left and right subtrees
        left_subtree = self.grow_tree(X[left_mask], y[left_mask], depth - 1, min_samples_split)
        right_subtree = self.grow_tree(X[right_mask], y[right_mask], depth - 1, min_samples_split)

        # Return a tuple representing the current node in the tree
        return (best_feature, best_value, left_subtree, right_subtree)

    # Method to fit the decision tree to the training data
    def fit(self, X, y, max_depth=None, min_samples_split=None):
        # Grow the tree using the training data
        self.tree = self.grow_tree(X, y, max_depth, min_samples_split)
        # Calculate and store feature importances
        self.feature_importances_ = self.calculate_feature_importances(X, y)

    # Method to calculate feature importances based on mean squared error
    def calculate_feature_importances(self, X, y):
        # Initialize a list to store feature importances
        mse_values = []
        # Iterate over features
        for feature in X.columns:
            # Get unique values for the current feature
            unique_values = X[feature].unique()
            # print(feature) all are column in X 
            #print(X[feature]) #row value pair 101:2
            # print(unique values) array of each unique value in X e.g. stop [0,1,2,3,4]
            
            # Initialize a variable to store the sum of mean squared errors for the feature
            mse_sum = 0
            
            # Iterate over unique values
            for value in unique_values:
                # Create a boolean mask for the current value
                value_mask = X[feature] == value
                
                # print(value_mask) 101 : True, value=2 , X[feature]=101:2
              
                # Get y values for the current value
                y_values = y[value_mask]  
                # print(y_values) 101 price value to y_values
                
               
                # Calculate mean squared error for the current value and weight it by the number of samples
                mse = self.get_mse(y_values) * len(y_values) / len(y)
                # Add the weighted mean squared error to the sum
                mse_sum += mse
            # Append the feature and its total weighted mean squared error to the list
            mse_values.append((feature, mse_sum))
        # Sort feature importances in ascending order
        mse_values.sort(key=lambda x: x[1])
        # print(mse_values) ('Total_Stops', 10919834.762078494)
        # Return the sorted feature importances
        return mse_values

    # Method to predict the output for a single observation using the trained tree
    # Method to predict the output for a single observation using the trained tree
    def predict_obs(self, x, tree):
            # Check if the current node is an internal node
        if isinstance(tree, tuple):
                # Unpack the tuple representing the internal node
            feature, value, left_subtree, right_subtree = tree
                # Make a recursive call based on the split condition
            if x[feature] <= value:
                 return float(self.predict_obs(x, left_subtree))  # Convert to float
            else:
                return float(self.predict_obs(x, right_subtree))  # Convert to float
        else:  # Leaf node
            # Return the value associated with the leaf node, converted to float
            return float(tree)

    # Method to predict the outputs for a set of observations using the trained tree
    def predict(self, X):
        # Check if the tree has been trained
        if self.tree is None:
            raise ValueError("The model has not been trained yet. Call fit() first.")
        # Initialize a list to store predictions
        predictions = []
        # Iterate over rows of the input DataFrame
        for _, x in X.iterrows():
            #_ is throw away character as we donot need the index 
            # Make a prediction for each observation and append it to the list
            prediction = self.predict_obs(x, self.tree)
            predictions.append(prediction)
        # Return the list of predictions
        return predictions
    
    def k_fold_cross_validation_DT(self, X, y, k=5):
        mg =1 
        count =mg
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        r2_scores, mse_scores, mae_scores = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create an instance of SimpleRandomForestRegressor and fit the model
            simple_dt_regressor = SimpleDecisionTreeRegressor()
            simple_dt_regressor.fit(X_train, y_train, max_depth=10, min_samples_split=10)

            # Make predictions on the test data
            predictions_dt = simple_dt_regressor.predict(X_test)

            # Calculate metrics
            r2_scores.append(r2_score(y_test, predictions_dt))
            mse_scores.append(mean_squared_error(y_test, predictions_dt))
            mae_scores.append(mean_absolute_error(y_test, predictions_dt))
            
            print(f"Finished the {count} Iteration")
            count +=1
        avg_r2 = np.mean(r2_scores)
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)

        print(f'Average R-squared: {avg_r2:.4f}')
        print(f'Average MSE: {avg_mse:.4f}')
        print(f'Average MAE: {avg_mae:.4f}')
#################################################################################

# __init__(self, n_estimators=100, max_depth=None, min_samples_split=2): 
# The constructor method initializes the random forest regressor object with the
# number of estimators (n_estimators), maximum depth of the trees (max_depth), and
# the minimum number of samples required to split an internal node (min_samples_split).

# mean_squared_error(self, y): 
# Calculates the mean squared error for a given set of target values y.

# get_mse(self, y): 
# Wrapper function for mean_squared_error, returning the mean squared error for a 
# given set of target values y.

# bootstrap_sample(self, X, y): Generates a bootstrap sample from the provided
# dataset X and target values y. It randomly selects samples from X and y with replacement.

# train_tree(self, X, y): Trains an individual decision tree using the bootstrap
# sample generated by bootstrap_sample(). It returns the trained decision tree.

# fit(self, X, y): Fits the random forest regressor to the training data X and target
# values y. It trains n_estimators decision trees using bootstrapped samples and stores
# them in the trees attribute.

# predict(self, X): Makes predictions for the input dataset X. It computes predictions
# for each decision tree in the forest and returns the average prediction across all trees.

# k_fold_cross_validation(self, X, y, k=5): Performs k-fold cross-validation to evaluate
# the performance of the random forest regressor. It splits the dataset into k folds,
# trains the model on k-1 folds, and evaluates it on the remaining fold. It returns
# the average R-squared, mean squared error, and mean absolute error scores across
# all folds.


class SimpleRandomForestRegressor:
    
    #At first the initialization is done,n_estimators is the paramater for number of trees
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  #To store the individual decision tree

    def mean_squared_error(self, y):
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def get_mse(self, y):
        return self.mean_squared_error(y)

    def bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), len(X), replace=True) #create a bootstrap sample for random subset of datas
        X_sample, y_sample = X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
        return X_sample, y_sample #new sample 

    def train_tree(self, X, y):
        tree = SimpleDecisionTreeRegressor()
        tree.fit(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        return tree #multiple decision tree 

    def fit(self, X, y):
        print("Starting Random Forest fitting...")
        for _ in range(self.n_estimators):
            print(f"Fitting tree {_ + 1}...")
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = self.train_tree(X_sample, y_sample)
            print(f"Tree {_ + 1} fitted.")
            self.trees.append(tree)
        print("Random Forest fitting complete.")



    def predict(self, X):
        predictions = np.zeros(len(X))

        for tree in self.trees:
            # Reset the index of the input DataFrame to avoid KeyError
            X_reset_index = X.reset_index(drop=True)

            tree_predictions = np.array([tree.predict_obs(x, tree.tree) for _, x in X_reset_index.iterrows()])
            predictions += tree_predictions

        return predictions / self.n_estimators
    def k_fold_cross_validation(self, X, y, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        r2_scores, mse_scores, mae_scores = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create an instance of SimpleRandomForestRegressor and fit the model
            simple_rf_regressor = SimpleRandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=10)
            simple_rf_regressor.fit(X_train, y_train)

            # Make predictions on the test data
            predictions_rf = simple_rf_regressor.predict(X_test)

            # Calculate metrics
            r2_scores.append(r2_score(y_test, predictions_rf))
            mse_scores.append(mean_squared_error(y_test, predictions_rf))
            mae_scores.append(mean_absolute_error(y_test, predictions_rf))
        
        avg_r2 = np.mean(r2_scores)
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)

        print(f'Average R-squared: {avg_r2:.4f}')
        print(f'Average MSE: {avg_mse:.4f}')
        print(f'Average MAE: {avg_mae:.4f}')


