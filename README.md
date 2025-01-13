# Supervised_ML_Classification
Supervised ML_Classification - Predicting Housing Prices


In this project, we will create a model that predicts the price of a house based on its characteristics.
 overview of steps:
 ![Steps Project](images/Steps.jpeg)


 
 
## Iteration 0: intuition-based model
Establishing a baseline model using simple, intuitive rules provides a reference point to evaluate the effectiveness of more complex machine learning models. By comparing the performance of these models against the baseline, one can assess their added value and ensure they offer significant improvements over straightforward approaches

## Iteration 1: train a decision tree
Training a Decision Tree
A decision tree is a model that predicts target variables by learning decision rules from data features.
### Scikit-Learn Pipelines
Scikit-Learn's Pipeline class streamlines machine learning workflows by chaining preprocessing and modeling steps, ensuring proper execution order and reproducibility.
### Cross-Validation
Cross-validation evaluates model performance by partitioning data into subsets for training and validation, helping assess generalization to unseen data and detect overfitting.
Integrating pipelines and cross-validation into decision tree training enhances model efficiency, reliability, and maintainability.


## Iteration 2: grid search
GridSearchCV automates hyperparameter tuning by exhaustively searching a predefined parameter grid and evaluating each combination using cross-validation to identify the optimal model configuration.
To define the parameter grid for cross validation, we need to create a dictionary, where:

- The keys are the name of the pipeline step, followed by two underscores and the name of the parameter we want to tune.
- The values are lists (or "ranges") with all the values we want to try for each parameter.

```
param_grid = {
    'decisiontreeclassifier__max_depth': range(2, 12),
    'decisiontreeclassifier__min_samples_leaf': range(3, 10, 2),
    'decisiontreeclassifier__min_samples_split': range(3, 40, 5),
    'decisiontreeclassifier__criterion':['gini', 'entropy']
    }
```
```
search = GridSearchCV(pipe, # you have defined this beforehand
                      param_grid, # your parameter grid
                      cv=5, # the value for K in K-fold Cross Validation
                      scoring='accuracy', # the performance metric to use,
                      verbose=1) # we want informative outputs during the training process, try changing it to 2 and see what happens
```
Check the docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


## Iteration 3: one-hot encoding
One-hot encoding transforms categorical variables into binary vectors, enabling machine learning algorithms to process non-numeric data effectively. Each category is represented by a vector with a single high (1) bit and all others low (0), ensuring no ordinal relationships are inferred between categories. While this method facilitates the inclusion of categorical data in models, it can increase dataset dimensionality, especially with features containing numerous unique categories. 

Cheack here the documentation for One Hot Encoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html


## Iteration 4: ordinal encoding
Ordinal encoding takes all the classes in a categorical feature and assigns a number to them, starting at 0.
This will require you to create another branch within the categorical branch, so a new ColumnTransformer will be needed. The pipeline you aim for should look like this: 
![Steps Project](images/Ordinal_cat.png)
Check here the documentation for the Ordinal encoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html


### Explore the best parameters and the best score achieved with your cross validation:

```
search.best_params_
search.best_score_
```


