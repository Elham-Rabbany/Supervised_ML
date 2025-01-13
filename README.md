# Supervised_ML
Supervised ML - Predicting Housing Prices


In this project, we will create a model that predicts the price of a house based on its characteristics.
 overview of steps:
 ![Steps Project](images/Steps.jpeg)


 
 ## Impute missing values: 
 ### Fit on train, transform train & test:
```
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer().set_output(transform='pandas')      # initialise
my_imputer.fit(X_num_train)                                      # fit on the train set
X_num_imputed_train = my_imputer.transform(X_num_train)          # transform the train set
X_num_imputed_test = my_imputer.transform(X_num_test)            # transform the test set
```


## Pipeline creation
Scikit-Learn Pipelines: They streamline data preparation and modeling into one step, but they will not increase the performance of  model.
```
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(imputer, dtree).set_output(transform='pandas')
```

## Use GridSearchCV to find the best parameters of the model
Check the docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

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

### Explore the best parameters and the best score achieved with your cross validation:

```
search.best_params_
search.best_score_
```

## Categorical encoding - "Automated" approach (Using Pipelines)

In the manual approach, to encode the categorical columns numerically, we have:

1. Selected the categorical columns.
2. Fitted a `OneHotEncoder` to them.
3. Transformed the categorical columns with the encoder.
4. Converted the sparse matrix into a dataframe.
5. Recovered the names of the columns.
6. Concatenated the one-hot columns with the numerical columns.

All these steps can be synthetised by using Scikit-Learn Pipelines and specifically something called `ColumnTransformer`, which allows us to apply different transformations to two or more groups of columns: in our case, categorical and numerical columns.

This process is also called creating "branches" in the pipeline. One branch for the categorical columns and another for the numerical columns. Each branch will contain as many transformers as we want. Then, the branches will meet again, and the transformed columns will be automatically concatenated. 
