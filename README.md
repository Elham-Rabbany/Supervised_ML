# Supervised_ML
Supervised ML - Predicting Housing Prices


In this project, we will create a model that predicts the price of a house based on its characteristics.
 overview of steps:
 ![Steps Project](images/Steps.jpeg)


 
 ## Impute missing values: 
 ### Fit on train, transform train & test:
<br>from sklearn.impute import SimpleImputer
<br>**initialise:** my_imputer = SimpleImputer().set_output(transform='pandas')
<br>**fit on the train set:** my_imputer.fit(X_num_train)
<br>**transform the train set:** X_num_imputed_train = my_imputer.transform(X_num_train)
<br>**transform the test set:** X_num_imputed_test = my_imputer.transform(X_num_test)


