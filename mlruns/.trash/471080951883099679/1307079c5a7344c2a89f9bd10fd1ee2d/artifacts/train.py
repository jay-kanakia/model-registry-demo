from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')


# Splitting data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [75,125,200],
    'max_depth': [2,5,7]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment("diabetes-rf_new")
with mlflow.start_run(run_name="hyperparameter_tuning",description="Best hyperparameter trained RF Model") as parent:
    grid_search.fit(X_train,y_train)
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i],'')
            mlflow.log_metric('accuracy',grid_search.cv_results_['mean_test_score'][i])
            mlflow.log_artifact(__file__)

            #dataFrame
            train_df=pd.DataFrame(X_train)
            train_df['outcome']=y_train
            test_df=pd.DataFrame(X_test)
            test_df['outcome']=y_test

            train_df=mlflow.data.from_pandas(train_df)
            test_df=mlflow.data.from_pandas(test_df)
            mlflow.log_input(train_df,'training')
            mlflow.log_input(test_df,'testing')

            # signature=mlflow.models.infer_signature(X_train,grid_search.cv_results_['params'][i].predict(X_train))
            # mlflow.sklearn.log_model(grid_search.cv_results_['params'][i],"random_forest",signature=signature)

            mlflow.set_tag('authore','jay')
    
    best_param=grid_search.best_params_
    best_score=grid_search.best_score_

    mlflow.log_params(best_param)
    mlflow.log_metric('accuracy',best_score)
    mlflow.log_artifact(__file__)

    train_df=pd.DataFrame(X_train)
    train_df['outcome']=y_train
    test_df=pd.DataFrame(X_test)
    test_df['outcome']=y_test
    train_df=mlflow.data.from_pandas(train_df)
    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(train_df, context="training")
    mlflow.log_input(test_df, context="testing")


    signature=mlflow.models.infer_signature(X_train,grid_search.best_estimator_.predict(X_train))
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random_forest",signature=signature)

    mlflow.set_tag('authore','jay')