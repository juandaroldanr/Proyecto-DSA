#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10

@authors: Diego Garcia, Jairo Rueda, Laura Peralta, Juan Roldán 
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#Se cargan los datos.

#df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
df = pd.read_csv('C:/Users/Juan/Downloads/ObesityDataSet_raw_and_data_sinthetic.csv')

#Variables a conservar : SMOKE si fuma o no, CALC cantidad de alchool que consume, NCP comidas al día, CH2O litros agua al día,FAF qué tan seguido se ejercita, TUE tiempo usando dispositivos electrónicos, MTRANS medio de transporte, FCVC frecuencia consumo vegetales 
df = df.drop(columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'CAEC', 'SCC', 'FAF'])

#preprocesamiento

#Se separa la acoluma a predecir en el modelo de machine learning y también se vuelve binaria la variable de respuesta
y = df["NObeyesdad"]
df = df.drop(columns=['NObeyesdad'])
#y = np.where((y == 'Normal_Weight') | (y == 'Insufficient_Weight'), 0, 1)


#Conversión de variables categóricas binarias a unos y ceros

df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})

catcols = df.select_dtypes(exclude = ['int64','float64']).columns
intcols = df.select_dtypes(include = ['int64']).columns
floatcols = df.select_dtypes(include = ['float64']).columns

# one-hot encoding para variables categóricas
df = pd.get_dummies(df, columns = catcols)

# minmax scaling para variabls numéricas
for col in df[floatcols]:
    df[col] = MinMaxScaler().fit_transform(df[[col]])

for col in df[intcols]:
    df[col] = MinMaxScaler().fit_transform(df[[col]])



# Se dividen los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("Randomclassifier_Obesity")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 100 
    max_depth = 6
    max_features = 4
    # Cree el modelo con los parámetros definidos y entrénelo
    
    # Modelo drandom forest
    modelo_random_forest = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    modelo_random_forest.fit(X_train, y_train) 
    #rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    #rf.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = modelo_random_forest.predict_proba(X_test)
  
    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)
  
    # Registre el modelo
    mlflow.sklearn.log_model(modelo_random_forest, "random-forest-model")
  
    # Cree y registre la métrica de interés
    exactitud = accuracy_score(y_test, modelo_random_forest.predict(X_test))
    mlflow.log_metric("Exactitud", exactitud)
    print(f'Exactitud: {exactitud:.2f}')
    #report = classification_report(y_test, modelo_random_forest.predict(X_test))
    #print(report)
    #mlflow.log_metric("Reporte", report)




   #mse = mean_squared_error(y_test, predictions)
   # mlflow.log_metric("mse", mse)
   # print(mse)