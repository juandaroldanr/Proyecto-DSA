
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns

#Como primer paso se importa la base de datos
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
df

print("Las dimensiones del dataframe con la base de datos de los datos sobre la obesidad y hábitos de vida:", df.shape, " es decir, hay ",df.shape[0], "registros y ",df.shape[1],"atributos o características. \n")
print("El tamaño de la base de datos, es decir la cantidad total de datos es:", df.size, "\n")
print("Las filas están organizádas a través de un índice en el siguiente rango: ", df.index, "\n")
print("Los nombres de las columnas dentro de las base de datos son:", df.columns, "\n")
print("\nLa información básica de la base de datos es: \n")
print(df.info())

#Se analizan los valores unicos de las columnas categoricas con el fin de explorar los diversos valores que pueden tomar
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

for i in categorical_columns:
  print(i)
  print(df[i].unique())
  print("=" * 30)  # Separador visual entre columnas

#Se extraen las columnas numéricas se extraen estadísticas descriptivas básicas
num_columns = [col for col in df.columns if df[col].dtype == 'float64']
df[num_columns].describe().T

