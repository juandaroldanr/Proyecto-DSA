
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

# Histogramas
df[num_columns].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histogramas de columnas de tipo numérico")
plt.show()

#Boxplot 
for i in num_columns:
  df.boxplot(column=[i], by=['NObeyesdad'])
  fig = plt.gcf()
  fig.set_size_inches(15, 5)
  plt.show()
  
# Gráfico de dispersión por tipo de obesidad
plt.figure(figsize=(15, 5))
for i in num_columns:
    for obesity_type in df['NObeyesdad'].unique():
        subset = df[df['NObeyesdad'] == obesity_type]
        plt.scatter(subset.index, subset[i], label=obesity_type)
    plt.xlabel("Índice de datos")
    plt.ylabel(i)
    plt.legend(title='Tipo de Obesidad')
    plt.title(f"Gráfico de Dispersión de {i} por Tipo de Obesidad")
    plt.show()

# Gráfico de violín por tipo de obesidad para todas las columnas numéricas
plt.figure(figsize=(12, 8))
for column in num_columns:
    sns.violinplot(x='NObeyesdad', y=df[column], data=df)
    plt.title(f"Gráfico de Violín para {column} por Tipo de Obesidad")
    plt.xticks(rotation=45)
    plt.show()

# Matriz de correlaciones
correlation_matrix = df[num_columns].corr()
correlation_matrix

# Correlograma (scatter plot matrix)
sns.pairplot(df[num_columns])
plt.show()

# Mapa de calor de la matriz de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlaciones")
plt.show()

grouped = df.groupby('NObeyesdad').describe().select_dtypes(include='number')
grouped


