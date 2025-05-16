import pandas as pd
import numpy as np

# 1) Data Cleaning
# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv')

# Mostrar resumen de los datos
print(df.info())

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Convertir fechas (last_review) a formato datetime
df['last_review'] = pd.to_datetime(df['last_review'], format='%d/%m/%Y', errors='coerce')

# Ver los primeros 5 registros después de la limpieza
print(df.head())

# 2) Descriptive Statistics
# Estadísticas descriptivas
print(df.describe())

# Agrupar por 'neighbourhood' y calcular la media de columnas numéricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
grouped = df.groupby('neighbourhood')[numeric_columns].mean()

# Mostrar las primeras filas del agrupamiento
print(grouped.head())
