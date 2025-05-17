import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv')

# 'last_review' como fecha relacionada con actividad del host
df.dropna(subset=['last_review', 'price'], inplace=True)
df['last_review'] = pd.to_datetime(df['last_review'])
df = df.sort_values('last_review')
df.set_index('last_review', inplace=True)

# Calcular días desde la primera fecha
df['days_since'] = (df.index - df.index.min()).days

# Modelo de regresión lineal
X = df['days_since'].values.reshape(-1, 1)
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Predicción
df['price_pred'] = model.predict(X)

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['price'], label='Precio real', alpha=0.5)
plt.plot(df.index, df['price_pred'], label='Predicción', linestyle='--', color='red')
plt.title('Predicción de precios según fecha de última reseña')
plt.xlabel('Fecha de última reseña')
plt.ylabel('Precio')
plt.legend()
plt.tight_layout()
plt.show()
