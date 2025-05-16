import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv') 

# Eliminar filas con datos faltantes
df.dropna(subset=['latitude', 'longitude', 'minimum_nights', 'price'], inplace=True)

# Variables independientes y dependiente
X = df[['latitude', 'longitude', 'minimum_nights']]
y = df['price']

# Entrenar modelo lineal
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calcular R²
r2 = r2_score(y, y_pred)
print('R² score:', r2)

# Gráfico: precio real vs. predicho
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Precio real vs. Precio predicho')
plt.tight_layout()
plt.show()
