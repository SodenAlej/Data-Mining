import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv')

df.dropna(subset=['latitude', 'longitude', 'minimum_nights', 'room_type'], inplace=True)

X = df[['latitude', 'longitude', 'minimum_nights']]
y = df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)

# Separar entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DATA CLASSIFICATION
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print('Precisión (k-NN):', accuracy)

# DATA CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Gráfico de clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['latitude'], df['longitude'], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.title('Clústeres de propiedades según ubicación')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.tight_layout()
plt.show()
