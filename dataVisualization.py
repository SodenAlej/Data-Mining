import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv')  

# Convertir la fecha de la última reseña
df['last_review'] = pd.to_datetime(df['last_review'], format='%d/%m/%Y', errors='coerce')

# Eliminar registros con precio o barrio nulo
df.dropna(subset=['neighbourhood', 'price'], inplace=True)

# Lista de gráficos a generar
charts = ['histogram', 'boxplot', 'scatter', 'pie', 'line']

# Generar los gráficos
for chart in charts:
    plt.figure(figsize=(10, 6))
    
    if chart == 'histogram':
        sns.histplot(df['price'], bins=30)
        plt.title('Distribución de precios')
        
    elif chart == 'boxplot':
        sns.boxplot(x='room_type', y='price', data=df)
        plt.title('Precio por tipo de habitación')
        
    elif chart == 'scatter':
        sns.scatterplot(x='longitude', y='latitude', hue='price', data=df)
        plt.title('Ubicación de propiedades y precios')
        
    elif chart == 'pie':
        df['room_type'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Distribución de tipos de habitación')
        plt.ylabel('')
        
    elif chart == 'line':
        # Agrupar por fecha de última reseña para obtener precio promedio
        df.groupby('last_review')['price'].mean().plot()
        plt.title('Precio medio a lo largo del tiempo (última reseña)')
        plt.xlabel('Fecha')
        plt.ylabel('Precio promedio')
        
    plt.tight_layout()
    plt.show()
