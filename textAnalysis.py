import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar el dataset 
df = pd.read_csv('csv/AB_NYC_2019.csv')

# Eliminar valores nulos en la columna de nombres
df.dropna(subset=['host_name'], inplace=True)

# Unir todos los nombres en un solo texto
text = ' '.join(df['host_name'].astype(str).tolist())

# Crear y mostrar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras con nombres de anfitriones')
plt.tight_layout()
plt.show()
