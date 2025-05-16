from scipy.stats import f_oneway, ttest_ind
import pandas as pd

# Cargar el dataset
df = pd.read_csv('csv/AB_NYC_2019.csv') 

# Eliminar filas con valores nulos en 'room_type' o 'price'
df.dropna(subset=['room_type', 'price'], inplace=True)

# ANOVA: comparar precios entre tipos de habitación
anova_result = f_oneway(
    df[df['room_type'] == 'Entire home/apt']['price'],
    df[df['room_type'] == 'Private room']['price'],
    df[df['room_type'] == 'Shared room']['price']
)
print('Resultado ANOVA:', anova_result)

# T-test: comparar precios entre 'Entire home/apt' y 'Private room'
ttest_result = ttest_ind(
    df[df['room_type'] == 'Entire home/apt']['price'],
    df[df['room_type'] == 'Private room']['price'],
    equal_var=False  # Welch’s t-test, por si las varianzas son distintas
)
print('Resultado T-test:', ttest_result)
