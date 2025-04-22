import pandas as pd
import matplotlib.pyplot as plt

# Diccionario con etiquetas y rutas de los archivos
files = {
    'Lab1': './results/lab2/notebook_test_speed_lab1.csv',
    'SinCos': './results/lab2/notebook_test_speed_sincos.csv',
    'Vectorized': './results/lab2/notebook_test_speed_vectorized.csv'
}


# Leer los CSV y calcular promedio y máximo de PUS
labels = []
promedios = []
maximos = []

for label, path in files.items():
    df = pd.read_csv(path)
    pus = df['pus']
    labels.append(label)
    promedios.append(pus.mean())
    maximos.append(pus.max())

# Posiciones en el eje x
x = list(range(len(labels)))

# Crear el gráfico de líneas con marcadores
fig, ax = plt.subplots(figsize=(8,5))

# Línea y marcadores para promedios
ax.plot(x, promedios,
        marker='o',       # círculo
        linestyle='-',    # línea sólida
        linewidth=2,
        markersize=8,
        label='Promedio P/µs')

# Línea y marcadores para máximos
ax.plot(x, maximos,
        marker='s',       # cuadrado
        linestyle='--',   # línea discontinua
        linewidth=2,
        markersize=8,
        label='Máximo P/µs')

# Ajustar ejes y leyenda
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('P/µs (fotones por microsegundo)')
ax.set_title('Comparación de P/µs entre diferentes implementaciones, PHOTONS=16777216, compilado con gcc -ffast-math -O2 -march=native -flto')

# Que el eje Y empiece en 0
ax.set_ylim(bottom=0)

ax.legend()
plt.tight_layout()
plt.show()