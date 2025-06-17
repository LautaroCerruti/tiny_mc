import pandas as pd
import matplotlib.pyplot as plt

# Diccionario con etiquetas y rutas de los archivos
files = {
    'Atomic Global': './results/kernel_atomic_global.csv',
    'Atomic Shared': './results/kernel_shared.csv',
    'Polares': './results/kernel_polares.csv',
    'Xoshiro': './results/kernel_xoshiro.csv',
    'Rsqrt': './results/kernel_rsqrt.csv',
}

# Leer los CSV y calcular máximo de PUS
labels = []
maximos = []

for label, path in files.items():
    df = pd.read_csv(path)
    pus = df['pus']
    labels.append(label)
    maximos.append(pus.max())

# Posiciones en el eje x
x = list(range(len(labels)))

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, maximos, color='purple')

# Ajustar etiquetas y título
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('P/s (millones de fotones por segundo)')
ax.set_title('Comparación de P/s (en millones) entre implementaciones en GTX 1060')
ax.set_ylim(bottom=0)

# Mostrar los valores numéricos encima de cada barra con "M"
for i, v in enumerate(maximos):
    ax.text(i, v + max(maximos)*0.01, f"{v:.2f}M", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
