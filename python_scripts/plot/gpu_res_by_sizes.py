import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el CSV
df = pd.read_csv("results/titanxp_by_size_v2.csv")

# Agrupar por cantidad de fotones y calcular promedio de PUS
grouped = df.groupby('photons')['pus'].max().reset_index()
grouped['pus_millones'] = grouped['pus']

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(grouped['photons'], grouped['pus_millones'], marker='o', linestyle='-')

# Escala logarítmica en X
plt.xscale('log')

# Valores de X y sus etiquetas como potencias de 2
x_vals = grouped['photons'].values
x_labels = [f"$2^{{{int(np.log2(x))}}}$" for x in x_vals]
plt.xticks(x_vals, x_labels, rotation=0)

# Desactivar minor ticks
plt.gca().tick_params(axis='x', which='minor', bottom=False)

# Forzar el eje Y a empezar en 0
plt.ylim(bottom=0)

# Etiquetas y título
plt.xlabel("Fotones simulados")
plt.ylabel("P/s (millones de fotones/segundo)")
plt.title("Rendimiento de simulación vs Fotones simulados en GTX 1060")
plt.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
