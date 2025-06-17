import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ruta a tu CSV
csv_path = "./results/lab3/best_pus_notebook.csv"  # ← Cámbialo por la ruta correcta

# Leer el CSV
df = pd.read_csv(csv_path, skipinitialspace=True)

# Crear las etiquetas "version cores=N"
labels = [f"{ver} theads={cores}" for ver, cores in zip(df["version"], df["cores"])]

# Valores de P/µs
pus_values = df["pus"].values

# Posiciones en el eje X
x = np.arange(len(labels))
width = 0.4  # Mitad del ancho por defecto

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, pus_values, width=width, color="purple", alpha=0.8)

# Configuración del gráfico
ax.set_ylabel("P/µs")
ax.set_title("P/µs por versión y número de threads")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")

# Anotar cada barra con su valor
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom")

plt.tight_layout()
plt.savefig("pus_por_version_cores.png")
plt.show()
