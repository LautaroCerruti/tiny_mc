import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Definir la lista de archivos y sus etiquetas
files = [
    {"path": "./results/photon_size_atom.csv", "label": "Atom rand()"},
    {"path": "./results/photon_size_atom_opt.csv", "label": "Atom rand() opt"},
    {"path": "./results/photon_size_atom_xoshiro.csv", "label": "Atom xoshiro256+"},
    {"path": "./results/photon_size_atom_xoshiro_opt.csv", "label": "Atom xoshiro256+ opt"},
    {"path": "./results/photon_size_notebook.csv", "label": "Local rand()"},
    {"path": "./results/photon_size_notebook_opt.csv", "label": "Local rand() opt"},
    {"path": "./results/photon_size_notebook_xoshiro.csv", "label": "Local xoshiro256+"},
    {"path": "./results/photon_size_notebook_xoshiro_opt.csv", "label": "Local xoshiro256+ opt"}
]

max_pus_values = []
labels = []

# Filtrar y obtener el máximo de la columna "pus" solo para filas donde "photons" es 1048576
for f in files:
    df = pd.read_csv(f["path"], skipinitialspace=True)
    df_filtered = df[df["photons"] == 1048576]
    max_val = df_filtered["pus"].max()
    max_pus_values.append(max_val)
    labels.append(f["label"])

# Crear el gráfico de barras con la mitad de ancho
x = np.arange(len(labels))
width = 0.4  # Mitad del ancho por defecto

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, max_pus_values, width=width, color="purple", alpha=0.8)

# Configurar el gráfico
ax.set_ylabel("Máximo P/µs")
ax.set_title("Máximos de P/µs (photons = 1048576) por archivo")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")

# Agregar etiquetas con el valor sobre cada barra
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom")

plt.tight_layout()
plt.savefig("maximos_p_us.png")
plt.show()
