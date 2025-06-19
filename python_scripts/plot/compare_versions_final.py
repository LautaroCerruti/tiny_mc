import pandas as pd
import matplotlib.pyplot as plt

# 1) Define aquí tus archivos (ruta → etiqueta)
files = {
    'Original Atom': './results/lab1/photon_size_atom.csv',
    'Lab1 Atom': './results/lab1/photon_size_atom_xoshiro_opt.csv',
    'Lab2 Atom': './results/lab2/atom_best_flag_v2.csv',
    'Lab3 Atom': './results/lab3/best_pus_atom.csv',
    'Lab4 GPU GTX 1060': './results/lab4/entrega/gtx1060_by_size.csv',
    'Final GPU GTX 1060': './results/lab4/gtx1060_by_size_v2.csv',
    'Lab4 GPU GTX 2080 TI': './results/lab4/entrega/gtx2080ti_by_size.csv',
    'Lab4 GPU Titan Xp': './results/lab4/entrega/titanxp_by_size.csv',
    'Final GPU Titan Xp': './results/lab4/titanxp_by_size_v2.csv',
    'Final GPU GTX 2080 TI': './results/lab4/gtx2080ti_by_size_v2.csv',
    # ...
}

labels = []
max_values = []

for label, path in files.items():
    df = pd.read_csv(path)
    max_pus = df['pus'].max()
    labels.append(label)
    max_values.append(max_pus)
# 2) Graficar
x = range(len(labels))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, max_values)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('P/s (millones de fotones por segundo)')
ax.set_title('Comparación de P/s Máximo obtenido')

# Aquí aseguramos que Y empiece en 0 y deje un 10% de espacio por encima de la barra más alta
y_top = max(max_values) * 1.10
ax.set_ylim(0, y_top)

# 3) Anotar cada barra con el valor "X.XX M"
for i, v in enumerate(max_values):
    ax.text(i, v * 1.01, f"{v:.2f}M", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()