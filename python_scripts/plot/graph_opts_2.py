import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV
df = pd.read_csv('./results/lab2/atom_optimize_flags.csv')

# Asegurar que 'pus' sea float
df['pus'] = pd.to_numeric(df['pus'], errors='coerce')

# Agrupar por compilador_flag, obteniendo el máximo de pus
grouped = (
    df
    .groupby('compilador_flag')['pus']
    .max()
    .reset_index(name='pus_max')
)

# Extraer el nombre de compilador para asignar colores
def get_compiler(name):
    low = name.lower()
    if 'gcc' in low:
        return 'GCC'
    if 'clang' in low:
        return 'Clang'
    if 'icx' in low:
        return 'ICX'
    return 'Otro'

grouped['compiler'] = grouped['compilador_flag'].apply(get_compiler)

# Mapa de colores por compilador
color_map = {
    'GCC':   plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
    'Clang': plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
    'ICX':   plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
    'Otro':  plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
}
colors = grouped['compiler'].map(color_map)

# Plot de barras
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(grouped['compilador_flag'], grouped['pus_max'], color=colors)

# Anotar el valor máximo sobre cada barra
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),            # desplazamiento vertical
        textcoords='offset points',
        ha='center',
        va='bottom'
    )

# Etiquetas y ajustes
ax.set_xlabel('Flag de compilación')
ax.set_ylabel('P/µs (máximo)')
ax.set_title('P/µs máximo por flag de compilador')
ax.set_xticklabels(grouped['compilador_flag'], rotation=45, ha='right')
# Quitar la leyenda
ax.legend_.remove() if ax.get_legend() else None

plt.tight_layout()

# Guardar
filename = 'max_pus_por_flag.png'
plt.savefig(filename)
plt.close()

print(f'Gráfico guardado en: {filename}')
