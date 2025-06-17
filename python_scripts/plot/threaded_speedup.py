import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el CSV (ajusta la ruta o nombre de archivo si es necesario)
df = pd.read_csv('./results/lab3/atom_threaded_results.csv')

# Calcula el máximo de 'pus' por ('photons','threads')
df_max = df.groupby(['photons', 'threads'], as_index=False)['pus'].max()

# Define aquí tu mapeo de etiquetas
label_map = {
    8388608:   '2^23 fotones',
    16777216:  '2^24 fotones',
    33554432:  '2^25 fotones',
    67108864:  '2^26 fotones',
    134217728: '2^27 fotones'
}

# Prepara los ticks del eje X (solo los threads presentes)
x_ticks = sorted(df_max['threads'].unique())

plt.figure()

# Añade un título al gráfico
plt.title('Rendimiento de P/μs vs Cores')

for size in sorted(df_max['photons'].unique()):
    subset = df_max[df_max['photons'] == size]
    plt.plot(subset['threads'],
             subset['pus'],
             marker='o',
             label=label_map.get(size, f'{size}'))

plt.xlabel('Cores')
# Cambia la etiqueta del eje Y usando el símbolo micro (μ)
plt.ylabel('P/μs')
plt.grid(True)

# Que Y arranque en 0
plt.ylim(bottom=0)

# Que X solo muestre los valores de threads utilizados
plt.xticks(x_ticks)

plt.legend(title='Tamaño problema')
plt.tight_layout()
plt.show()
