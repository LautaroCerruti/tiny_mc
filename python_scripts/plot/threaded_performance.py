import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# Cargar datos desde el CSV (ajusta la ruta o nombre de archivo si es necesario)
df = pd.read_csv('./results/lab3/atom_threaded_results.csv')

# Calcula el tiempo mínimo ('time') por ('photons','threads')
df_min = df.groupby(['photons', 'threads'], as_index=False)['time'].min()

# Mapeo de etiquetas
label_map = {
    8388608:   '2^23 fotones',
    16777216:  '2^24 fotones',
    33554432:  '2^25 fotones',
    67108864:  '2^26 fotones',
    134217728: '2^27 fotones'
}

# Ticks deseados
x_ticks = [1, 6, 24, 48]
y_ticks = [0, 10, 20, 30, 40]

plt.figure()
plt.title('Tiempo mínimo (s) vs Threads')

# Plot por cada tamaño de problema
for size in sorted(df_min['photons'].unique()):
    subset = df_min[df_min['photons'] == size]
    plt.plot(
        subset['threads'],
        subset['time'],
        marker='o',
        label=label_map[size]
    )

plt.xlabel('Threads')
plt.ylabel('Segundos (s)')

# Ejes logarítmicos (symlog en Y para poder incluir el 0)
plt.xscale('log')
plt.yscale('symlog', linthresh=1)

ax = plt.gca()

# Configura únicamente los ticks mayores en X
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in x_ticks]))
ax.xaxis.set_minor_locator(NullLocator())

# Configura únicamente los ticks mayores en Y
ax.yaxis.set_major_locator(FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in y_ticks]))
ax.yaxis.set_minor_locator(NullLocator())

# Cuadrícula solo para esos ticks
plt.grid(which='major', linestyle='--', alpha=0.7)

plt.legend(title='Tamaño problema')
plt.tight_layout()
plt.show()