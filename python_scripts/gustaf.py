import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_csv('results/lab3/notebook_gustaf.csv')  # asegúrate de poner la ruta correcta

# 2. Tiempo medio por número de hilos
min_times = df.groupby('threads')['time'].min()  # Serie indexed by threads

# 3. Speedup escalado
T1 = min_times.loc[1]
threads = min_times.index.values
speedup = threads * T1 / min_times.values

# 4. Curva ideal
ideal = threads

# 5. Plot
plt.figure(figsize=(6,4))
plt.plot(threads, speedup, 'o-', label='Speedup escalado (Gustafson)')
plt.plot(threads, ideal, '--', label='Speedup ideal (S=N)')
plt.xlabel('Número de threads')
plt.ylabel('Speedup escalado S(N)')
plt.title('Ley de Gustafson: weak scaling')
plt.xticks(threads)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()