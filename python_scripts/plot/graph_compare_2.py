import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Leer los dos archivos CSV
df1 = pd.read_csv("./results/photon_size_atom.csv", skipinitialspace=True)
df2 = pd.read_csv("./results/photon_size_atom_xoshiro.csv", skipinitialspace=True)

# Agrupar y calcular las estadísticas para cada archivo:
# Se calcula el máximo y el promedio de PUS e IUS para cada valor de photons.
agg1 = df1.groupby("photons").agg(
    max_pus=("pus", "max"),
    mean_pus=("pus", "mean"),
    max_ius=("ius", "max"),
    mean_ius=("ius", "mean")
).reset_index()

agg2 = df2.groupby("photons").agg(
    max_pus=("pus", "max"),
    mean_pus=("pus", "mean"),
    max_ius=("ius", "max"),
    mean_ius=("ius", "mean")
).reset_index()

# Graficar para PUS
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale("log")
ax.set_xlabel("Cantidad de Fotones (escala logarítmica)", labelpad=15)
ax.set_ylabel("P/µs")
ax.set_xticks(agg1["photons"])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which='major', linestyle='--')
ax.legend(framealpha=0.5)

# Línea para archivo 1: PUS máximo y promedio
ax.plot(agg1["photons"], agg1["max_pus"], marker="o", linestyle="-", color="tab:blue", label="Máximo P/µs rand libc")
ax.plot(agg1["photons"], agg1["mean_pus"], marker="D", linestyle="--", color="tab:blue", label="Promedio P/µs rand libc")

# Línea para archivo 2: PUS máximo y promedio
ax.plot(agg2["photons"], agg2["max_pus"], marker="o", linestyle="-", color="tab:red", label="Máximo P/µs xoshiro256+")
ax.plot(agg2["photons"], agg2["mean_pus"], marker="D", linestyle="--", color="tab:red", label="Promedio P/µs xoshiro256+")

ax.legend()
plt.title("Comparación de Promedio y Máximo de P/µs")
plt.tight_layout()
plt.savefig("grafica_pus.png")
plt.close()

# Graficar para IUS
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale("log")
ax.set_xlabel("Cantidad de Fotones (escala logarítmica)", labelpad=15)
ax.set_ylabel("I/µs")
ax.set_xticks(agg1["photons"])  # Se asume que los valores de photons son iguales en ambos archivos
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which='major', linestyle='--')
ax.legend(framealpha=0.5)

# Línea para archivo 1: IUS máximo y promedio
ax.plot(agg1["photons"], agg1["max_ius"], marker="s", linestyle="-", color="tab:blue", label="Máximo IUS Archivo 1")
ax.plot(agg1["photons"], agg1["mean_ius"], marker="^", linestyle="--", color="tab:blue", label="Promedio IUS Archivo 1")

# Línea para archivo 2: IUS máximo y promedio
ax.plot(agg2["photons"], agg2["max_ius"], marker="s", linestyle="-", color="tab:red", label="Máximo IUS Archivo 2")
ax.plot(agg2["photons"], agg2["mean_ius"], marker="^", linestyle="--", color="tab:red", label="Promedio IUS Archivo 2")

ax.legend()
plt.title("Comparación de Promedio y Máximo de I/µs")
plt.tight_layout()
plt.savefig("grafica_ius.png")
plt.close()
