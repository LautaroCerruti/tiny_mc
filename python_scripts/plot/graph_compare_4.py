import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Leer los 4 archivos CSV
df1 = pd.read_csv("./results/photon_size_atom_opt.csv", skipinitialspace=True)
df2 = pd.read_csv("./results/photon_size_atom_xoshiro_opt.csv", skipinitialspace=True)
df3 = pd.read_csv("./results/photon_size_notebook_opt.csv", skipinitialspace=True)
df4 = pd.read_csv("./results/photon_size_notebook_xoshiro_opt.csv", skipinitialspace=True)

# Función para agrupar y calcular las estadísticas: máximo y promedio de P/µs e I/µs
def get_agg(df):
    return df.groupby("photons").agg(
        max_pus=("pus", "max"),
        mean_pus=("pus", "mean"),
        max_ius=("ius", "max"),
        mean_ius=("ius", "mean")
    ).reset_index()

agg1 = get_agg(df1)
agg2 = get_agg(df2)
agg3 = get_agg(df3)
agg4 = get_agg(df4)

# Configuración para cada archivo:
# Usaremos el mismo marker para archivos "atom" y otro para "notebook".
files_config = [
    {"agg": agg1, "label": "Atom rand() opt", "marker": "o", "color": "tab:blue"},
    {"agg": agg2, "label": "Atom xoshiro256+ opt", "marker": "o", "color": "tab:red"},
    {"agg": agg3, "label": "Local rand() opt", "marker": "^", "color": "tab:green"},
    {"agg": agg4, "label": "Local xoshiro256+ opt", "marker": "^", "color": "tab:orange"},
]

# -------------------- Gráfico para P/µs --------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale("log")
ax.set_xlabel("Cantidad de Fotones (escala logarítmica)", labelpad=15)
ax.set_ylabel("P/µs")
ax.set_xticks(agg1["photons"])  # Se asume que los valores de photons son comunes entre archivos
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
ax.xaxis.set_minor_locator(ticker.NullLocator())  # Eliminar ticks menores
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which="major", linestyle="--")

for config in files_config:
    agg = config["agg"]
    label = config["label"]
    marker = config["marker"]
    color = config["color"]
    # Línea sólida para el máximo y línea discontinua para el promedio
    ax.plot(agg["photons"], agg["max_pus"], marker=marker, linestyle="-", color=color,
            label=f"{label} Max P/µs")
    ax.plot(agg["photons"], agg["mean_pus"], marker=marker, linestyle="--", color=color,
            label=f"{label} Mean P/µs")

ax.legend(framealpha=0.3)
plt.title("Comparación de Promedio y Máximo de P/µs")
plt.tight_layout()
plt.savefig("grafica_p_us.png")
plt.close()

# -------------------- Gráfico para I/µs --------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale("log")
ax.set_xlabel("Cantidad de Fotones (escala logarítmica)", labelpad=15)
ax.set_ylabel("I/µs")
ax.set_xticks(agg1["photons"])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
ax.xaxis.set_minor_locator(ticker.NullLocator())  # Eliminar ticks menores
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which="major", linestyle="--")

for config in files_config:
    agg = config["agg"]
    label = config["label"]
    marker = config["marker"]
    color = config["color"]
    ax.plot(agg["photons"], agg["max_ius"], marker=marker, linestyle="-", color=color,
            label=f"{label} Max I/µs")
    ax.plot(agg["photons"], agg["mean_ius"], marker=marker, linestyle="--", color=color,
            label=f"{label} Mean I/µs")

ax.legend(framealpha=0.3)
plt.title("Comparación de Promedio y Máximo de I/µs")
plt.tight_layout()
plt.savefig("grafica_i_us.png")
plt.close()
