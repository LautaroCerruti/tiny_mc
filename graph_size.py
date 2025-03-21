import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import math

# Cargar el archivo CSV
df = pd.read_csv("./results/photon_size.csv", skipinitialspace=True)

# Calcular los mínimos y máximos globales para la transformación
pus_data_min = df["pus"].min()
pus_data_max = df["pus"].max()
ius_data_min = df["ius"].min()
ius_data_max = df["ius"].max()

# Calcular la relación lineal: IUS = a * PUS + b
a = (ius_data_max - ius_data_min) / (pus_data_max - pus_data_min)
b = ius_data_min - a * pus_data_min

# Agrupar para obtener los valores máximos (líneas principales)
agg = df.groupby("photons").agg(max_pus=("pus", "max"),
                                max_ius=("ius", "max")).reset_index()

# Calcular Q1, Q3 y la media para 'pus'
stats_pus = df.groupby("photons")["pus"].agg(
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    mean="mean"
).reset_index()

# Calcular Q1, Q3 y la media para 'ius'
stats_ius = df.groupby("photons")["ius"].agg(
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    mean="mean"
).reset_index()

# Crear la figura y configurar los ejes
fig, ax1 = plt.subplots(figsize=(10,6))
color1 = 'tab:blue'
color2 = 'tab:red'

# Configuración del eje izquierdo para PUS
ax1.set_xlabel("Photons (escala logarítmica)", labelpad=15)
ax1.set_ylabel("Máximo P/µs", color=color1)
# Línea para máximo PUS con marcador "o"
ax1.plot(agg["photons"], agg["max_pus"], marker="o", color=color1, label="Máximo P/µs")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xscale("log")
ax1.set_xticks(agg["photons"])
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
ax1.tick_params(axis="x", rotation=45)
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.grid(True, which='major', linestyle='--')
ax1.set_ylim(bottom=0.10)  # Forzamos que el eje PUS comience en 0.10

# Configuración del eje derecho para IUS
ax2 = ax1.twinx()
ax2.set_ylabel("Máximo ius", color=color2)
# Línea para máximo ius con marcador "s"
ax2.plot(agg["photons"], agg["max_ius"], marker="s", color=color2, label="Máximo I/µs")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.ticklabel_format(style='plain', axis='y')

# Ajustar la escala del eje IUS para que sea proporcional a la de PUS
pus_ylim = ax1.get_ylim()  # (0.13, upper_limit_PUS)
ax2.set_ylim(a * pus_ylim[0] + b, a * pus_ylim[1] + b)

# Título general
fig.suptitle("Máximo y Promedio P/µs e I/µs según Photons")

# Parámetros para el desplazamiento y ancho en la escala logarítmica
delta = 0.02      # Desplazamiento en el logaritmo para separar las cajas
width_factor = 0.04  # Ancho relativo en el logaritmo

# Dibujar las cajas para PUS (eje izquierdo) desplazadas a la izquierda
for idx, row in stats_pus.iterrows():
    p = row["photons"]
    q1 = row["q1"]
    q3 = row["q3"]
    mean_val = row["mean"]
    # Posición desplazada a la izquierda en la escala logarítmica
    left_pus = p * math.exp(-delta - width_factor/2)
    right_pus = p * math.exp(-delta + width_factor/2)
    width_pus = right_pus - left_pus
    # Dibujar el rectángulo representando el rango intercuartil
    rect = Rectangle((left_pus, q1), width_pus, q3 - q1,
                     facecolor=color1, alpha=0.3, edgecolor=color1)
    ax1.add_patch(rect)

# Dibujar las cajas para ius (eje derecho) desplazadas a la derecha
for idx, row in stats_ius.iterrows():
    p = row["photons"]
    q1 = row["q1"]
    q3 = row["q3"]
    mean_val = row["mean"]
    left_ius = p * math.exp(delta - width_factor/2)
    right_ius = p * math.exp(delta + width_factor/2)
    width_ius = right_ius - left_ius
    rect = Rectangle((left_ius, q1), width_ius, q3 - q1,
                     facecolor=color2, alpha=0.3, edgecolor=color2)
    ax2.add_patch(rect)
    ax2.plot([left_ius, right_ius], [mean_val, mean_val],
             color=color2, linestyle='--', lw=2)

# Agregar líneas que conecten los promedios
# Se usan símbolos distintos: diamante ("D") para PUS y triángulo ("^") para ius.
ax1.plot(stats_pus["photons"], stats_pus["mean"], marker="D", color=color1,
         linestyle="--", label="Promedio P/µs")
ax2.plot(stats_ius["photons"], stats_ius["mean"], marker="^", color=color2,
         linestyle="--", label="Promedio I/µs")

# Combinar leyendas de ambos ejes y moverla abajo a la derecha
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

# Ajustar el layout para evitar recortes y guardar la figura sin mostrarla
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("grafica.png")
plt.close()
