import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV
df = pd.read_csv('./results/optimize_wflags_atom.csv')

# Opcional: asegurar que las columnas numéricas sean de tipo float
df['pus'] = pd.to_numeric(df['pus'], errors='coerce')
df['photons'] = pd.to_numeric(df['photons'], errors='coerce')

# Filtrar por compilador: gcc, clang e icx
df_gcc = df[df['compilador_flag'].str.contains('gcc', case=False)]
df_clang = df[df['compilador_flag'].str.contains('clang', case=False)]
df_icx = df[df['compilador_flag'].str.contains('icx', case=False)]

def procesar_datos(data):
    # Agrupar por compilador_flag y photons, calculando promedio y máximo de pus
    agrupado = data.groupby(['compilador_flag', 'photons']).agg({
        'pus': ['mean', 'max']
    }).reset_index()
    # Aplanar el MultiIndex de columnas
    agrupado.columns = ['compilador_flag', 'photons', 'pus_mean', 'pus_max']
    return agrupado

grouped_gcc = procesar_datos(df_gcc)
grouped_clang = procesar_datos(df_clang)
grouped_icx = procesar_datos(df_icx)

def plot_metric(agrupado, compiler_label, metric='pus'):
    """
    Grafica para la métrica indicada (en este caso solo 'pus') de los datos agrupados.
    Se generan dos líneas para cada valor de photons: promedio (línea punteada) y máximo (línea continua).
    Para 'pus' se usa la etiqueta "P/µs".
    El gráfico se guarda en un archivo PNG.
    """
    # Definir la etiqueta personalizada para 'pus'
    if metric == 'pus':
        label_metric = "P/µs"
    else:
        label_metric = metric.upper()
        
    # Extraer los valores únicos de photons (ordenados)
    unique_photons = sorted(agrupado['photons'].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure(figsize=(10, 6))
    for i, photon in enumerate(unique_photons):
        sub_df = agrupado[agrupado['photons'] == photon].sort_values('compilador_flag')
        x = sub_df['compilador_flag']
        mean_col = f"{metric}_mean"
        max_col = f"{metric}_max"
        
        # Promedio: línea punteada
        plt.plot(x, sub_df[mean_col], marker=markers[i % len(markers)], linestyle=':',
                 color=colors[i % len(colors)], label=f'{photon} fotones promedio')
        # Máximo: línea continua
        plt.plot(x, sub_df[max_col], marker=markers[i % len(markers)], linestyle='-',
                 color=colors[i % len(colors)], label=f'{photon} fotones máximo')
    
    plt.xlabel('compilador_flag')
    plt.ylabel(label_metric)
    plt.title(f'{label_metric}: Promedio y Máximo para {compiler_label}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.ylim(0.175, 0.86)  # Limitar el eje Y
    plt.tight_layout()
    
    # Guardar el gráfico en un archivo PNG y cerrar la figura
    filename = f'{compiler_label.lower()}_{metric}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Imagen guardada en: {filename}')

# Graficar y guardar para GCC (si hay datos)
if not grouped_gcc.empty:
    plot_metric(grouped_gcc, 'GCC', metric='pus')
else:
    print("No se encontraron datos para GCC.")

# Graficar y guardar para Clang (si hay datos)
if not grouped_clang.empty:
    plot_metric(grouped_clang, 'Clang', metric='pus')
else:
    print("No se encontraron datos para Clang.")

# Graficar y guardar para ICX (si hay datos)
if not grouped_icx.empty:
    plot_metric(grouped_icx, 'ICX', metric='pus')
else:
    print("No se encontraron datos para ICX.")
