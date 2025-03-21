import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV
df = pd.read_csv('./results/merged.csv')

# Opcional: asegurar que las columnas numéricas sean de tipo float
df['pus'] = pd.to_numeric(df['pus'], errors='coerce')
df['ius'] = pd.to_numeric(df['ius'], errors='coerce')
df['photons'] = pd.to_numeric(df['photons'], errors='coerce')

# Filtrar por compilador: gcc y clang
df_gcc = df[df['compilador_flag'].str.contains('gcc', case=False)]
df_clang = df[df['compilador_flag'].str.contains('clang', case=False)]

def procesar_datos(data):
    # Agrupar por compilador_flag y photons, calculando promedio y máximo de pus e ius
    agrupado = data.groupby(['compilador_flag', 'photons']).agg({
        'pus': ['mean', 'max'],
        'ius': ['mean', 'max']
    }).reset_index()
    # Aplanar el MultiIndex de columnas
    agrupado.columns = ['compilador_flag', 'photons', 'pus_mean', 'pus_max', 'ius_mean', 'ius_max']
    return agrupado

grouped_gcc = procesar_datos(df_gcc)
grouped_clang = procesar_datos(df_clang)

def plot_metric(agrupado, compiler_label, metric='pus'):
    """
    Grafica para la métrica indicada (pus o ius) de los datos agrupados.
    Se generan dos líneas para cada valor de photons: promedio (línea punteada) y máximo (línea continua).
    """
    # Extraer los valores únicos de photons (ordenados)
    unique_photons = sorted(agrupado['photons'].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # para variar los marcadores si hay muchos grupos
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure(figsize=(10, 6))
    for i, photon in enumerate(unique_photons):
        # Filtrar y ordenar según compilador_flag para el valor actual de photons
        sub_df = agrupado[agrupado['photons'] == photon].sort_values('compilador_flag')
        x = sub_df['compilador_flag']
        mean_col = f"{metric}_mean"
        max_col = f"{metric}_max"
        
        # Promedio: línea punteada
        plt.plot(x, sub_df[mean_col], marker=markers[i % len(markers)], linestyle=':',
                 color=colors[i % len(colors)], label=f'Photons {photon} promedio')
        # Máximo: línea continua
        plt.plot(x, sub_df[max_col], marker=markers[i % len(markers)], linestyle='-',
                 color=colors[i % len(colors)], label=f'Photons {photon} maximo')
    
    plt.xlabel('compilador_flag')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()}: Promedio y Máximo para {compiler_label}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Si se desea guardar el gráfico:
    # plt.savefig(f'{compiler_label.lower()}_{metric}.png')

# Graficar para GCC (si hay datos)
if not grouped_gcc.empty:
    plot_metric(grouped_gcc, 'GCC', metric='pus')
    plot_metric(grouped_gcc, 'GCC', metric='ius')
else:
    print("No se encontraron datos para GCC.")

# Graficar para Clang (si hay datos)
if not grouped_clang.empty:
    plot_metric(grouped_clang, 'Clang', metric='pus')
    plot_metric(grouped_clang, 'Clang', metric='ius')
else:
    print("No se encontraron datos para Clang.")
