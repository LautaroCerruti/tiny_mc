import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_efficiency(csv_file):
    # Leer datos
    df = pd.read_csv(csv_file)

    # Baseline: lab2 con 1 core
    baseline_pus = df[(df['version'] == 'lab2') & (df['cores'] == 1)]['pus'].iloc[0]

    # Filtrar lab3 y calcular eficiencia
    df_lab3 = df[df['version'] == 'lab3'].copy()
    df_lab3.sort_values('cores', inplace=True)
    df_lab3['efficiency'] = (df_lab3['pus'] / df_lab3['cores'] / baseline_pus) * 100

    # Insertar primer punto baseline
    baseline_row = {'version': 'lab2', 'cores': 1, 'pus': baseline_pus, 'efficiency': 100}
    df_plot = pd.concat([pd.DataFrame([baseline_row]), df_lab3], ignore_index=True)

    # Etiquetas y posiciones X
    labels = df_plot.apply(lambda r: f"{r['version']} threads={int(r['cores'])}", axis=1)
    x = list(range(len(df_plot)))
    y = df_plot['efficiency']

    # Graficar
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_xlabel("Configuración")
    ax.set_ylabel("Eficiencia (%)")
    ax.set_title("Eficiencia vs lab2 (1 thread = 100%)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Cambia aquí la ruta a tu CSV
    csv_path = 'results/lab3/best_pus_notebook.csv'
    plot_efficiency(csv_path)
