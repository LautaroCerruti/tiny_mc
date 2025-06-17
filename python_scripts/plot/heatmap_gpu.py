#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_pus_heatmap(csv_path, output_path=None):
    # Leer CSV
    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    # Comprobar columnas
    expected = {'threads', 'blocks', 'pus'}
    if not expected.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {expected}. Columnas encontradas: {df.columns.tolist()}")
    
    # Pivot table: máximo de pus por (threads, blocks)
    pivot = (
        df
        .groupby(['threads', 'blocks'])['pus']
        .max()
        .unstack(fill_value=0)
    )
    
    # Plot: más rectangular y margen derecho reducido
    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        pivot,
        interpolation='nearest',
        aspect='auto',
        origin='lower'
    )
    
    # Anotar cada celda
    vmax = pivot.values.max()
    for i, thr in enumerate(pivot.index):
        for j, blk in enumerate(pivot.columns):
            val = pivot.iat[i, j]
            plt.text(
                j, i,
                f"{val:.2f}M P/S",
                ha='center', va='center',
                fontsize=8,
                color='white' if val < vmax/2 else 'black'
            )
    
    # Etiquetas
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns, rotation=45)
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.xlabel('Número de bloques')
    plt.ylabel('Número de hilos por bloque')
    plt.title('Heatmap de P/S máximo')
    
    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('P/S máximo')
    
    # Ajuste de márgenes: right más cercano a 1 → menos margen derecho
    plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.15)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Guardado heatmap en {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera un heatmap de P/S (valor máximo) con anotaciones "P/S" a partir de un CSV'
    )
    parser.add_argument('csv', help='Ruta al archivo CSV de entrada')
    parser.add_argument('-o', '--output', help='Ruta para guardar la imagen (opcional)', default=None)
    args = parser.parse_args()
    
    plot_pus_heatmap(args.csv, args.output)
