#!/usr/bin/env python3
import csv

# ————— Configuración —————
CSV_FILE = "results/lab3/notebook_threaded_results.csv"      # <- aquí pones el nombre de tu archivo CSV
TARGET_THREADS = 4          # <- aquí pones el número de threads que quieres filtrar
# ————————————————————————

def main():
    max_pus = None
    best_row = None

    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            try:
                threads = int(row['threads'])
                pus = float(row['pus'])
            except ValueError:
                # saltamos filas mal formateadas
                continue

            if threads == TARGET_THREADS:
                if best_row is None or pus > max_pus:
                    max_pus = pus
                    best_row = row

    if best_row:
        # imprimimos la fila completa, respetando el orden de columnas original
        print(','.join(best_row[col] for col in fieldnames))
    else:
        print(f"No se encontraron filas con threads = {TARGET_THREADS}")

if __name__ == "__main__":
    main()