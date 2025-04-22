import csv
from itertools import cycle

# Nombres de los archivos de entrada y salida
flags_file = './results/registro_flags.csv'
opt_file = './results/atom_optimize_flags.csv'
output_file = './results/atom_optimize_flags_merged.csv'

# Leer el CSV de flags
with open(flags_file, newline='') as f:
    reader = csv.DictReader(f)
    flags_list = [row['compilador_flag'] for row in reader]

# Leer el CSV de resultados (opt_opts)
with open(opt_file, newline='') as f:
    reader = csv.DictReader(f)
    opt_rows = list(reader)

# Determinar el iterador para las flags: si hay menos flags que filas en opt_opts, se repiten
flags_iter = cycle(flags_list) if len(flags_list) < len(opt_rows) else iter(flags_list)

# Crear la lista de filas combinadas. Se define el nuevo encabezado colocando primero 'compilador_flag'
if opt_rows:
    opt_fieldnames = list(opt_rows[0].keys())
else:
    opt_fieldnames = []
new_fieldnames = ['compilador_flag'] + opt_fieldnames

merged_rows = []
for opt_row in opt_rows:
    flag_val = next(flags_iter)
    # Se crea un diccionario con la flag y se agregan las columnas del CSV de opt_opts
    merged_row = {'compilador_flag': flag_val}
    merged_row.update(opt_row)
    merged_rows.append(merged_row)

# Escribir el CSV unificado
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)

print(f"Se ha generado el archivo {output_file} con la unión de ambos CSV.")
