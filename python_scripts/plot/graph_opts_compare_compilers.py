import pandas as pd
import matplotlib.pyplot as plt
import re

# Parámetros configurables
photons_value = 16777216  # Cambia este valor según lo que necesites
metric_choice = 'max'   # Usa 'mean' para promedio o 'max' para máximo

# Leer el CSV
df = pd.read_csv('./results/lab2/atom_optimize_flags.csv')

# Convertir las columnas a numérico
df['pus'] = pd.to_numeric(df['pus'], errors='coerce')
df['photons'] = pd.to_numeric(df['photons'], errors='coerce')

# Filtrar el dataframe para el valor de photons deseado
df_filtrado = df[df['photons'] == photons_value].copy()

# Función para extraer el compilador y la flag, removiendo guiones bajos al inicio de la flag
def extraer_compiler_flag(s):
    # Se espera un formato tipo "gcc -O2", "clang -O3" o "icx -O1", etc.
    match = re.match(r'^(gcc|clang|icx)\s*(.*)', s, flags=re.IGNORECASE)
    if match:
        compiler = match.group(1).upper()  # Normaliza a mayúsculas
        flag = match.group(2).strip().lstrip('_')  # Quitar espacios y guiones bajos al inicio
        return pd.Series([compiler, flag])
    else:
        return pd.Series([None, s.strip().lstrip('_')])

# Crear nuevas columnas 'compiler' y 'flag'
df_filtrado[['compiler', 'flag']] = df_filtrado['compilador_flag'].apply(extraer_compiler_flag)

# Imprimir por consola el valor máximo de 'pus' y de qué compilador y flags proviene
max_pus_value = df_filtrado['pus'].max()
max_row = df_filtrado[df_filtrado['pus'] == max_pus_value].iloc[0]
print(f"El valor máximo de 'pus' es: {max_pus_value} obtenido con el compilador {max_row['compiler']} y las flags: {max_row['flag']}")

# Agrupar por 'flag' y 'compiler' calculando la métrica deseada
if metric_choice.lower() == 'mean':
    grouped = df_filtrado.groupby(['flag', 'compiler'])['pus'].mean().reset_index()
    metric_label = "Promedio"
elif metric_choice.lower() == 'max':
    grouped = df_filtrado.groupby(['flag', 'compiler'])['pus'].max().reset_index()
    metric_label = "Máximo"
else:
    raise ValueError("El parámetro metric_choice debe ser 'mean' o 'max'.")

# Pivotear los datos para tener las flags en el eje x y columnas para cada compilador
pivot = grouped.pivot(index='flag', columns='compiler', values='pus')
pivot = pivot.sort_index()  # Ordena las flags alfabéticamente

# Graficar: se traza una línea por cada compilador
plt.figure(figsize=(10, 6))
for compiler in pivot.columns:
    plt.plot(pivot.index, pivot[compiler], marker='o', label=compiler)

plt.xlabel('Flag utilizada')
plt.ylabel(f'P/µs ({metric_label})')
plt.title(f'P/µs ({metric_label}) para {photons_value} photons\npor flag y compilador')
plt.xticks(rotation=45)
plt.legend(title='Compilador')
plt.grid(True)
plt.tight_layout()

# Guardar la figura en un archivo PNG
filename = f'flags_compilador_{metric_choice}_{photons_value}.png'
plt.savefig(filename)
plt.close()
print(f"Imagen guardada en: {filename}")
