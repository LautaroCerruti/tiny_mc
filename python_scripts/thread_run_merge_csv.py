import pandas as pd

# Lee ambos CSV
threads_df = pd.read_csv('results/registro_cores.csv')       # Contiene columna "threads"
data_df    = pd.read_csv('results/notebook_cores.csv')          # Contiene columnas "photons,time,pus"

# Concatena lado a lado (por índice)
merged_df  = pd.concat([threads_df, data_df], axis=1)

# Guarda el resultado
merged_df.to_csv('merged.csv', index=False)
