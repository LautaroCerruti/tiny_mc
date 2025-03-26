import csv
import subprocess
import os

def limpiar():
    """
    Elimina el ejecutable 'headless', 'head' y todos los archivos .o generados.
    """
    print("Limpiando archivos generados previamente...")
    subprocess.run("rm -f headless head *.o", shell=True)

def compile_executable(compiler, additional_flags, usar_lto, usar_native, macro):
    """
    Compila el ejecutable 'headless' usando los parámetros pasados.
    """
    # Limpiar archivos previos antes de compilar
    limpiar()

    default_flags = "-std=c11 -Wall -Wextra"
    flags = default_flags
    if additional_flags:
        flags += " " + additional_flags
    if usar_lto and '-flto' not in flags:
        flags += " -flto"
    if usar_native and '-march=native' not in flags:
        flags += " -march=native"
    if macro is not None:
        flags += f" -DPHOTONS={macro}"

    print(f"\nCompilando con:\n  Compilador: {compiler}\n  Flags: {flags}")

    # Lista de archivos fuente a compilar
    fuentes = ['tiny_mc.c', 'wtime.c', 'photon.c', 'xoshiro.c']
    objetos = []
    for fuente in fuentes:
        objeto = os.path.splitext(fuente)[0] + '.o'
        print(f"Compilando {fuente} -> {objeto}...")
        cmd_compilacion = f"{compiler} {flags} -c {fuente} -o {objeto}"
        resultado = subprocess.run(cmd_compilacion, shell=True)
        if resultado.returncode != 0:
            print(f"Error compilando {fuente}.")
            return False
        objetos.append(objeto)
    
    # Opciones de enlace (en este ejemplo, -lm)
    ldflags = "-lm"
    objetos_enlazar = " ".join(objetos)
    ejecutable = "headless"
    cmd_enlace = f"{compiler} {flags} -o {ejecutable} {objetos_enlazar} {ldflags}"
    print("Enlazando objetos para generar el ejecutable headless...")
    resultado = subprocess.run(cmd_enlace, shell=True)
    if resultado.returncode != 0:
        print("Error durante el enlace.")
        return False
    print(f"Compilación exitosa. Ejecutable generado: {ejecutable}")
    return True

def guardar_ejecucion(nombre_ejecucion):
    """
    Guarda en el archivo 'registro_flags.csv' el nombre de la ejecución.
    Se usa una única columna llamada 'compilador_flag'.
    """
    registro_csv = "./results/registro_flags.csv"
    file_exists = os.path.exists(registro_csv)
    with open(registro_csv, "a", newline="") as csvfile:
        fieldnames = ["compilador_flag"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"compilador_flag": nombre_ejecucion})

def ejecutar_headless(runs, nombre_ejecucion):
    """
    Ejecuta el ejecutable 'headless' la cantidad de veces indicada.
    Antes de cada ejecución se guarda en el CSV el nombre de la ejecución.
    """
    for i in range(runs):
        print(f"Ejecución {i+1} de {runs}...")
        guardar_ejecucion(nombre_ejecucion)
        cmd = "./headless -q -o ./results/photon_size_atom_xoshiro_opt.csv"
        resultado = subprocess.run(cmd, shell=True)
        if resultado.returncode != 0:
            print("Error al ejecutar headless.")
            break

def main():
    compilacion_csv = './opciones_compilacion_photon_size_opt.csv'
    runs_csv = './runs_photon_size.csv'

    # Cargar el archivo runs.csv en una lista
    runs_list = []
    try:
        with open(runs_csv, newline='') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                photons = fila['photons'].strip()
                try:
                    runs_count = int(fila['runs'].strip())
                except ValueError:
                    print("Valor inválido para 'runs' en el archivo runs.csv. Se omite esta fila.")
                    continue
                runs_list.append({"photons": photons, "runs": runs_count})
    except FileNotFoundError:
        print(f"El archivo {runs_csv} no se encontró.")
        return

    # Para cada fila del CSV de opciones de compilación, se recorre cada fila de runs
    try:
        with open(compilacion_csv, newline='') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                compiler = fila['compiler'].strip()
                additional_flags = fila['flags'].strip()
                usar_lto = fila['lto'].strip().lower() in ['true', '1', 'yes']
                usar_native = fila['native'].strip().lower() in ['true', '1', 'yes']
                print("\n======================================")
                print(f"Opciones de compilación: Compiler={compiler}, Flags adicionales='{additional_flags}', LTO={usar_lto}, Native={usar_native}")
                print("======================================")
                
                for run in runs_list:
                    photons_macro = run["photons"]
                    runs_count = run["runs"]
                    print(f"\n--- Compilando para -Dphotons={photons_macro} y ejecutando {runs_count} veces ---")
                    if not compile_executable(compiler, additional_flags, usar_lto, usar_native, photons_macro):
                        print("La compilación falló para esta configuración. Se omite la ejecución.")
                        continue
                    # Eliminar '-' y espacios de additional_flags
                    clean_flags = additional_flags.replace("-", "").replace(" ", "")
                    nombre_ejecucion = f"{compiler}_{clean_flags}"
                    if usar_lto:
                        nombre_ejecucion += "_lto"
                    if usar_native:
                        nombre_ejecucion += "_native"
                    ejecutar_headless(runs_count, nombre_ejecucion)
    except FileNotFoundError:
        print(f"El archivo {compilacion_csv} no se encontró.")

if __name__ == '__main__':
    main()
