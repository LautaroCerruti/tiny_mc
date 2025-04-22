import csv
import subprocess
import os

def limpiar():
    """
    Elimina el ejecutable 'headless', 'head' y todos los archivos .o generados.
    """
    print("Limpiando archivos generados previamente...")
    subprocess.run("rm -f headless head *.o", shell=True)


def compile_executable(compiler, base_flags, opt_flags, macro):
    """
    Compila el ejecutable 'headless' usando los parámetros pasados.
    base_flags: flags base (e.g., -march=native -flto ...)
    opt_flags: flags de optimización (e.g., -O2)
    macro: valor para -DPHOTONS
    """
    limpiar()

    default_flags = "-std=c11 -Wall -Wextra -g"
    flags = f"{default_flags} {base_flags}"
    if opt_flags:
        flags += f" {opt_flags}"
    if macro is not None:
        flags += f" -DPHOTONS={macro}"

    print(f"\nCompilando con:\n  Compilador: {compiler}\n  Flags: {flags}")

    fuentes = ['tiny_mc.c', 'wtime.c', 'photon.c', 'xoshiro.c']
    objetos = []
    for fuente in fuentes:
        objeto = os.path.splitext(fuente)[0] + '.o'
        print(f"Compilando {fuente} -> {objeto}...")
        cmd = f"{compiler} {flags} -c {fuente} -o {objeto}"
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            print(f"Error compilando {fuente}.")
            return False
        objetos.append(objeto)
    
    ldflags = "-lm"
    objs = " ".join(objetos)
    ejecutable = "headless"
    cmd_link = f"{compiler} {flags} -o {ejecutable} {objs} {ldflags}"
    print("Enlazando objetos para generar el ejecutable headless...")
    res = subprocess.run(cmd_link, shell=True)
    if res.returncode != 0:
        print("Error durante el enlace.")
        return False
    print(f"Compilación exitosa. Ejecutable generado: {ejecutable}")
    return True


def guardar_ejecucion(nombre_ejecucion):
    registro_csv = "./results/registro_flags.csv"
    file_exists = os.path.exists(registro_csv)
    with open(registro_csv, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["compilador_flag"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"compilador_flag": nombre_ejecucion})


def ejecutar_headless(runs, nombre_ejecucion):
    for i in range(runs):
        print(f"Ejecución {i+1} de {runs}...")
        guardar_ejecucion(nombre_ejecucion)
        cmd = "./headless -q -o ./results/notebook_optimize_flags.csv"
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            print("Error al ejecutar headless.")
            break


def main():
    compilacion_csv = './opciones_compilacion_optimize.csv'
    runs_csv = './runs_optimize.csv'

    # Leer runs
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

    # Leer opciones de compilación
    try:
        with open(compilacion_csv, newline='') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                compiler = fila['compiler'].strip()
                base_flags = fila['base_flags'].strip()
                opt_flags = fila['flags'].strip()
                print("\n======================================")
                print(f"Opciones de compilación: Compiler={compiler}, BaseFlags='{base_flags}', Flags='{opt_flags}'")
                print("======================================")
                
                for run in runs_list:
                    photons_macro = run["photons"]
                    runs_count = run["runs"]
                    print(f"\n--- Compilando para -Dphotons={photons_macro} y ejecutando {runs_count} veces ---")
                    if not compile_executable(compiler, base_flags, opt_flags, photons_macro):
                        print("La compilación falló para esta configuración. Se omite la ejecución.")
                        continue
                    clean_base = base_flags.replace('-', '').replace(' ', '')
                    clean_opt = opt_flags.replace('-', '').replace(' ', '')
                    nombre_ejecucion = f"{compiler}_"
                    if clean_opt:
                        nombre_ejecucion += f"{clean_opt}"
                    ejecutar_headless(runs_count, nombre_ejecucion)
    except FileNotFoundError:
        print(f"El archivo {compilacion_csv} no se encontró.")

if __name__ == '__main__':
    main()