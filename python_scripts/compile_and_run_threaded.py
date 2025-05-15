import csv
import subprocess
import os

# === Configuración inicial ===
compiler = "gcc"                            # Compilador a usar
global_compile_flags = "-std=c11 -Wall -Wextra -O2 -ffast-math -fopenmp -fdisable-tree-cunrolli -ftree-vectorize -march=native -flto"  # Flags de compilación fijas
sources = ["tiny_mc.c", "wtime.c", "photon.c", "xoshiro.c"]
executable = "headless"                     # Nombre del ejecutable
dirs = {"results": "results"}

# Nombre del CSV de ejecuciones (nthreads, photons, runs)
runs_csv = "runs_threading_atom.csv"
# Registro de ejecuciones realizadas
environment_log = os.path.join(dirs["results"], "registro_cores.csv")
# Archivo de salida generado por el ejecutable
output_csv = os.path.join(dirs["results"], "atom_cores.csv")


def limpiar():
    """
    Elimina el ejecutable y todos los .o previos.
    """
    print("Limpiando archivos generados previamente...")
    subprocess.run(f"rm -f {executable} *.o", shell=True)


def compile_executable(photons_macro):
    """
    Compila el ejecutable usando las flags fijas y -DPHOTONS.
    """
    limpiar()
    flags = f"-std=c11 -Wall -Wextra {global_compile_flags} -DPHOTONS={photons_macro}"

    print(f"\nCompilando con: {compiler} {flags}")
    objetos = []
    for src in sources:
        obj = os.path.splitext(src)[0] + '.o'
        print(f"  {src} -> {obj}")
        cmd = f"{compiler} {flags} -c {src} -o {obj}"
        if subprocess.run(cmd, shell=True).returncode != 0:
            print(f"Error compilando {src}.")
            return False
        objetos.append(obj)

    print("Enlazando objetos...")
    objs = ' '.join(objetos)
    cmd_link = f"{compiler} {flags} -o {executable} {objs} -lm"
    if subprocess.run(cmd_link, shell=True).returncode != 0:
        print("Error durante el enlace.")
        return False

    print(f"✅ Compilación exitosa: {executable}")
    return True


def guardar_ejecucion(nthreads):
    """
    Guarda solo la cantidad de threads en el CSV.
    """
    os.makedirs(dirs["results"], exist_ok=True)
    existe = os.path.exists(environment_log)
    with open(environment_log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threads"])
        if not existe:
            writer.writeheader()
        writer.writerow({"threads": nthreads})


def ejecutar_headless(nthreads, runs, photons):
    """
    Ejecuta headless runs veces con OMP_NUM_THREADS.
    """
    for i in range(1, runs+1):
        print(f"Ejecución {i}/{runs} — threads={nthreads}, photons={photons}")
        guardar_ejecucion(nthreads)
        cmd = f"OMP_NUM_THREADS={nthreads} ./{executable} -q -o {output_csv}"
        if subprocess.run(cmd, shell=True).returncode != 0:
            print("Error al ejecutar headless.")
            break


def main():
    if not os.path.exists(runs_csv):
        print(f"ERROR: No se encontró {runs_csv}")
        return

    with open(runs_csv, newline="") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            try:
                nthreads = int(fila['nthreads'].strip())
                photons  = fila['photons'].strip()
                runs     = int(fila['runs'].strip())
            except Exception as e:
                print(f"Fila inválida {fila}: {e}. Se omite.")
                continue

            print("\n" + "="*40)
            print(f"Config: threads={nthreads}, photons={photons}, runs={runs}")
            print("="*40)

            if not compile_executable(photons):
                continue

            ejecutar_headless(nthreads, runs, photons)


if __name__ == '__main__':
    main()
