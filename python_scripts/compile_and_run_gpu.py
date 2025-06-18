import csv
import subprocess
import os
import time

# === Configuración inicial ===
compiler       = "nvcc"
compile_flags  = "-std=c++11 -arch=sm_60 -use_fast_math -O3 -I/usr/local/cuda/include"
sources        = ["tiny_mc_cuda.cu", "photon_cuda.cu"]
executable     = "photon_cuda"
link_flags     = "-L/usr/local/cuda/lib64 -lcudart -lcurand -lm"
dirs           = {"results": "results"}

# CSV con formato: blocks,threads,photons,runs
runs_csv   = "runs_config/runs_gpu_best_config_1060.csv"
# Archivo de salida generado por el ejecutable
output_csv = os.path.join(dirs["results"], "gtx1060_by_size.csv")


def limpiar():
    """Elimina el ejecutable previo."""
    subprocess.run(f"rm -f {executable}", shell=True)


def compile_executable(blocks, threads, photons):
    """
    Compila con nvcc inyectando las macros GPU_BLOCKS, GPU_THREADS y PHOTONS.
    """
    limpiar()
    defs = f"-DGPU_BLOCKS={blocks} -DGPU_THREADS={threads} -DPHOTONS={photons}"
    cmd = (
        f"{compiler} {compile_flags} {defs} "
        f"{' '.join(sources)} -o {executable} {link_flags}"
    )
    print(f"\nCompilando:\n  {cmd}")
    if subprocess.run(cmd, shell=True).returncode != 0:
        print("Error en compilación.")
        return False
    print(f"{executable} compilado correctamente.")
    return True


def ejecutar_gpu(runs, blocks, threads, photons):
    """
    Lanza el ejecutable 'runs' veces.
    """
    for i in range(1, runs + 1):
        print(f"Ejecución {i}/{runs} — blocks={blocks}, threads={threads}, photons={photons}")
        cmd = f"./{executable} -o {output_csv}"
        if subprocess.run(cmd, shell=True).returncode != 0:
            print("Error al ejecutar el binario.")
            break
        time.sleep(1)


def main():
    if not os.path.exists(runs_csv):
        print(f"ERROR: No se encontró {runs_csv}")
        return

    os.makedirs(dirs["results"], exist_ok=True)

    with open(runs_csv, newline="") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            try:
                blocks  = int(fila['blocks'])
                threads = int(fila['threads'])
                photons = int(fila['photons'])
                runs    = int(fila['runs'])
            except Exception as e:
                print(f"Fila inválida {fila}: {e}. Se omite.")
                continue

            print("\n" + "="*60)
            print(f"Config: blocks={blocks}, threads={threads}, photons={photons}, runs={runs}")
            print("="*60)

            if not compile_executable(blocks, threads, photons):
                continue

            ejecutar_gpu(runs, blocks, threads, photons)


if __name__ == '__main__':
    main()
