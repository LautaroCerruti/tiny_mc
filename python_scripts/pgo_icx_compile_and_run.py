import subprocess
import os

# ——— Configuración ———
compiler       = "icx"
base_flags     = "-std=c11 -Wall -Wextra -g -O3 -ffast-math -ftree-vectorize -xHost -ipo -fno-unroll-loops"
generate_flag  = "-fprofile-sample-generate"
use_flag       = "-fprofile-sample-use=headless.freq.prof -mllvm -unpredictable-hints-file=headless.misp.prof"
profgen_tool  = "/opt/intel/oneapi/compiler/latest/bin/compiler/llvm-profgen"
photons        = 16777216   # valor para -DPHOTONS
runs           = 20          # cuántas veces ejecutar el binario optimizado
sources        = ['tiny_mc.c', 'wtime.c', 'photon.c', 'xoshiro.c']
# ————————————————————

def limpiar():
    """Elimina ejecutable, .o y archivos de perfil."""
    print("Limpiando artefactos anteriores…")
    subprocess.run("rm -f headless *.o", shell=True)

def compile_executable(flags, macro):
    """
    Compila 'headless' usando:
      - flags: todos los flags (base + profile)
      - macro: valor para -DPHOTONS
    """
    limpiar()
    all_flags = f"{base_flags} {flags} -DPHOTONS={macro}"
    print(f"\nCompilando con:\n  {compiler} {all_flags}")

    objetos = []
    for src in sources:
        obj = os.path.splitext(src)[0] + '.o'
        print(f"  Compilando {src} → {obj}")
        cmd = f"{compiler} {all_flags} -c {src} -o {obj}"
        if subprocess.run(cmd, shell=True).returncode != 0:
            print(f"Error compilando {src}")
            return False
        objetos.append(obj)

    objs = " ".join(objetos)
    link_cmd = f"{compiler} {all_flags} -o headless {objs} -lm"
    print("  Enlazando objetos…")
    if subprocess.run(link_cmd, shell=True).returncode != 0:
        print("Error al enlazar")
        return False

    print("  → Ejecutable 'headless' listo.")
    return True

def run_profgen():
    cmd = f"{profgen_tool} --perfdata headless.perf.data --binary headless --output headless.freq.prof --sample-period 1000003 --perf-event br_inst_retired.near_taken:uppp"
    if subprocess.run(cmd, shell=True).returncode != 0:
        print("Error al generar perfil.")
        return False
    cmd = f"{profgen_tool} --perfdata headless.perf.data --binary headless --output headless.misp.prof --sample-period 1000003 --perf-event mr_misp_retired.all_branches:upp --leading-ip-only"
    if subprocess.run(cmd, shell=True).returncode != 0:
        print("Error al generar perfil.")
        return False
    return True

def main():
    # 1) Generar perfil
    print("\n=== Paso 1: Compilación PGO generate ===")
    if not compile_executable(generate_flag, photons):
        return
    print("Ejecutando './headless' para generar profiling")
    if subprocess.run("perf record -o headless.perf.data -b -c 1000003 -e br_inst_retired.near_taken:uppp,br_misp_retired.all_branches:upp -- ./headless", shell=True).returncode != 0:
        print("Error al ejecutar headless (generación de perfil)")
        return

    # 2) Fusionar .profraw → .profdata
    if not run_profgen():
        return

    # 3) Compilar con perfil
    print("\n=== Paso 2: Compilación PGO use ===")
    if not compile_executable(use_flag, photons):
        return

    # 4) Ejecutar binario optimizado varias veces
    print(f"\n=== Paso 3: Ejecutando headless optimizado {runs} veces ===")
    for i in range(1, runs+1):
        print(f"  Ejecución {i}/{runs}…")
        ret = subprocess.run(
            "./headless -q -o ./results/notebook_pgo_icx.csv",
            shell=True
        ).returncode
        if ret != 0:
            print(f"Error en ejecución {i}")
            break

if __name__ == "__main__":
    main()
