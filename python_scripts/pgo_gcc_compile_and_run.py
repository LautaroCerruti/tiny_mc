import subprocess
import os

# ——— Configuración ———
compiler      = "gcc"
base_flags    = "-std=c11 -Wall -Wextra -g -Ofast -fdisable-tree-cunrolli -ffast-math -ftree-vectorize -march=native -flto"
generate_flag = "-fprofile-generate"
use_flag      = "-fprofile-use"
photons       = 16777216   # cantidad de photones para el macro PHOTONS
runs          = 20          # cuántas veces ejecutar el binario final
sources       = ['tiny_mc.c', 'wtime.c', 'photon.c', 'xoshiro.c']
# ————————————————————

def limpiar():
    """Elimina el ejecutable y todos los .o."""
    print("Limpiando artefactos anteriores...")
    subprocess.run("rm -f headless *.o", shell=True)

def compile_executable(compiler, flags, macro):
    """
    Compila 'headless' usando:
      - compiler: compilador (gcc)
      - flags: cadena con todos los flags (base + profile)
      - macro: valor para -DPHOTONS
    """
    limpiar()
    all_flags = f"{flags} -DPHOTONS={macro}"
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

    objs_str = " ".join(objetos)
    link_cmd = f"{compiler} {all_flags} -o headless {objs_str} -lm"
    print("  Enlazando objetos...")
    if subprocess.run(link_cmd, shell=True).returncode != 0:
        print("Error durante el enlace")
        return False

    print("  → Ejecutable 'headless' listo.")
    return True

def main():
    # 1) Generación de perfil
    print("\n=== Paso 1: Generación de datos de perfil ===")
    if not compile_executable(compiler, f"{base_flags} {generate_flag}", photons):
        return
    print("Ejecutando './headless' para generar datos de perfil...")
    if subprocess.run("./headless", shell=True).returncode != 0:
        print("Error al ejecutar headless (perfil)")
        return

    # 2) Uso de perfil
    print("\n=== Paso 2: Compilación con datos de perfil ===")
    if not compile_executable(compiler, f"{base_flags} {use_flag}", photons):
        return

    # 3) Ejecuciones finales
    print(f"\n=== Paso 3: Ejecutando optimized PGO {runs} veces ===")
    for i in range(1, runs + 1):
        print(f"  Ejecución {i}/{runs} …")
        ret = subprocess.run(
            "./headless -q -o ./results/atom_pgo_gcc.csv",
            shell=True
        ).returncode
        if ret != 0:
            print(f"Error en la ejecución {i}")
            break

if __name__ == "__main__":
    main()
