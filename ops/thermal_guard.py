#!/usr/bin/env python3
"""Gobernador térmico para el nodo 1: mantiene la GPU fuera de la banda de corte.

Base empírica (18 días de muestreo, 3 apagones instrumentados): el nodo se corta
en seco cuando la GPU supera ~85 °C. Los tres cortes cayeron dentro de los 47,7
minutos que la máquina pasó por encima de esa temperatura; fuera de ahí encadenó
13 días sin incidencias. La potencia NO discrimina — sobrevivió picos de 84,5 W
y murió a 45 W estando caliente.

Un cap de reloj fijo funciona, pero cuesta rendimiento todo el tiempo aunque la
máquina esté fría. Esto lo aplica solo cuando hace falta: baja el techo de reloj
al acercarse a la banda y lo devuelve cuando se enfría.

En GB10 `nvidia-smi -pl` no existe; el único control es `-lgc`.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# Temperaturas de decisión. WARN queda 5 °C por debajo del umbral observado de
# 85 °C: hay que reaccionar antes de entrar en la banda, no dentro.
WARN_C = float(os.environ.get("GUARD_WARN_C", "80"))
DANGER_C = float(os.environ.get("GUARD_DANGER_C", "83"))
COOL_C = float(os.environ.get("GUARD_COOL_C", "72"))

# Escalera de techos de reloj, de más rápido a más lento.
STEPS = [int(x) for x in os.environ.get(
    "GUARD_STEPS", "3003,2400,2100,1900,1700,1500").split(",")]
MIN_CLOCK = int(os.environ.get("GUARD_MIN_CLOCK", "210"))

POLL_S = float(os.environ.get("GUARD_POLL_S", "2"))
# Subir de nuevo exige estar frío un rato: sin histéresis el gobernador
# oscilaría entre dos escalones y la GPU no llegaría a estabilizarse nunca.
RECOVER_S = float(os.environ.get("GUARD_RECOVER_S", "120"))

_running = True


def _stop(signum, frame):
    global _running
    _running = False


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}",
          file=sys.stderr, flush=True)


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def apply_cap(mhz):
    """Fija el techo de reloj. El primer escalón equivale a no limitar."""
    if mhz >= STEPS[0]:
        cmd = ["nvidia-smi", "-rgc"]
    else:
        cmd = ["nvidia-smi", "-lgc", f"{MIN_CLOCK},{mhz}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f"ERROR aplicando cap {mhz}: {exc}")
        return False
    if out.returncode != 0:
        # nvidia-smi manda algunos errores (permisos, entre ellos) a stdout,
        # así que mirar solo stderr deja el log en blanco justo cuando falla.
        detail = (out.stderr.strip() or out.stdout.strip() or "sin detalle")
        log(f"ERROR aplicando cap {mhz}: {detail[:200]}")
        return False
    return True


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    idx = 0
    apply_cap(STEPS[idx])
    log(f"gobernador activo | warn={WARN_C}C danger={DANGER_C}C cool={COOL_C}C "
        f"escalones={STEPS}")

    cool_since = None
    while _running:
        t = gpu_temp()
        if t is None:
            time.sleep(POLL_S)
            continue

        if t >= DANGER_C and idx < len(STEPS) - 1:
            # Dos escalones de golpe: en la banda de peligro el margen es de
            # segundos, y quedarse corto significa el apagón.
            idx = min(idx + 2, len(STEPS) - 1)
            apply_cap(STEPS[idx])
            log(f"PELIGRO {t:.0f}C -> techo {STEPS[idx]} MHz")
            cool_since = None
        elif t >= WARN_C and idx < len(STEPS) - 1:
            idx += 1
            apply_cap(STEPS[idx])
            log(f"aviso {t:.0f}C -> techo {STEPS[idx]} MHz")
            cool_since = None
        elif t <= COOL_C and idx > 0:
            if cool_since is None:
                cool_since = time.monotonic()
            elif time.monotonic() - cool_since >= RECOVER_S:
                idx -= 1
                apply_cap(STEPS[idx])
                log(f"frio {t:.0f}C -> techo {STEPS[idx]} MHz")
                cool_since = None
        else:
            cool_since = None

        time.sleep(POLL_S)

    # Al parar se deja la GPU limitada a propósito: soltar el cap durante un
    # apagado o un fallo del servicio devolvería la máquina a la banda de riesgo
    # sin nadie vigilando.
    log("parando; el cap actual se mantiene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
