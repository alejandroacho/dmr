#!/usr/bin/env python3
"""Post-mortem de los CSV que deja node_forensics.py.

Localiza las sesiones que no cerraron con `#CLEAN_STOP` — es decir, las veces
que el nodo se fue sin avisar — e imprime los minutos previos: temperaturas,
consumo de GPU, contadores AER y estado del enlace con el otro nodo.

Uso:
    python3 crash_report.py              # la última muerte sin avisar
    python3 crash_report.py --all        # todas las que haya en el histórico
    python3 crash_report.py --window 40  # nº de muestras finales a mostrar
"""

import argparse
import os
import sys

LOG_DIR = os.environ.get("FORENSICS_LOG_DIR", "/home/alejandroacho/Server/ops/logs")

# Columnas que se imprimen en la tabla final. El resto siguen en el CSV para
# quien quiera hilar más fino, pero estas son las que cuentan la historia.
FOCUS = [
    "ts", "cpu_pct", "gpu_c", "gpu_w", "gpu_w_max", "gpu_throttle",
    "rp_rxerr", "rp_badtlp", "aer_cor_all", "peer_link", "peer_ping_ms",
    # Deja constancia de si el experimento de reposo profundo estaba activo
    # cuando el nodo murió, que es lo que decide si la hipótesis queda refutada.
    "deep_idle_off",
]


def load_sessions(path):
    """Trocea un CSV en sesiones: (inicio, muestras, cerrada_limpiamente)."""
    sessions = []
    header = None
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("ts,"):
                header = line.split(",")
                # Una cabecera nueva a media sesión significa que cambió el
                # conjunto de columnas. Las filas de antes y de después NO son
                # comparables por índice, así que abren segmento aparte.
                if current is not None:
                    current["segments"].append((header, []))
            elif line.startswith("#BOOT_START"):
                current = {"start": line, "segments": [(header, [])],
                           "clean": False, "file": path}
                sessions.append(current)
            elif line.startswith("#CLEAN_STOP"):
                if current is not None:
                    current["clean"] = True
                    current["stop"] = line
            elif line.startswith("#ROLLOVER"):
                if current is not None:
                    current["rolled"] = True
            else:
                if current is None:
                    # Muestras antes de cualquier marcador: es la cola de una
                    # sesión que cruzó la medianoche. Los logs anteriores al
                    # marcador #ROLLOVER no lo señalan, así que se deduce.
                    current = {
                        "start": "#CONTINUACION (sin marcador)",
                        "segments": [(header, [])],
                        "clean": False, "file": path, "implicit": True,
                    }
                    sessions.append(current)
                fields = line.split(",")
                # Guarda contra ficheros históricos escritos por una versión
                # que no reescribía la cabecera al cambiar de columnas: si el
                # ancho no cuadra, los índices no significan nada y la fila se
                # descarta en vez de producir lecturas cruzadas silenciosas.
                seg_header = current["segments"][-1][0]
                if seg_header is None or len(fields) == len(seg_header):
                    current["segments"][-1][1].append(fields)
                else:
                    current["skipped"] = current.get("skipped", 0) + 1
    return sessions


def all_sessions():
    try:
        names = sorted(
            n for n in os.listdir(LOG_DIR)
            if n.startswith("forensics-") and n.endswith(".csv")
        )
    except OSError:
        return []
    out = []
    for name in names:
        for sess in load_sessions(os.path.join(LOG_DIR, name)):
            cont = sess.get("implicit") or "resumed=1" in sess["start"]
            if out and cont:
                # Misma sesión partida por el cambio de fichero: se pega a la
                # anterior en vez de contarse como una muerte más.
                prev = out[-1]
                prev["segments"].extend(sess["segments"])
                prev["clean"] = sess["clean"]
                prev["file"] = sess["file"]
                continue
            out.append(sess)
    return out


# Bits de clocks_throttle_reasons. Los tres últimos son los que importan aquí:
# significan que el hardware frenó la GPU por temperatura o por corriente, y
# ver uno de esos justo antes de un corte apuntaría directamente a alimentación.
THROTTLE_BITS = [
    (0x0001, "gpu_idle"),
    (0x0002, "app_clocks"),
    (0x0004, "sw_power_cap"),
    (0x0008, "HW_SLOWDOWN"),
    (0x0010, "sync_boost"),
    (0x0020, "sw_thermal"),
    (0x0040, "HW_THERMAL"),
    (0x0080, "HW_POWER_BRAKE"),
]
SMOKING_GUNS = 0x0008 | 0x0040 | 0x0080


def decode_throttle(value):
    try:
        bits = int(value, 16)
    except (TypeError, ValueError):
        return None
    if bits == 0:
        return []
    return [name for mask, name in THROTTLE_BITS if bits & mask]


def col(header, row, name):
    try:
        return row[header.index(name)]
    except (ValueError, IndexError):
        return ""


def parse_start(line):
    """Saca los pares clave=valor de una línea #BOOT_START."""
    out = {}
    for token in line.split()[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def session_is_live(session):
    """¿Esta sesión abierta es el muestreador que corre ahora mismo?

    Sin esto, la sesión en curso se confundiría con una muerte. Un boot_id
    distinto al actual significa que la máquina rebotó desde entonces, así que
    esa sesión murió sí o sí; con el mismo boot_id, basta ver si el PID sigue vivo.
    """
    meta = parse_start(session["start"])
    current_boot = ""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            current_boot = f.read().strip()
    except OSError:
        pass
    if meta.get("boot_id") != current_boot:
        return False
    pid = meta.get("pid")
    if not pid or not pid.isdigit():
        return False
    return os.path.exists(f"/proc/{pid}")


def peak(header, rows, name, cast=float):
    vals = []
    for r in rows:
        v = col(header, r, name)
        try:
            vals.append(cast(v))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else None


def report(session, window):
    # Cada segmento trae su propia cabecera. La tabla final usa la del último
    # (la vigente al morir); los picos recorren todos, cada uno con la suya.
    segments = [(h, r) for h, r in session["segments"] if h and r]
    if not segments:
        print("=" * 78)
        print(f"MUERTE SIN AVISO  ({os.path.basename(session['file'])})")
        print("  Sin muestras en esta sesión.")
        return
    header, rows = segments[-1]
    total = sum(len(r) for _, r in segments)
    print("=" * 78)
    print(f"MUERTE SIN AVISO  ({os.path.basename(session['file'])})")
    print(f"  {session['start']}")
    if not rows:
        print("  Sin muestras en esta sesión.")
        return
    print(f"  Última muestra : {col(header, rows[-1], 'ts')}")
    print(f"  Muestras       : {total}")
    if session.get("skipped"):
        print(f"  Descartadas    : {session['skipped']} (ancho != cabecera; "
              f"log escrito por una versión anterior)")
    print()

    # Picos de la sesión: contexto para saber si venía calentándose o murió fría.
    print("  Picos de la sesión:")
    for label, name in (
        ("temp NIC mlx5 (máx)", None),
        ("temp GPU", "gpu_c"),
        ("consumo GPU (W)", "gpu_w"),
        ("CPU %", "cpu_pct"),
    ):
        vals = []
        for hh, rr in segments:
            if name is None:
                for c in (c for c in hh if c.startswith("mlx5") and "peak" not in c):
                    p = peak(hh, rr, c)
                    if p is not None:
                        vals.append(p)
            else:
                p = peak(hh, rr, name)
                if p is not None:
                    vals.append(p)
        value = max(vals) if vals else None
        print(f"    {label:<22}: {'' if value is None else round(value, 1)}")

    first_aer = col(header, rows[0], "aer_cor_all")
    last_aer = col(header, rows[-1], "aer_cor_all")
    try:
        print(f"    {'AER corregibles':<22}: {int(last_aer) - int(first_aer)} durante la sesión")
    except (TypeError, ValueError):
        pass

    # Frenadas por hardware: si la GPU fue limitada por temperatura o por
    # corriente antes del corte, eso es una pista de alimentación, no de software.
    hw_events = []
    for hh, rr in segments:
      for r in rr:
        raw = col(hh, r, "gpu_throttle")
        try:
            bits = int(raw, 16)
        except (TypeError, ValueError):
            continue
        if bits & SMOKING_GUNS:
            hw_events.append((col(hh, r, "ts"), decode_throttle(raw)))
    if hw_events:
        print()
        print(f"  *** {len(hw_events)} frenada(s) por HARDWARE durante la sesión ***")
        for ts, names in hw_events[-5:]:
            print(f"    {ts}  {'+'.join(names)}")
    print()

    tail = rows[-window:]
    present = [c for c in FOCUS if c in header]
    widths = {
        c: max([len(c), 10] + [len(col(header, r, c)) for r in tail])
        for c in present
    }
    print(f"  Últimas {len(tail)} muestras:")
    print("    " + "  ".join(c.ljust(widths[c]) for c in present))
    print("    " + "  ".join("-" * widths[c] for c in present))
    for r in tail:
        print("    " + "  ".join(col(header, r, c).ljust(widths[c]) for c in present))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="mostrar todas las muertes, no solo la última")
    ap.add_argument("--window", type=int, default=25, help="nº de muestras finales a imprimir")
    args = ap.parse_args()

    sessions = all_sessions()
    if not sessions:
        print(f"No hay logs de forense en {LOG_DIR}.", file=sys.stderr)
        return 1

    # Una sesión sin #CLEAN_STOP es una muerte, salvo que sea la que corre ahora.
    dirty = [s for s in sessions if not s["clean"] and not session_is_live(s)]

    if not dirty:
        print(f"{len(sessions)} sesión(es) registrada(s), ninguna murió sin avisar.")
        print("Ningún corte inesperado desde que el logger está activo.")
        return 0

    for s in (dirty if args.all else dirty[-1:]):
        report(s, args.window)
    if not args.all and len(dirty) > 1:
        print(f"(hay {len(dirty)} muertes registradas en total — usa --all para verlas todas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
