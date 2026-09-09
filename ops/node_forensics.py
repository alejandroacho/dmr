#!/usr/bin/env python3
"""Muestreador de forense de hardware para los nodos GX10.

Escribe una muestra por intervalo a un CSV diario, con fsync en cada línea.
El fsync es el punto entero de este script: el fallo que perseguimos es un
corte de corriente duro, así que lo último que se escribió tiene que estar ya
en el disco y no en el page cache, o perdemos justo los segundos que importan.

Cada sesión abre con una línea `#BOOT_START` y, si el proceso recibe SIGTERM
(apagado ordenado), cierra con `#CLEAN_STOP`. Una sesión sin `#CLEAN_STOP` es
exactamente la firma de una muerte sin avisar, y es lo que busca crash_report.py.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta

LOG_DIR = os.environ.get("FORENSICS_LOG_DIR", "/home/alejandroacho/Server/ops/logs")
INTERVAL = float(os.environ.get("FORENSICS_INTERVAL", "15"))
PEER = os.environ.get("FORENSICS_PEER", "192.168.200.13")
PEER_IFACE = os.environ.get("FORENSICS_PEER_IFACE", "enp1s0f1np1")
RETENTION_DAYS = int(os.environ.get("FORENSICS_RETENTION_DAYS", "21"))

# El root port cuya tormenta AER nos trajo hasta aquí. Se le sacan los
# contadores desglosados, no solo el total.
WATCHED_PORT = os.environ.get("FORENSICS_WATCHED_PORT", "0000:00:00.0")
WATCHED_AER_FIELDS = ("RxErr", "BadTLP", "BadDLLP", "Timeout", "TOTAL_ERR_COR")

PCI_DEVICES = "/sys/bus/pci/devices"

_running = True


def _stop(signum, frame):
    global _running
    _running = False


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def read_int(path, scale=1.0):
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw) / scale
    except ValueError:
        return None


def read_aer(path):
    """Parsea un fichero aer_dev_* ('RxErr 12\\nBadTLP 0\\n...') a dict."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        out[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def discover_thermal_zones():
    zones = []
    base = "/sys/class/thermal"
    try:
        names = sorted(
            (n for n in os.listdir(base) if n.startswith("thermal_zone")),
            key=lambda n: int(n[len("thermal_zone"):]),
        )
    except OSError:
        return zones
    for name in names:
        zones.append((f"tz{name[len('thermal_zone'):]}_c", f"{base}/{name}/temp"))
    return zones


def discover_hwmon():
    """Mapea sensores hwmon a nombres de columna estables.

    El índice hwmonN puede bailar entre arranques, así que la columna se nombra
    por la dirección PCI del dispositivo cuando la hay — imprescindible con las
    cuatro mlx5, que son puertos idénticos y sin eso no se distinguen.
    """
    sensors = []
    base = "/sys/class/hwmon"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return sensors
    for entry in entries:
        hpath = f"{base}/{entry}"
        chip = read_text(f"{hpath}/name") or entry
        # acpitz ya sale por /sys/class/thermal como tzN_c; duplicarlo aquí solo
        # ensancha el CSV con las mismas siete lecturas.
        if chip == "acpitz":
            continue
        pci = None
        try:
            target = os.path.realpath(f"{hpath}/device")
            for part in reversed(target.split("/")):
                if len(part) == 12 and part.count(":") == 2 and part.count(".") == 1:
                    pci = part
                    break
        except OSError:
            pass
        try:
            inputs = sorted(n for n in os.listdir(hpath) if n.startswith("temp") and n.endswith("_input"))
        except OSError:
            continue
        for idx, inp in enumerate(inputs):
            tag = chip if pci is None else f"{chip}_{pci.replace(':', '_').replace('.', '_')}"
            suffix = "" if len(inputs) == 1 else f"_{idx}"
            sensors.append((f"{tag}{suffix}_c", f"{hpath}/{inp}"))
            # Las mlx5 exponen un máximo latcheado por hardware. Vale la pena
            # porque un pico térmico más corto que el intervalo de muestreo se
            # nos escaparía, y aquí queda registrado igualmente.
            highest = f"{hpath}/{inp[: -len('_input')]}_highest"
            if os.path.exists(highest):
                sensors.append((f"{tag}{suffix}_peak_c", highest))
    return sensors


def discover_aer_devices():
    devs = []
    try:
        for name in sorted(os.listdir(PCI_DEVICES)):
            if os.path.exists(f"{PCI_DEVICES}/{name}/aer_dev_correctable"):
                devs.append(name)
    except OSError:
        pass
    return devs


def gpu_sample():
    """Consulta nvidia-smi con timeout.

    El timeout no es paranoia: si la GPU se queda colgada en el bus, nvidia-smi
    se cuelga con ella, y ese cuelgue es en sí mismo el dato que queremos ver.
    """
    fields = (
        "temperature.gpu,power.draw,utilization.gpu,clocks.sm,"
        "clocks_throttle_reasons.active"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        return ["TIMEOUT"] * 5
    except OSError:
        return [""] * 5
    if out.returncode != 0:
        return ["ERR"] * 5
    line = out.stdout.strip().splitlines()
    if not line:
        return [""] * 5
    vals = [v.strip() for v in line[0].split(",")]
    vals = [("" if v in ("[N/A]", "[Not Supported]") else v) for v in vals]
    while len(vals) < 5:
        vals.append("")
    return vals[:5]


class PowerSampler:
    """Lee power.draw a 200 ms en un hilo aparte.

    Una muestra cada 15 s no ve un transitorio de corriente: entre dos lecturas
    la GPU sube 30 W y vuelve sin dejar rastro. Este hilo mantiene el mínimo y
    el máximo observados desde la última muestra, así que el transitorio queda
    registrado aunque dure una fracción de segundo.
    """

    def __init__(self, period_ms=200):
        self.period_ms = period_ms
        self.lock = threading.Lock()
        self.lo = None
        self.hi = None
        self.alive = True

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            try:
                proc = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits",
                     f"--loop-ms={self.period_ms}"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
            except OSError:
                return
            for line in proc.stdout:
                if not self.alive:
                    break
                try:
                    v = float(line.strip())
                except ValueError:
                    continue
                with self.lock:
                    if self.lo is None or v < self.lo:
                        self.lo = v
                    if self.hi is None or v > self.hi:
                        self.hi = v
            # nvidia-smi murió (GPU colgada, driver recargado): se reintenta,
            # pero sin apretar, para no convertir el fallo en un bucle ocupado.
            try:
                proc.kill()
            except Exception:
                pass
            time.sleep(5)

    def take(self):
        with self.lock:
            lo, hi = self.lo, self.hi
            self.lo = self.hi = None
        return ("" if lo is None else f"{lo:.2f}",
                "" if hi is None else f"{hi:.2f}")


def read_cpuidle():
    """Entradas acumuladas a cada estado de reposo, sumadas sobre todas las CPU.

    Si el nodo se cae al salir de un reposo profundo, el ritmo de entradas en
    los segundos previos es la única huella que quedaría.
    """
    base = "/sys/devices/system/cpu"
    totals = {}
    try:
        cpus = [c for c in os.listdir(base) if c.startswith("cpu") and c[3:].isdigit()]
    except OSError:
        return totals
    for cpu in cpus:
        idle = f"{base}/{cpu}/cpuidle"
        try:
            states = os.listdir(idle)
        except OSError:
            continue
        for st in states:
            if not st.startswith("state"):
                continue
            val = read_int(f"{idle}/{st}/usage")
            if val is not None:
                totals[st] = totals.get(st, 0) + int(val)
    return totals


def deep_idle_disabled():
    """¿Están desactivados los estados de reposo profundo?

    Se registra en cada muestra, no una sola vez al arrancar: así el
    post-mortem puede decir con certeza si el experimento estaba activo cuando
    el nodo se cayó, en vez de tener que fiarse de la memoria de nadie.
    """
    base = "/sys/devices/system/cpu/cpu0/cpuidle"
    flags = []
    for idx in (2, 3):
        val = read_text(f"{base}/state{idx}/disable")
        if val is not None:
            flags.append(val.strip())
    if not flags:
        return ""
    return "1" if all(f == "1" for f in flags) else "0"


def ping_peer():
    try:
        out = subprocess.run(
            ["ping", "-c", "1", "-W", "1", "-n", PEER],
            capture_output=True, text=True, timeout=4,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if out.returncode != 0:
        return "LOST"
    for token in out.stdout.split():
        if token.startswith("time="):
            return token[5:]
    return ""


class CpuMeter:
    """%CPU agregado a partir de deltas de /proc/stat."""

    def __init__(self):
        self.prev = None

    def read(self):
        line = read_text("/proc/stat")
        if not line:
            return ""
        parts = line.splitlines()[0].split()[1:]
        try:
            vals = [int(v) for v in parts]
        except ValueError:
            return ""
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        prev, self.prev = self.prev, (idle, total)
        if prev is None:
            return ""
        d_idle = idle - prev[0]
        d_total = total - prev[1]
        if d_total <= 0:
            return ""
        return f"{100.0 * (1.0 - d_idle / d_total):.1f}"


def uptime_s():
    raw = read_text("/proc/uptime")
    if not raw:
        return ""
    try:
        return f"{float(raw.split()[0]):.0f}"
    except (ValueError, IndexError):
        return ""


def load1():
    raw = read_text("/proc/loadavg")
    if not raw:
        return ""
    try:
        return raw.split()[0]
    except IndexError:
        return ""


def mem_used_pct():
    total = avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
                if total is not None and avail is not None:
                    break
    except (OSError, ValueError):
        return ""
    if not total:
        return ""
    return f"{100.0 * (total - avail) / total:.1f}"


def fmt(value, digits=1):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


class Writer:
    """CSV diario con fsync por línea y rotación por fecha."""

    def __init__(self, header):
        self.header = header
        self.day = None
        self.fh = None
        # Cabecera de la sesión en curso, para poder repetirla si la rotación
        # de medianoche parte la sesión en dos ficheros.
        self.session_line = None
        os.makedirs(LOG_DIR, exist_ok=True)

    def _path(self, day):
        return os.path.join(LOG_DIR, f"forensics-{day.isoformat()}.csv")

    def _roll(self, day):
        if self.fh is not None:
            # Cierre explícito por cambio de fichero. Sin este marcador el
            # post-mortem ve un fichero que acaba sin #CLEAN_STOP y lo
            # denuncia como una muerte, cuando solo ha pasado la medianoche.
            self._raw(f"#ROLLOVER ts={datetime.now().isoformat(timespec='seconds')}")
            self.fh.close()
        path = self._path(day)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        # Si el conjunto de columnas cambió (se añadió una métrica y se
        # reinició el servicio), hay que escribir cabecera nueva aunque el
        # fichero ya exista. Sin esto quedan filas de N campos bajo una
        # cabecera de M, y el post-mortem lee todas las columnas corridas.
        need_header = fresh or not self._header_matches(path)
        self.fh = open(path, "a", buffering=1)
        if need_header:
            self.fh.write(",".join(self.header) + "\n")
        self.day = day
        self._prune()
        if self.session_line is not None:
            self._raw(self.session_line + " resumed=1")

    def _raw(self, text):
        """Escribe sin comprobar la rotación (evita recursión desde _roll)."""
        self.fh.write(text + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def start_session(self, line):
        self.write(line)
        self.session_line = line

    def _header_matches(self, path):
        """¿La última cabecera escrita en el fichero es la que usamos ahora?"""
        want = ",".join(self.header)
        last = None
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("ts,"):
                        last = line.rstrip("\n")
        except OSError:
            return False
        return last == want

    def _prune(self):
        cutoff = date.today() - timedelta(days=RETENTION_DAYS)
        try:
            for name in os.listdir(LOG_DIR):
                if not (name.startswith("forensics-") and name.endswith(".csv")):
                    continue
                try:
                    stamp = date.fromisoformat(name[len("forensics-"):-len(".csv")])
                except ValueError:
                    continue
                if stamp < cutoff:
                    os.remove(os.path.join(LOG_DIR, name))
        except OSError:
            pass

    def write(self, text):
        today = date.today()
        if self.fh is None or today != self.day:
            self._roll(today)
        self._raw(text)

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def previous_session_verdict():
    """Mira el log más reciente y dice si la sesión anterior murió sin avisar."""
    try:
        files = sorted(
            n for n in os.listdir(LOG_DIR)
            if n.startswith("forensics-") and n.endswith(".csv")
        )
    except OSError:
        return None
    for name in reversed(files):
        last_start = last_stop = None
        last_sample = None
        try:
            with open(os.path.join(LOG_DIR, name)) as f:
                for line in f:
                    if line.startswith("#BOOT_START"):
                        last_start = line.strip()
                        last_stop = None
                    elif line.startswith("#CLEAN_STOP"):
                        last_stop = line.strip()
                    elif line and not line.startswith(("#", "ts,")):
                        last_sample = line.strip()
        except OSError:
            continue
        if last_start is None:
            continue
        if last_stop is None:
            return (name, last_start, last_sample)
        return None
    return None


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    zones = discover_thermal_zones()
    hwmon = discover_hwmon()
    aer_devs = discover_aer_devices()
    cpu = CpuMeter()
    idle_keys = sorted(read_cpuidle().keys())
    power = PowerSampler()
    power.start()

    header = ["ts", "epoch", "uptime_s", "load1", "cpu_pct", "mem_used_pct"]
    header += [name for name, _ in zones]
    header += [name for name, _ in hwmon]
    header += ["gpu_c", "gpu_w", "gpu_util", "gpu_sm_mhz", "gpu_throttle"]
    header += ["gpu_w_min", "gpu_w_max"]
    header += [f"{k}_entries" for k in idle_keys]
    header += ["deep_idle_off"]
    header += ["aer_cor_all", "aer_nonfatal_all", "aer_fatal_all"]
    header += [f"rp_{f.lower()}" for f in WATCHED_AER_FIELDS]
    header += ["peer_link", "peer_ping_ms"]

    writer = Writer(header)

    verdict = previous_session_verdict()
    if verdict is not None:
        name, start, sample = verdict
        print(
            f"[forensics] LA SESION ANTERIOR NO CERRO LIMPIAMENTE ({name})\n"
            f"[forensics]   inicio : {start}\n"
            f"[forensics]   ultima muestra: {sample}",
            file=sys.stderr, flush=True,
        )

    boot_id = read_text("/proc/sys/kernel/random/boot_id") or "?"
    writer.start_session(
        f"#BOOT_START ts={datetime.now().isoformat(timespec='seconds')} "
        f"boot_id={boot_id} interval={INTERVAL}s pid={os.getpid()}"
    )
    cpu.read()  # ceba el delta para que la primera muestra ya traiga %CPU

    next_tick = time.monotonic()
    while _running:
        now = datetime.now()
        row = [
            now.isoformat(timespec="seconds"),
            f"{time.time():.0f}",
            uptime_s(),
            load1(),
            cpu.read(),
            mem_used_pct(),
        ]
        row += [fmt(read_int(path, 1000.0)) for _, path in zones]
        row += [fmt(read_int(path, 1000.0)) for _, path in hwmon]
        row += gpu_sample()
        row += list(power.take())
        idle_now = read_cpuidle()
        row += [str(idle_now.get(k, "")) for k in idle_keys]
        row.append(deep_idle_disabled())

        cor = nonfatal = fatal = 0
        for dev in aer_devs:
            cor += read_aer(f"{PCI_DEVICES}/{dev}/aer_dev_correctable").get("TOTAL_ERR_COR", 0)
            nf = read_aer(f"{PCI_DEVICES}/{dev}/aer_dev_nonfatal")
            fa = read_aer(f"{PCI_DEVICES}/{dev}/aer_dev_fatal")
            nonfatal += sum(v for k, v in nf.items() if k.startswith("TOTAL_ERR"))
            fatal += sum(v for k, v in fa.items() if k.startswith("TOTAL_ERR"))
        row += [str(cor), str(nonfatal), str(fatal)]

        watched = read_aer(f"{PCI_DEVICES}/{WATCHED_PORT}/aer_dev_correctable")
        row += [str(watched.get(f, "")) for f in WATCHED_AER_FIELDS]

        row.append(read_text(f"/sys/class/net/{PEER_IFACE}/carrier") or "")
        row.append(ping_peer())

        writer.write(",".join(row))

        next_tick += INTERVAL
        delay = next_tick - time.monotonic()
        if delay < 0:
            next_tick = time.monotonic()
            delay = 0
        # Sueño troceado para reaccionar a SIGTERM sin esperar el intervalo entero.
        while delay > 0 and _running:
            time.sleep(min(delay, 0.5))
            delay = next_tick - time.monotonic()

    power.alive = False
    writer.write(f"#CLEAN_STOP ts={datetime.now().isoformat(timespec='seconds')}")
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
