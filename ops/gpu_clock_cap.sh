#!/usr/bin/env bash
# Limita la frecuencia máxima de la GPU para mantenerla fuera de la banda
# térmica en la que el nodo 1 se apaga.
#
# Por qué la frecuencia y no la potencia: en GB10 el límite de potencia no
# existe (`nvidia-smi -pl` responde "not supported in current scope"), pero
# bloquear el reloj sí está soportado.
#
#   sudo bash gpu_clock_cap.sh on [MHZ]   # por defecto 2100 (sin cap sube a ~2400)
#   sudo bash gpu_clock_cap.sh off        # quita el límite
#   bash gpu_clock_cap.sh status
set -euo pipefail

MAX_MHZ="${2:-${GPU_CAP_MHZ:-2100}}"
MIN_MHZ="${GPU_CAP_MIN_MHZ:-210}"

status() {
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
        --format=csv 2>/dev/null
}

case "${1:-status}" in
    on)
        # -lgc fija la ventana de reloj permitida. El mínimo se deja bajo para
        # que la GPU siga pudiendo bajar en reposo y no consuma de más.
        nvidia-smi -lgc "${MIN_MHZ},${MAX_MHZ}"
        echo "Reloj limitado a ${MIN_MHZ}-${MAX_MHZ} MHz"
        status
        ;;
    off)
        nvidia-smi -rgc
        echo "Límite retirado"
        status
        ;;
    status) status ;;
    *) echo "uso: $0 {on [MHZ]|off|status}" >&2; exit 2 ;;
esac
