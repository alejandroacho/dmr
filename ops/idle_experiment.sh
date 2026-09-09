#!/usr/bin/env bash
# Activa o desactiva los estados de reposo profundo de la CPU (LPI-2 / LPI-3).
#
# Hipótesis que pone a prueba: el nodo se apaga al entrar o salir de un estado
# de reposo profundo del SoC. Encaja con lo observado — el corte no deja rastro
# ni en el kernel ni en el firmware, y ocurre tanto con carga como sin ella.
#
# Es reversible y no necesita reinicio. El coste de desactivarlos es unos pocos
# vatios más en reposo; nada más.
#
#   sudo bash idle_experiment.sh off     # desactiva LPI-2 y LPI-3
#   sudo bash idle_experiment.sh on      # los vuelve a activar
#   bash idle_experiment.sh status       # estado y contadores actuales
set -euo pipefail

DEEP_STATES="${DEEP_STATES:-2 3}"
CPUS=/sys/devices/system/cpu

status() {
    printf '%-8s %-8s %-12s %s\n' ESTADO NOMBRE DESACTIVADO ENTRADAS
    for s in "$CPUS"/cpu0/cpuidle/state*; do
        [ -d "$s" ] || continue
        n=$(basename "$s")
        total=0
        for c in "$CPUS"/cpu*/cpuidle/"$n"/usage; do
            [ -r "$c" ] && total=$((total + $(cat "$c")))
        done
        printf '%-8s %-8s %-12s %s\n' "$n" "$(cat "$s/name")" \
            "$(cat "$s/disable" 2>/dev/null || echo '?')" "$total"
    done
}

set_disable() {
    local value=$1
    for idx in $DEEP_STATES; do
        local n=0
        for f in "$CPUS"/cpu*/cpuidle/state"$idx"/disable; do
            [ -w "$f" ] || continue
            echo "$value" > "$f"
            n=$((n + 1))
        done
        echo "state$idx: disable=$value aplicado a $n CPU(s)"
    done
}

case "${1:-status}" in
    off) set_disable 1; echo; status ;;
    on)  set_disable 0; echo; status ;;
    status) status ;;
    *) echo "uso: $0 {off|on|status}" >&2; exit 2 ;;
esac
