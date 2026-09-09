#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Instala las unidades systemd del nodo. Sirve para AMBOS nodos sin editar
#  nada: detecta cuál es por su dirección en la red del cluster y deriva de ahí
#  el peer y el rol. Un directorio, dos nodos — copias editadas a mano acabarían
#  divergiendo justo en lo que importa.
#
#  Idempotente: se puede ejecutar las veces que haga falta.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR=/etc/systemd/system
CLUSTER_IFACE="${CLUSTER_IFACE:-enp1s0f1np1}"

if [[ $EUID -ne 0 ]]; then
    echo "Este script necesita root: sudo bash $0" >&2
    exit 1
fi

# ── Identificar el nodo ──────────────────────────────────────
LOCAL_IP="$(ip -o -4 addr show dev "$CLUSTER_IFACE" 2>/dev/null \
            | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
case "$LOCAL_IP" in
    192.168.200.12) PEER_IP=192.168.200.13; ROLE=head ;;
    192.168.200.13) PEER_IP=192.168.200.12; ROLE=worker ;;
    *)
        echo "ERROR: no reconozco este nodo." >&2
        echo "  $CLUSTER_IFACE tiene '${LOCAL_IP:-nada}'; se esperaba .12 (head) o .13 (worker)." >&2
        echo "  Si la interfaz del cluster es otra: CLUSTER_IFACE=xxx sudo bash $0" >&2
        exit 1 ;;
esac
echo "==> Nodo detectado: $LOCAL_IP ($ROLE) | peer $PEER_IP vía $CLUSTER_IFACE"

# ── Muestreador de forense ───────────────────────────────────
echo "==> Instalando node-forensics"
install -m 644 "$OPS_DIR/node-forensics.service" "$UNIT_DIR/"
# El peer va en un drop-in en vez de en la unidad: así el fichero del repo es
# idéntico en los dos nodos y lo que cambia queda fuera, generado aquí.
install -d -m 755 "$UNIT_DIR/node-forensics.service.d"
cat > "$UNIT_DIR/node-forensics.service.d/10-peer.conf" <<CONF
# Generado por install.sh — no editar a mano.
[Service]
Environment=FORENSICS_PEER=$PEER_IP
Environment=FORENSICS_PEER_IFACE=$CLUSTER_IFACE
CONF

# ── Protección térmica y experimentos ────────────────────────
echo "==> Instalando thermal-guard / gpu-clock-cap / idle-experiment"
install -m 644 "$OPS_DIR/thermal-guard.service"   "$UNIT_DIR/"
install -m 644 "$OPS_DIR/gpu-clock-cap.service"   "$UNIT_DIR/"
install -m 644 "$OPS_DIR/idle-experiment.service" "$UNIT_DIR/"

# ── Cluster vLLM: solo en el head ────────────────────────────
# El head es quien lanza ambos contenedores y quien vigila al worker por SSH.
# Instalar el timer en el worker lo pondría a pelearse con su propio contenedor.
CLUSTER_DIR="$OPS_DIR/../ray-cluster"
if [[ "$ROLE" == head && -d "$CLUSTER_DIR" ]]; then
    echo "==> Instalando vllm-cluster (service + timer) — solo head"
    install -m 644 "$CLUSTER_DIR/vllm-cluster.service" "$UNIT_DIR/"
    install -m 644 "$CLUSTER_DIR/vllm-cluster.timer"   "$UNIT_DIR/"
else
    echo "==> vllm-cluster: omitido (rol=$ROLE)"
fi

systemctl daemon-reload

echo "==> Habilitando"
# `enable --now` no reinicia un servicio que ya corría, así que un install.sh
# repetido tras editar el código dejaba en marcha la versión vieja sin avisar.
systemctl enable node-forensics.service
systemctl restart node-forensics.service
if [[ "$ROLE" == head && -d "$CLUSTER_DIR" ]]; then
    systemctl enable vllm-cluster.service
    systemctl enable --now vllm-cluster.timer
fi

echo
echo "==> Estado"
systemctl --no-pager --lines=3 status node-forensics.service || true
echo
echo "Listo. El muestreo escribe en $OPS_DIR/logs/"
echo "Tras un corte:  python3 $OPS_DIR/crash_report.py"
echo
echo "Protección (no se activa sola — arráncala a propósito):"
echo "  gobernador térmico (recomendado):"
echo "    systemctl enable --now thermal-guard.service"
echo "  cap de reloj fijo (alternativa excluyente):"
echo "    systemctl enable --now gpu-clock-cap.service"
