# Stacked Sparks — Configuracion de dos DGX Spark en cluster

Guia para conectar dos sistemas NVIDIA DGX Spark via QSFP 200GbE, habilitar NCCL para comunicacion GPU-to-GPU entre nodos, y ejecutar inferencia distribuida con vLLM.

| Nodo | Rol | IP (QSFP) | Contenedor |
|---|---|---|---|
| Node 1 | Head + gateway | 192.168.200.12 | `ray-node-head` |
| Node 2 | Worker | 192.168.200.13 | `ray-node-worker` |

---

## 1. Conectar dos Sparks

> Documentacion oficial: [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)

### 1.1 Verificar que ambos sistemas usan el mismo usuario

```bash
whoami
```

Si los nombres de usuario no coinciden, crear uno comun en ambos nodos:

```bash
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia
sudo passwd nvidia
su - nvidia
```

### 1.2 Conexion fisica

Conectar el cable QSFP entre ambos DGX Spark. Verificar que al menos una interfaz aparece como `Up`:

```bash
ibdev2netdev
```

Salida esperada:

```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

> **Nota:** Cada puerto fisico tiene dos nombres (ej. `enp1s0f1np1` y `enP2p1s0f1np1`). Usar solo los que empiezan con `enp1...`. Si ninguna interfaz muestra `Up`, verificar el cable y reiniciar ambos sistemas.

### 1.3 Configurar las interfaces de red

Hay tres opciones. La Opcion 1 y 2 son mutuamente excluyentes.

> **Nota:** Se obtiene ancho de banda completo con un solo cable QSFP. Con dos cables, las cuatro interfaces deben tener IP asignada.

#### Opcion A: IP automatica via netplan (solo con 1 cable QSFP)

Ejecutar en **ambos nodos**:

```bash
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    enp1s0f1np1:
      link-local: [ ipv4 ]
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

#### Opcion B: IP manual via netplan (soporta 1 o 2 cables)

**Node 1:**

```bash
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enp1s0f1np1:
      addresses:
        - 192.168.200.12/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.100.14/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.200.16/24
      dhcp4: no
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

**Node 2:**

```bash
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enp1s0f1np1:
      addresses:
        - 192.168.200.13/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.100.15/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.200.17/24
      dhcp4: no
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

#### Opcion C: IP manual por linea de comandos (no persiste tras reboot)

Identificar la interfaz activa:

```bash
ibdev2netdev
```

**Node 1:**

```bash
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

**Node 2:**

```bash
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

Verificar en ambos nodos:

```bash
ip addr show enp1s0f1np1
```

### 1.4 Configurar SSH sin contrasena

#### Automatico (recomendado)

Descargar y ejecutar el script [discover-sparks.sh](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) desde cualquiera de los nodos:

```bash
bash ./discover-sparks
```

La primera vez pedira la contrasena de cada nodo.

#### Manual

Obtener las IPs de las interfaces activas en ambos nodos:

```bash
ip addr show enp1s0f1np1
```

Luego en **ambos nodos**:

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub <usuario>@<IP_Node_1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <usuario>@<IP_Node_2>
```

### 1.5 Verificar comunicacion

```bash
ssh <IP_Node_1> hostname
ssh <IP_Node_2> hostname
```

### Rollback

```bash
# Si usaste netplan (Opcion A/B):
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply

# Si usaste IP manual (Opcion C):
sudo ip addr del 192.168.100.10/24 dev enp1s0f1np1   # ajustar IP e interfaz
```

### Troubleshooting

| Sintoma | Causa | Solucion |
|---|---|---|
| "Network unreachable" | Interfaces no configuradas | Verificar netplan y ejecutar `sudo netplan apply` |
| SSH falla autenticacion | Llaves no distribuidas | Re-ejecutar `./discover-sparks` |
| Node 2 no visible | Conectividad de red | Verificar cable QSFP y configuracion IP |

---

## 2. Habilitar NCCL

> Documentacion oficial: [NCCL on Stacked Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks)

NCCL (NVIDIA Collective Communication Library) habilita comunicacion GPU-to-GPU de alto rendimiento entre nodos. Debe compilarse desde fuente con soporte para Blackwell (SM 121).

**Prerequisitos:** Haber completado la [Seccion 1](#1-conectar-dos-sparks) (conectividad de red y SSH).

### 2.1 Compilar NCCL con soporte Blackwell

Ejecutar en **ambos nodos**:

```bash
sudo apt-get update && sudo apt-get install -y libopenmpi-dev

git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
```

Configurar las variables de entorno:

```bash
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

### 2.2 Compilar NCCL tests

Ejecutar en **ambos nodos**:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

### 2.3 Identificar la interfaz de red activa

```bash
ibdev2netdev
```

Usar la interfaz que muestra `(Up)` — tipicamente `enp1s0f1np1`. Obtener las IPs en ambos nodos:

```bash
ip addr show enp1s0f1np1
```

### 2.4 Ejecutar test de comunicacion NCCL

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

mpirun -np 2 -H <IP_Node_1>:1,<IP_Node_2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

Test con buffer grande para verificar ancho de banda completo (200 Gbps):

```bash
mpirun -np 2 -H <IP_Node_1>:1,<IP_Node_2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> **Nota:** Las IPs van seguidas de `:1`. Ejemplo: `mpirun -np 2 -H 192.168.200.12:1,192.168.200.13:1`

### Rollback

```bash
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

### Troubleshooting

| Sintoma | Causa | Solucion |
|---|---|---|
| `mpirun` se cuelga o timeout | Problemas de SSH | Probar `ssh <IP_remota>` y verificar llaves. Probar `mpirun -np 2 -H <IP1>:1,<IP2>:1 hostname` |
| Interfaz de red no encontrada | Nombre incorrecto o interfaz caida | Verificar con `ibdev2netdev` |
| NCCL no compila | Faltan dependencias (OpenMPI, CUDA) | Verificar instalacion de CUDA y librerias |

---

## 3. Inferencia distribuida con vLLM

> Documentacion oficial: [vLLM on Stacked Sparks](https://build.nvidia.com/spark/vllm/stacked-sparks)

Una vez configurada la red y validado NCCL, se puede ejecutar inferencia distribuida con vLLM usando tensor parallelism (TP=2) a traves de un cluster Ray.

**Prerequisitos:** Haber completado la [Seccion 1](#1-conectar-dos-sparks) y la [Seccion 2](#2-habilitar-nccl).

### 3.1 Obtener la imagen de Docker

```bash
# Imagen oficial NVIDIA vLLM (ultima version en NGC):
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
docker pull nvcr.io/nvidia/vllm:26.03-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.03-py3
```

Para la familia de modelos Gemma 4, usar la imagen custom:

```bash
docker pull vllm/vllm-openai:gemma4-cu130
```

### 3.2 Descargar el script de despliegue

En **ambos nodos**:

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```

> **Nota:** Este proyecto incluye una version propia del script en `Server/ray-cluster/run_cluster.sh` con mejoras de resiliencia (`--restart unless-stopped`, nombres deterministas). Puedes usar esa en su lugar.

### 3.3 Iniciar el nodo head (Node 1)

```bash
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Usando interfaz $MN_IF_NAME con IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

### 3.4 Iniciar el nodo worker (Node 2)

```bash
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

# IP del Node 1 (head)
export HEAD_NODE_IP=<IP_NODE_1>

echo "Worker IP: $VLLM_HOST_IP, conectando al head en: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

> **Nota:** Reemplazar `<IP_NODE_1>` con la IP real del Node 1 en la interfaz QSFP.

### 3.5 Verificar el cluster

```bash
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Contenedor: $VLLM_CONTAINER"

docker exec $VLLM_CONTAINER ray status
```

Debe mostrar 2 nodos activos con recursos GPU disponibles.

### 3.6 Descargar un modelo y lanzar inferencia

Ejemplo con Llama 3.3 70B:

```bash
# Dentro del contenedor del head:
docker exec -it $VLLM_CONTAINER /bin/bash

# Autenticarse en HuggingFace y descargar el modelo
hf auth login
hf download meta-llama/Llama-3.3-70B-Instruct

# Iniciar el servidor con TP=2
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 \
  --max_model_len 2048
```

### 3.7 Probar la inferencia

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

### 3.8 Validar el despliegue

```bash
# Estado del cluster Ray
docker exec $VLLM_CONTAINER ray status

# Health check del servidor
curl http://localhost:8000/health

# Uso de GPU en ambos nodos
nvidia-smi
docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Modelos soportados

| Modelo | Cuantizacion | HuggingFace |
|---|---|---|
| Gemma 4 31B IT | Base | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 31B IT | NVFP4 | [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) |
| Nemotron-3-Super-120B | NVFP4 | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| GPT-OSS-120B | MXFP4 | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) |
| GPT-OSS-20B | MXFP4 | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) |
| Llama-3.3-70B-Instruct | NVFP4 | [`nvidia/Llama-3.3-70B-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4) |
| Qwen3-32B | NVFP4 | [`nvidia/Qwen3-32B-NVFP4`](https://huggingface.co/nvidia/Qwen3-32B-NVFP4) |
| Qwen3-14B | FP8 / NVFP4 | [`nvidia/Qwen3-14B-FP8`](https://huggingface.co/nvidia/Qwen3-14B-FP8) |
| Phi-4-reasoning-plus | FP8 / NVFP4 | [`nvidia/Phi-4-reasoning-plus-FP8`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-FP8) |

> Ver la [matriz completa de modelos](https://build.nvidia.com/spark/vllm/stacked-sparks) en la documentacion oficial.

### Troubleshooting

| Sintoma | Causa | Solucion |
|---|---|---|
| Node 2 no aparece en Ray | Conectividad de red | Verificar cable QSFP y configuracion IP |
| Descarga de modelo falla | Autenticacion HuggingFace | Re-ejecutar `hf auth login`. Para modelos gated, solicitar acceso en HF |
| CUDA out of memory con modelos grandes | VRAM insuficiente | Reducir `--max-model-len` o usar un modelo con cuantizacion mas agresiva |
| Contenedor no arranca | Imagen ARM64 no disponible | Verificar con `docker images` que la imagen existe |

> **Nota:** DGX Spark usa Unified Memory Architecture (UMA), que permite compartir memoria entre GPU y CPU. Si hay problemas de memoria, limpiar la cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

---

## Referencias

- [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) — Configuracion fisica y de red
- [NCCL on Stacked Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks) — Compilacion y validacion de NCCL
- [vLLM on Stacked Sparks](https://build.nvidia.com/spark/vllm/stacked-sparks) — Inferencia distribuida con vLLM
- [DGX Spark Playbooks (GitHub)](https://github.com/NVIDIA/dgx-spark-playbooks) — Scripts y recursos auxiliares
