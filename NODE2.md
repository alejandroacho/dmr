# Node 2 — worker del cluster (192.168.200.13)

Inventario de qué hay realmente en el nodo worker, de dónde sale cada cosa y qué
no está versionado. Sirve para reconstruirlo desde cero y para saber, sin entrar
por SSH, qué debería estar ahí.

**Verificado el 2026-09-10** contra la máquina (`gx10-45d8`). Los checksums de
los 22 ficheros desplegados coincidían byte a byte con este repo.

El montaje general de los dos nodos está en [README.md](README.md) → *Cluster
Setup*, que se ejecuta entero **desde Node 1**. Este documento cubre solo el
extremo del worker.

---

## 1. Qué corre hoy

| Elemento | Valor |
|---|---|
| Hostname | `gx10-45d8` |
| IP del cluster | `192.168.200.13/24` en `enp1s0f1np1` (MTU 1500) |
| Contenedor | `vllm_node`, imagen `vllm-node-b12x:latest` (`bbd336999292`) |
| PID 1 del contenedor | `sleep infinity` — el Gateway gestiona el *proceso*, nunca el contenedor |
| Política de reinicio | `restart=no` — **no vuelve solo tras un reboot** (ver §5) |
| Pesos montados | `~/hf-cache` → `/root/.cache/huggingface` |
| Unidades systemd activas | `node-forensics`, `thermal-guard` |

El Gateway alcanza este nodo por SSH y ejecuta `docker exec vllm_node …` para
levantar el rank headless — ver `_run_on_worker()` en
[gateway/orchestrator.py](gateway/orchestrator.py). El worker **no sirve la API**.

### Modelos que dependen de este nodo

| Modelo | Modo | Pesos en Node 2 |
|---|---|---|
| `deepseek-v4-flash` | `cluster_nodes=2`, rank headless aquí | ✅ `~/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731` (156 GB) |
| `qwen3.5-122b` | `requires_ray_cluster=True`, Ray sobre los dos contenedores | ❌ no descargados **en ningún nodo** |

---

## 2. Código que debe estar, y su fuente en git

Todo lo de abajo es copia literal de este repo. No hay nada escrito a mano en
el worker: si un checksum no coincide, alguien editó en caliente.

| Ruta en Node 2 | Fuente en el repo | Cómo llega |
|---|---|---|
| `~/Server/ops/` (12 ficheros) | [ops/](ops/) | `git clone` o copia del directorio |
| `/etc/systemd/system/node-forensics.service` | [ops/node-forensics.service](ops/node-forensics.service) | `sudo bash ops/install.sh` |
| `/etc/systemd/system/thermal-guard.service` | [ops/thermal-guard.service](ops/thermal-guard.service) | `sudo bash ops/install.sh` |
| `/etc/systemd/system/gpu-clock-cap.service` | [ops/gpu-clock-cap.service](ops/gpu-clock-cap.service) | `install.sh` (queda *disabled*, es excluyente con thermal-guard) |
| `/etc/systemd/system/idle-experiment.service` | [ops/idle-experiment.service](ops/idle-experiment.service) | `install.sh` |

`ops/install.sh` es idempotente y **sirve para los dos nodos sin editar nada**:
detecta el rol por la IP de `enp1s0f1np1` (.12 → head, .13 → worker) y escribe
el peer en un drop-in generado, `node-forensics.service.d/10-peer.conf`, que por
diseño **no** está en git. En Node 2 contiene `FORENSICS_PEER=192.168.200.12`.

Los CSV de `~/Server/ops/logs/` son datos, no código, y `ops/.gitignore` ya los
excluye. Correcto: son ~8 ficheros diarios de muestreo.

### Ficheros legacy que siguen en el worker

`~/reset_ray_node.sh`, `~/run_cluster.sh`, `~/check_ray_cluster_status.sh`,
`~/ray-node-worker.service` y `~/README_worker_node.md` son del montaje Ray
anterior. Coinciden con [ray-cluster/](ray-cluster/), pero ya no sirven a nada
—ver §5. Ocupan unos pocos KB.

---

## 3. Qué NO está en git (y no puede estarlo)

Esto es lo que se pierde si se reinstala el nodo. **No hay copia.**

| Qué | Dónde vive | Nota |
|---|---|---|
| `/etc/netplan/40-cx7.yaml` | solo en Node 2 | Asigna `192.168.200.13/24` a `enp1s0f1np1`. Sin esto no hay cluster |
| `~/.ssh/authorized_keys` | solo en Node 2 | 3 claves: `shared-cluster-key`, `blackwell-gateway->spark-worker`, `alejandroacho@Mac.lan`. La segunda es la que usa el Gateway (paso 6 del README); sin ella el swap de DeepSeek falla |
| Imagen `vllm-node-b12x:latest` | Docker local | Se regenera con `./build-and-copy.sh --exp-b12x -c` desde Node 1 |
| Pesos `~/hf-cache` (156 GB) | disco local | Se recopian con `HF_HOME=~/hf-cache ./hf-download.sh … -c` |
| Drop-in `10-peer.conf` | generado | Lo rehace `ops/install.sh` |

Las claves y el netplan son los dos únicos elementos que exigen intervención
manual en una reinstalación. El resto se reconstruye desde Node 1.

---

## 4. Reconstruir el nodo desde cero

```bash
# En Node 2, una sola vez:
#  1. netplan: 192.168.200.13/24 en enp1s0f1np1
#  2. Docker + NVIDIA Container Toolkit
#  3. autorizar la clave del Gateway (desde Node 1):
#       ssh-copy-id -f -i ~/.ssh/id_spark_gateway.pub 192.168.200.13

# Vigilancia térmica y forense (en Node 2):
git clone <repo> ~/Server && sudo bash ~/Server/ops/install.sh

# Todo lo demás, desde Node 1:
cd ~/spark-vllm-docker
./build-and-copy.sh --exp-b12x -c                                    # imagen
HF_HOME=~/hf-cache ./hf-download.sh deepseek-ai/DeepSeek-V4-Flash-0731 -c
HF_HOME=~/hf-cache ./run-recipe.sh deepseek-v4-flash-0731 --port 8020 -d
```

---

## 5. Deuda conocida en este nodo

- **`ray-node-worker.service` está `enabled` y hay que desactivarlo.** Se ejecuta
  en cada arranque, no encuentra `blackwell-vllm:latest` y falla — pero
  `run_cluster.sh` no comprueba el código de salida de `docker run`, así que
  systemd lo registra como `status=0/SUCCESS` y, con `RemainAfterExit=yes`, lo
  muestra como *active*. Hoy es inofensivo solo porque falta la imagen: si
  alguien la recupera, arrancaría un `ray-node-worker` con `--gpus all`
  compitiendo por la memoria unificada con `vllm_node`.
  ```bash
  sudo systemctl disable --now ray-node-worker
  ```

- **`~/.cache/huggingface` ocupa 157 GB huérfanos.** Es la caché de la era Ray:
  no está montada en `vllm_node`, ninguno de sus modelos
  (`Qwen3-Coder-Next-FP8` 75 G, `gemma-4-31B-it` 59 G, `Qwen3.6-35B-A3B-FP8`,
  y `Qwen3-4B` **duplicado** como `Qwen3-4b`, 7,5 G cada uno) es un modelo de
  cluster, y está sin tocar desde abril/mayo. En Node 1 esa limpieza ya se hizo
  —allí pesa 149 MB—; Node 2 se quedó atrás. Es `root:root`, requiere `sudo`.

- **Imagen dangling de 26,7 GB.** `docker image prune`. Los tags
  `eugr/spark-vllm-b12x` y `vllm-node-b12x` comparten ID: borrar uno no libera nada.

- **No hay keep-alive del contenedor en este nodo, y es correcto.** `vllm_node`
  corre con `restart=no` en ambos lados y el head es quien lo recrea, vía
  `vllm-cluster.timer` en Node 1 (verificado activo el 2026-09-10). Node 2 no
  debe tener unidades `vllm-cluster.*`, y no las tiene.

- **MTU 1500 en el enlace de 200 GbE.** Para NCCL suele configurarse a 9000. Sin
  medir el impacto aquí.

---

## 6. Verificación rápida

Desde Node 1:

```bash
# ¿Contenedor arriba en los dos lados?
docker ps --filter name=vllm_node
ssh 192.168.200.13 'docker ps --filter name=vllm_node'

# ¿El Gateway alcanza el worker con SU clave, no con la tuya?
docker exec blackwell-gateway ssh -i /ssh/id_spark -o BatchMode=yes \
  alejandroacho@192.168.200.13 hostname

# ¿Lo desplegado sigue coincidiendo con el repo?
ssh 192.168.200.13 'cd ~/Server/ops && md5sum *.py *.sh *.service' | sort
(cd ~/Server/ops && md5sum *.py *.sh *.service | sort)

# Log del rank 1
ssh 192.168.200.13 'docker exec vllm_node tail -f /tmp/vllm_deepseek-v4-flash_r1.log'
```
