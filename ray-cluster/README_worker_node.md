# Node 2 — Worker Node (192.168.200.13)

## Requisitos previos

- Docker instalado y con acceso a GPU
- Imagen `blackwell-vllm:latest` cargada localmente (transferida desde Node 1)
- Node 1 (head) ya levantado y escuchando en `192.168.200.12`

### Transferir la imagen desde Node 1

```bash
# Desde Node 1:
docker save blackwell-vllm:latest | ssh alejandroacho@192.168.200.13 docker load
```

---

## 1. Copiar los scripts y el service file desde Node 1

```bash
scp alejandroacho@192.168.200.12:~/Server/ray-cluster/reset_ray_node.sh ~/
scp alejandroacho@192.168.200.12:~/Server/ray-cluster/run_cluster.sh ~/
scp alejandroacho@192.168.200.12:~/Server/ray-cluster/check_ray_cluster_status.sh ~/
scp alejandroacho@192.168.200.12:~/Server/ray-cluster/ray-node-worker.service ~/
```

O clonar el repo completo en Node 2 (`git clone ... ~/Server`).

---

## 2. Levantar el nodo worker

Asegúrate de que **Node 1 ya está corriendo** antes de ejecutar esto.

```bash
bash ~/Server/ray-cluster/reset_ray_node.sh --worker 192.168.200.12
```

Detiene cualquier contenedor `ray-node-*` previo y arranca `ray-node-worker` en segundo plano con `--restart unless-stopped`.

---

## 3. Verificar que se unió al cluster

Desde **Node 1**, verificar que aparecen 2 nodos:

```bash
bash ~/Server/ray-cluster/check_ray_cluster_status.sh
```

```
Active:
 1 ray-node-head   ← Node 1
 1 ray-node-worker ← este nodo

Resources:
 0.0/2.0 GPU
```

---

## 4. Configurar el servicio systemd (auto-arranque en reboot)

```bash
sudo cp ~/ray-node-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ray-node-worker
sudo systemctl start ray-node-worker
```

Verificar:

```bash
systemctl status ray-node-worker
docker inspect ray-node-worker | grep -A3 RestartPolicy
```

---

## Comandos útiles

| Acción | Comando |
|---|---|
| Ver logs del nodo | `docker logs -f ray-node-worker` |
| Abrir shell en el contenedor | `docker exec -it ray-node-worker /bin/bash` |
| Reiniciar el worker | `bash ~/Server/ray-cluster/reset_ray_node.sh --worker 192.168.200.12` |
| Detener el nodo | `docker stop ray-node-worker` |

---

## Notas

- El worker **no** sirve la API — eso lo hace el gateway desde Node 1.
- Si el head se reinicia, el worker se reconecta automáticamente gracias a `--restart unless-stopped`.
- Si el worker está caído, el gateway lo detecta en el próximo ciclo del watchdog (cada 30 s) o inmediatamente al intentar iniciar un modelo TP=2, y devuelve un error claro en lugar de colgar.
