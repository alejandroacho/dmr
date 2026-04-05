# Node 1 — Head Node (192.168.200.12)

## Requisitos previos

- Docker instalado y con acceso a GPU
- Imagen `blackwell-vllm:latest` construida (`just build-blackwell` desde `Server/`)
- Node 2 (worker) accesible en `192.168.200.13`

---

## 1. Levantar el nodo head

```bash
bash ~/Server/ray-cluster/reset_ray_node.sh --head
```

Detiene cualquier contenedor `ray-node-*` previo y arranca uno nuevo llamado `ray-node-head` en segundo plano con `--restart unless-stopped`.

---

## 2. Verificar que el cluster está listo

```bash
bash ~/Server/ray-cluster/check_ray_cluster_status.sh
```

Debe mostrar exactamente **2 nodos activos** y **2 GPUs** antes de continuar.
Si solo aparece 1 nodo, espera a que el worker (Node 2) se conecte.

```
Active:
 1 ray-node-head   ← este nodo
 1 ray-node-worker ← Node 2

Resources:
 0.0/2.0 GPU
```

---

## 3. Configurar el servicio systemd (auto-arranque en reboot)

```bash
sudo cp ~/Server/ray-cluster/ray-node-head.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ray-node-head
sudo systemctl start ray-node-head
```

Verificar:

```bash
systemctl status ray-node-head
docker inspect ray-node-head | grep -A3 RestartPolicy
```

---

## Comandos útiles

| Acción | Comando |
|---|---|
| Ver logs del nodo | `docker logs -f ray-node-head` |
| Abrir shell en el contenedor | `docker exec -it ray-node-head /bin/bash` |
| Ver estado del cluster | `bash ~/Server/ray-cluster/check_ray_cluster_status.sh` |
| Reiniciar todo | `bash ~/Server/ray-cluster/reset_ray_node.sh --head` |
| Detener el nodo | `docker stop ray-node-head` |

---

## Notas

- Lanzar siempre el head **antes** que el worker.
- El gateway detecta el container `ray-node-head` por nombre para lanzar modelos vía Ray.
- El watchdog del gateway verifica el estado del cluster cada 30 s y marca los modelos como `ERROR` si el head desaparece.
- Si un modelo TP=2 se intenta iniciar con el worker desconectado, el gateway falla inmediatamente con un error claro en lugar de colgar 10 min.
