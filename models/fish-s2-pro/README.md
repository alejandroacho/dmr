# Fish Audio S2 Pro (TTS)

Servicio independiente para el nodo ARM64 NVIDIA GB10. Usa la imagen local
`media-node:latest` como base para conservar PyTorch/CUDA compatibles con la GPU.
El código upstream y los pesos están fijados a revisiones concretas en el
Dockerfile y el script de descarga.

Desde la raíz de `Server`, con `huggingface_hub` instalado:

```bash
bash models/fish-s2-pro/download.sh
docker compose --profile tts build fish-tts
docker compose --profile tts up -d fish-tts
curl --fail http://localhost:8011/v1/health
```

Generar voz (el texto determina el idioma):

```bash
curl --fail-with-body http://localhost:8011/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hola, soy Fish y estoy funcionando en este nodo.","format":"wav","seed":42}' \
  -o /tmp/fish-demo.wav
```

API directa en el puerto 8011; el gateway multimedia del puerto 8000 mantiene
sus rutas existentes. El servidor carga S2 Pro al arrancar en BF16, con un solo
worker y sin `torch.compile`. Las referencias de voz se conservan en el volumen
`fish_references`; la API acepta `references` o `reference_id`.

Fish y ComfyUI comparten la memoria unificada, pero sus colas y gestores de
memoria son independientes. No hay expulsión automática de Fish desde ComfyUI.
Si necesitas toda la memoria para vídeo, libera Fish con:

```bash
docker compose --profile tts stop fish-tts
```

Documentación upstream: https://github.com/fishaudio/fish-speech/blob/main/docs/en/server.md
