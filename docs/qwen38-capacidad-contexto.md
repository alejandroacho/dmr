# Capacidad de Qwen3.8-Flash-Next en este clúster

Revisión: 2026-09-25. Inspección de configuración, código instalado, logs y métricas; sin cambiar la configuración ni ejecutar una prueba de saturación.

## Resultado práctico

Un agente no reserva una ventana permanente en vLLM. La unidad de ejecución es la petición/secuencia. La tabla supone una petición activa por agente, todos usando Qwen, contextos independientes y sin contar ahorro por prefijos compartidos. Contexto = entrada completa + salida, incluido razonamiento.

| Peticiones activas | Techo orientativo por petición, por memoria | Presupuesto inicial con margen por petición |
|---:|---:|---:|
| 1–2 | 1.000.000 | Hasta 1.000.000 |
| 3 | 1.000.000 | 900.000 |
| 4 | ~960.000 | 750.000 |
| 6 | ~630.000 | 500.000 |
| 8 | ~470.000 | 350.000 |
| 12 | ~300.000 | 225.000 |
| 16 | ~220.000 | 160.000 |

La columna con margen es una propuesta de presupuesto de memoria, no una garantía de latencia ni un benchmark. Para 8 agentes con presupuesto de 350.000 y salida máxima de 32.000, limitar la entrada tokenizada completa a 318.000. Incluir plantilla, herramientas y mensajes de sistema en esa entrada.

## Datos comprobados

- Modelo: local-inference-lab/Qwen3.8-Flash-Next-NVFP4.
- vLLM: 0.1.dev21460+gaf9e4dca1.d20260920, B12X.
- Dos nodos, tensor parallel 2: cooperan en cada inferencia. No son dos réplicas independientes.
- max_model_len=1.000.000, YaRN factor 4, MTP desactivado.
- max_num_seqs=16; max_num_batched_tokens=4096; chunked prefill y prefix caching activos.
- gpu_memory_utilization=0.7; KV FP8; Mamba align, estado SSM float32.
- Log del rank 0: 25,52 GiB disponibles para caché; pesos y memoria no PyTorch 57,21 GiB; activaciones pico 2,04 GiB; CUDA graphs 0,09 GiB.
- Métrica cache_config_info: block_size=2784, num_gpu_blocks=1423, kv_cache_size_tokens=3866847, kv_cache_max_concurrency=3.8668478260869565.
- Un bloque es nulo/reservado: quedan 1422 utilizables. No multiplicar la cifra de tokens por dos nodos.

## Qué significa la cifra de 3,87 millones

El código instalado calcula primero la concurrencia como bloques del pool / bloques necesarios por petición de longitud máxima. Después multiplica por max_model_len para anunciar una capacidad equivalente en tokens. Por tanto, 3.866.847 no es una bolsa lineal exacta de tokens aplicable a cualquier mezcla de longitudes.

El cociente publicado corresponde a 1423 / 368. La atención requiere ceil(1.000.000 / 2784)=360 bloques; los ocho restantes corresponden al coste adicional de los grupos de caché. Mamba align presupuesta dos páginas por grupo sin especulación ni checkpoints adicionales.

Para elaborar la tabla se ha usado el modelo conservador de bloques B(L)≈ceil(L/2784)+8, que reproduce el cálculo de arranque, con 1422 bloques utilizables. Es una reconstrucción del presupuesto de capacidad; no una medición de los bloques de cada petición viva. Estados transitorios, límites del planificador y reutilización de prefijos cambian la ocupación efectiva. La propuesta con margen evita acercarse al límite del pool.

Ejemplos calculados: 3 peticiones de 1M necesitan aproximadamente 1104 bloques y caben por memoria; 4 necesitan 1472 y no caben simultáneamente sin compartir bloques. 16 peticiones de 256.000 requieren unos 1600 bloques y tampoco caben, aunque max_num_seqs sea 16.

## Ciclo de una petición

1. El cliente envía instrucciones, historial, herramientas y documentos. El servidor tokeniza y valida la ventana individual.
2. Busca prefijos reutilizables. En el modelo híbrido necesita coincidencias compatibles de atención y estado recurrente; compartir documentos en posiciones distintas no equivale a compartir un prefijo.
3. El planificador admite trabajo según plazas, bloques libres y presupuesto de tokens. En esta versión scheduler_reserve_full_isl=True comprueba que quepa la entrada completa antes de admitirla. No es una reserva permanente de 1M por agente ni una garantía para toda la salida futura.
4. El prefill procesa la entrada por fragmentos. Los 4096 tokens son presupuesto de trabajo del planificador por paso, no una ventana de contexto por agente.
5. La generación alarga cada secuencia y consume bloques cuando hace falta. Si no puede asignar más, vLLM puede preemptar y recalcular peticiones; también puede mantenerlas en espera.
6. Al terminar, la petición libera su ocupación activa. Algunos bloques quedan reutilizables como prefijo y pueden expulsarse. No hay una sesión de chat permanente asociada a un agente: el cliente conserva y vuelve a enviar el historial.

Mientras un agente ejecuta herramientas fuera del modelo, no ocupa una petición de inferencia activa. Si lanza varios subagentes o solicitudes paralelas, consume varias. Los contextos no se mezclan y el historial acumulado de todos los agentes no tiene que residir permanentemente en la caché.

## Límites prácticos adicionales

- gateway/proxy.py usa TCPConnector(limit=100, limit_per_host=30) y ClientTimeout(total=300, connect=10). El timeout total también se aplica a streaming: incluye espera y procesamiento de ese intento HTTP. Una respuesta puede interrumpirse aunque haya memoria suficiente.
- MAX_QUEUE_SIZE=200 en request_buffer.py corresponde a la cola durante cambios de modelo, no a 200 inferencias simultáneas.
- El gateway intercambia modelos dentro de los mismos contenedores. Durante esta revisión otro cliente pidió DeepSeek y se inició el apagado de Qwen. Mezclar agentes que solicitan modelos diferentes puede interrumpir peticiones; esta tabla sólo corresponde a Qwen cargado.

## Observación de rendimiento, no benchmark

En una ventana de 73,5 segundos de tráfico existente se observaron 3427 tokens nuevos de salida (~46,6 tok/s agregados), 91048 tokens de entrada contabilizados, cero preempciones nuevas y cero peticiones esperando al final. Hubo instantáneas de 2–4 peticiones activas; no se conoce la concurrencia exacta durante cada instante de la ventana. La tasa de salida no debe dividirse y extrapolarse como garantía para 8 o 16 agentes o para contextos de 1M.

El cociente de sumas de inter-token latency en esa ventana fue ~45,9 ms por observación. Incluye una mezcla de peticiones y fases; no es un benchmark aislado. Los logs del gateway rotulan una tasa como decode aunque, si no viene time_to_first_token_ms, el denominador incluye todo el tiempo de la petición. Por eso no se ha usado esa etiqueta como medición pura de decodificación.

Para fijar una capacidad por latencia falta medir una matriz controlada de concurrencia, longitud de entrada, longitud de salida y caché fría/caliente. Los datos actuales establecen los límites configurados y una planificación de memoria; no validan 3×1M ni 16 agentes bajo carga máxima.

## Fuentes

Configuración local: gateway/config.py, gateway/proxy.py, gateway/request_buffer.py. Evidencia runtime: /metrics, /tmp/vllm_qwen3.8-flash-next_r0.log dentro de vllm_node.

Código de la imagen inspeccionado: vllm/v1/core/kv_cache_utils.py (get_max_concurrency_for_kv_cache_config, get_kv_cache_capacity), vllm/v1/kv_cache_interface.py (FullAttentionSpec, MambaSpec), vllm/v1/core/block_pool.py (get_usage), vllm/v1/core/kv_cache_manager.py (allocate_slots), vllm/v1/core/sched/scheduler.py y vllm/config/scheduler.py.

- https://docs.vllm.ai/en/latest/configuration/optimization/
- https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/

La documentación pública puede ir por delante o detrás del fork B12X. Para los detalles específicos se ha dado prioridad al código instalado y las métricas de este arranque.
