# ops — diagnóstico y protección del nodo 1 (gx10-52c3)

Herramientas nacidas de investigar por qué el nodo 1 se apagaba de golpe, sin
dejar rastro en el journal, en el kernel ni en el firmware.

## Conclusión de la investigación

El nodo se corta en seco cuando la **GPU supera ~85 °C** (SoC ~93 °C). Base
empírica: 18 días de muestreo continuo y 3 apagones instrumentados.

| Corte | GPU | tz0 | Potencia |
|---|---|---|---|
| 19-ago 01:44 | 91 °C | 96,0 °C | 67,5 W |
| 01-sep 16:34 | 86 °C | 94,3 °C | 66,2 W |
| 01-sep 20:38 | 85 °C | 93,1 °C | 45,2 W |

En esos 18 días la máquina pasó **47,7 minutos** por encima de 85 °C, y los tres
cortes cayeron ahí dentro — uno por cada ~16 min de exposición, frente a 13 días
seguidos de estabilidad fuera de la banda.

**No es sobrecorriente en esta unidad**: sobrevivió picos de 84,5 W y murió a
45 W estando caliente. Es un problema conocido de la plataforma GB10; hay quien
lo atribuye a OCP y quien lo atribuye a un corte térmico del EC. Los datos de
este nodo apuntan a lo segundo.

Descartado con datos: red eléctrica (el nodo 2 sobrevive a todos los cortes
salvo el del 15-ago, que sí fue un apagón real), errores AER de la ConnectX-7,
saturación de carga, memoria, y degradación de la refrigeración (18 días con el
reposo clavado en 54-56 °C).

El firmware de este equipo no tiene BERT, ERST, HEST, BMC ni variables EFI de
fallo: no guarda nada del apagón anterior. De ahí el muestreador propio.

## Qué hay aquí

### Protección (lo que evita el apagón)

- **`thermal_guard.py`** + `thermal-guard.service` — **el arreglo.** Vigila la
  GPU cada 2 s y baja el techo de reloj por escalones (3003→2400→2100→1900→
  1700→1500) al llegar a 80 °C, con doble salto desde 83 °C. Devuelve
  rendimiento tras 2 min por debajo de 72 °C. Los umbrales van 5 °C por debajo
  del corte observado para reaccionar antes de entrar en la banda.
  En GB10 `nvidia-smi -pl` no existe; el único control es `-lgc`.

- **`gpu_clock_cap.sh`** + `gpu-clock-cap.service` — alternativa de cap fijo a
  2100 MHz (el valor documentado por la comunidad). **Excluyente** con el
  gobernador: usa uno u otro. Cuesta rendimiento siempre, también en frío.

### Diagnóstico

- **`node_forensics.py`** + `node-forensics.service` — muestrea cada 5 s a CSV
  con **fsync por línea**. Ese fsync es el punto entero: en los tres cortes el
  log llegó más lejos que el journal (85 s más en el del 1-sep). Registra
  temperaturas (SoC, GPU, NVMe, las 4 ConnectX-7 con su pico latcheado por
  hardware), potencia de GPU con **mín/máx a 200 ms**, contadores AER por
  dispositivo, estados de reposo de CPU y latencia con el nodo 2.
  Marca `#BOOT_START` al arrancar y `#CLEAN_STOP` con SIGTERM: una sesión sin
  `#CLEAN_STOP` es exactamente la firma de una muerte sin avisar.

- **`crash_report.py`** — post-mortem. Localiza las sesiones que murieron y
  muestra los minutos previos, picos y frenadas por hardware.
  `--all` para el histórico, `--window N` para ajustar la cola.

### Operación

- **`install.sh`** — idempotente, instala todas las unidades. Las de experimento
  (`idle-experiment`, `gpu-clock-cap`, `thermal-guard`) quedan **desactivadas**;
  se arrancan a propósito. También instala `vllm-cluster.service/.timer` desde
  `../ray-cluster/`, que existían en el repo pero nunca se habían copiado a
  `/etc/systemd/system` — por eso, tras cada corte, nadie recreaba el
  `vllm_node` del head y DeepSeek se quedaba caído.

- **`idle_experiment.sh`** + `idle-experiment.service` — **hipótesis refutada**,
  se conserva por trazabilidad. Desactivaba los estados de reposo profundo
  (LPI-2/LPI-3). Pareció funcionar (13 días sin caídas) pero era un espejismo:
  la exposición a >86 °C había caído 7,6× en ese periodo, y la tasa por minuto
  de exposición no cambió.

## Uso

```bash
sudo bash install.sh
sudo systemctl enable --now thermal-guard.service   # la protección

journalctl -fu thermal-guard.service                # qué decide
python3 crash_report.py                             # tras un corte
```

Criterio de éxito, comprobable en horas y no en semanas: bajo carga máxima la
GPU **no debe pasar de 85 °C**. Si los roza, bajar umbrales con
`systemctl edit thermal-guard.service` (`GUARD_WARN_C`, `GUARD_DANGER_C`).

Si vuelve a caerse **por debajo** de 85 °C, el modelo está equivocado y toca
RMA — con 18 días de telemetría y tres post-mortem para respaldarlo.

## Notas de mantenimiento

Los logs rotan a diario y se podan a los 21 días (`FORENSICS_RETENTION_DAYS`).
Ocupan ~4 MB/día a 5 s de intervalo.

Si se añaden o quitan columnas al muestreador **hay que reiniciar el servicio**;
`node_forensics.py` reescribe la cabecera al detectar un juego de columnas
distinto. Sin eso quedaban filas bajo una cabecera antigua y el post-mortem leía
todas las columnas corridas — pasó dos veces durante la investigación, una de
ellas con el mismo ancho y distinto significado (desapareció el hwmon del WiFi y
entró `deep_idle_off`), que es el caso que ninguna comprobación de ancho detecta.
