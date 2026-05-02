# SilexCode 100% Finish Plan

Estado del proyecto a fecha 2026-05-02.

Este documento es la lista cerrada de trabajo para terminar el proyecto al 100%. La idea es que, cuando se ordene ejecutar, se pueda seguir de arriba abajo sin pedir confirmaciones y sin cambiar los objetivos a mitad.

## 0. Estado Actual Verificado

### Codigo ya implementado

- Arquitectura base `SilexCode-T18.6B-R64`.
- Tokenizer byte-level `V=258`.
- Kernels CUDA/C++ `TLinear` forward y backward-input.
- Runtime nativo de inferencia `silex_forward_cuda`.
- Runtime nativo de entrenamiento parcial `silex_train_chunk_cuda`.
- Workspace estatico de entrenamiento.
- Dataset sintetico determinista y curriculum.
- Checkpoints `.silex` y checkpoints plasticos.
- Scripts Vast.ai.
- Runner bootstrap.
- Runner accelerated curriculum.
- Adaptador de salida experimental opt-in.

### Pruebas locales recientes

- `python -m pytest -q`
- Resultado: `38 passed, 1 skipped`.

### Pruebas reales en Vast RTX 5090

GPU:

- NVIDIA GeForce RTX 5090.
- 32 GB VRAM.
- CUDA 12.9.

Resultados relevantes:

- Bootstrap B0 con adaptadores internos KFAC:
  - No aprendia de forma util.
  - `NLL_4` practicamente plano.

- Bootstrap B0 con adaptador de salida rank 64:
  - `NLL_4: 56.33 -> 0.66`.
  - `token_acc4: 0.09 -> 0.64`.
  - VRAM maxima aproximada: `4.1 GB`.
  - Tiempo: `~4.13 s/update`.

- Bootstrap B0 con adaptador de salida rank 256:
  - Peor que rank 64.
  - Mejor `NLL_4` aproximado: `0.717`.

Conclusion tecnica:

- El backbone produce senal entrenable.
- CUDA no esta colgado.
- El dataset si genera targets aprendibles.
- El bloqueo restante esta en la adaptacion interna/KFAC y en que la ruta nativa todavia no incorpora la cabeza/adaptador de salida calibrado.

## 1. Objetivo Final

Terminar el sistema para poder ejecutar Curriculum Learning real de forma verificable:

- Stage 1, Stage 2 y Stage 3 funcionales.
- Entrenamiento estable.
- Checkpoints recuperables.
- Validacion metrica reproducible.
- SSD-F activable en Stage 3.
- Uso de VRAM dentro del presupuesto del TDD o explicitamente marcado como modo experimental si usa mas.
- Scripts Vast reproducibles desde repo limpio.

## 2. Principio De No Romper El TDD

Hay dos rutas que deben quedar separadas:

### Ruta estricta TDD

- Backbone ternario congelado.
- `TLinear` nativo.
- `V=258`.
- Sin `torch.matmul` ni `nn.Linear` para el backbone.
- Python solo como launcher en inferencia nativa.
- KFAC sobre adaptadores internos.

### Ruta experimental bootstrap

- Puede usar adaptador de salida opt-in.
- Puede usar PyTorch/autograd temporalmente para diagnosticar y calibrar.
- Debe estar claramente detras de flags:
  - `--enable-output-adapter`
  - `--output-adapter-only`
- Nunca debe activarse por defecto.

## 3. Bloque Restante Principal

El sistema todavia no esta al 100% porque falta cerrar esta cadena:

1. Incorporar el adaptador de salida calibrado en la ruta de entrenamiento/inferencia donde se necesite.
2. Conectar el entrenamiento interno despues del bootstrap sin perder la mejora de logits.
3. Corregir o sustituir el KFAC interno si sigue sin aprender.
4. Ejecutar Stage 1 completo hasta umbrales reales.
5. Ejecutar Stage 2 completo.
6. Ejecutar Stage 3 con teacher cache y SSD-F.
7. Validar checkpoints y recuperacion.
8. Validar VRAM y rendimiento.

## 4. Plan De Implementacion Detallado

## Fase A: Hacer Que El Adaptador De Salida Sea Usable En Toda La Ruta

### A1. Native forward con output adapter

Problema:

- `forward_native` actualmente devuelve logits desde cabeza ternaria atada.
- Si `output_adapter_enabled=True`, la evaluacion usa Python para respetar el adaptador.
- Esto es correcto para diagnostico, pero lento para curriculum real.

Implementar:

- Binding C++ opcional para aplicar:

```text
logits = logits_base + (o_fp32 @ output_adapter_down.T) @ output_adapter_up.T
```

- Mantenerlo opt-in.
- No tocar la matematica del backbone.
- Aplicar solo sobre logits finales por profundidad.

Archivos:

- `silexcode/cuda/bindings.cpp`
- `silexcode/cuda/tlinear_kernels.cu` si se crea kernel dedicado.
- `silexcode/model.py`

Pruebas:

- Comparar `forward_python_reference` vs `forward_native` con output adapter activado.
- `torch.allclose(..., atol=1e-3)` para logits BF16/FP32 mixtos.

### A2. Checkpoints con output adapter

Problema:

- Los checkpoints plasticos guardan parametros `requires_grad`.
- Si se carga un checkpoint con output adapter en un modelo sin output adapter, debe fallar explicitamente.

Implementar:

- Metadata obligatoria:
  - `output_adapter_enabled`
  - `output_adapter_rank`
  - `route`
- Mensaje de error claro si hay mismatch.

Archivos:

- `silexcode/checkpoint.py`
- tests.

Pruebas:

- Export/import checkpoint de output adapter.
- Falla esperada si rank no coincide.
- Falla esperada si output adapter esta desactivado.

## Fase B: Usar Bootstrap Como Calibracion Inicial

### B1. Seleccionar checkpoint B0 bueno

Estado actual:

- Mejor conocido:
  - `runs/bootstrap_output_adapter_b0_200/bootstrap_latest.plastic.silex`
  - rank 64
  - `NLL_4~0.66`

Implementar:

- Script para copiar/descargar/promocionar checkpoint:

```text
runs/bootstrap_output_adapter_b0_200/bootstrap_latest.plastic.silex
-> runs/promoted/bootstrap_b0_output_adapter_rank64.silex
```

Archivos:

- `scripts/vast_promote_checkpoint.sh`
- opcional `promote_checkpoint.py`

### B2. Bootstrap B1-B4 con resume

Ejecutar:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ADAPTER_RANK=64 \
OUTPUT_ADAPTER_LR=0.001 \
BOOTSTRAP_LEVELS=1,2,3,4 \
UPDATES_PER_LEVEL=500 \
EVAL_EVERY=50 \
VAL_SIZE=16 \
scripts/vast_output_adapter_bootstrap.sh bootstrap_output_adapter_b1_b4_from_b0
```

Necesario:

- Anadir soporte `--resume` al script `vast_output_adapter_bootstrap.sh`.
- Validar que el checkpoint B0 se carga.

Exito minimo:

- B1 baja claramente.
- B2 baja claramente.
- Si B3/B4 no bajan, ajustar curriculum o capacidad.

## Fase C: Corregir Entrenamiento Interno/KFAC

### C1. Medir gradientes reales de adaptadores internos

Problema:

- KFAC interno no mostro aprendizaje.
- Antes de reescribir, hay que medir si los gradientes son cero, pequenos, mal escalados o recortados por trust-region.

Implementar diagnostico:

- Para cada capa o grupos de capas:
  - norma de gradiente de `A_m`
  - norma de gradiente de `B_m`
  - norma de gradiente de `A_f`
  - norma de gradiente de `B_f`
  - norma natural
  - `trust_chi`
  - update norm
  - param norm

Archivos:

- `silexcode/cuda/bindings.cpp`
- `silexcode/kfac.py`
- `run_accelerated_curriculum.py`
- `analyze_curriculum_metrics.py`

Exito:

- Saber si el fallo es:
  - gradiente ausente,
  - escala mala,
  - damping excesivo,
  - trust-region demasiado pequeno,
  - covarianzas degeneradas,
  - bug de backward.

### C2. Comparar KFAC nativo contra AdamW de adaptadores internos

Implementar modo experimental:

```text
--optimizer adamw_internal
```

Solo para diagnostico.

Uso:

- Congelar backbone.
- Entrenar `A_m/B_m/A_f/B_f` con AdamW PyTorch.
- Comparar si aprende mejor que KFAC.

Si AdamW aprende:

- El problema es KFAC/update.

Si AdamW no aprende:

- El problema es gradiente/arquitectura/adaptador.

### C3. Revisar inicializacion de B=0

Observacion:

- Los adaptadores son `B @ A @ u`.
- `A` inicial no cero.
- `B` inicial cero.
- Al inicio, gradiente fluye a `B`, pero no a `A` hasta que `B` cambia.

Accion:

- Confirmar que KFAC actual actualiza primero `B`.
- Confirmar que `A` empieza a recibir gradiente tras varios updates.
- Metrica por matriz:
  - `grad_A_norm`
  - `grad_B_norm`

### C4. Trust-region schedule

Problema observado antes:

- `natural_norm` enorme.
- `trust_chi` muy pequeno.

Implementar schedules:

- Warmup de KFAC con `eta=0`.
- Despues:
  - delta inicial mas alto durante bootstrap.
  - clipping por update norm.
  - damping adaptativo si `natural_norm` explota.

Mantener modo estricto original disponible.

Flags:

```text
--kfac-schedule conservative|bootstrap|strict
```

### C5. KFAC fallback por bloques diagonal

Si KFAC full-block sigue inestable, implementar fallback:

- Precondicion diagonal por bloque.
- Menos exacto que KFAC completo, pero verificable.
- Solo como modo experimental.

Flag:

```text
--kfac-mode block|diag|adamw
```

## Fase D: Curriculum Stage 1 Real

### D1. Stage 1 con bootstrap calibrado

Ejecutar Stage 1 usando:

- Checkpoint output adapter rank 64.
- Adaptadores internos corregidos.
- Curriculum real Stage 1.

Comando objetivo:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u run_accelerated_curriculum.py \
  --output-dir runs/stage1_real \
  --resume runs/promoted/bootstrap_b0_output_adapter_rank64.silex \
  --stages 1 \
  --updates-per-stage 200000 \
  --eval-every 2048 \
  --val-size 4096
```

Criterio TDD Stage 1:

- `NLL_4 <= 0.080`.
- `mono <= 0.0010`.
- `latent_gain >= 0.010`.
- `compile_pass >= 0.995`.
- 3 evaluaciones consecutivas.

### D2. Validacion de generacion de codigo

Implementar o verificar:

- `greedy_generate_code`.
- Parseo de `<C>...</C>`.
- `compile_and_unit_check_generated_code`.

Exito:

- No basta NLL.
- Tiene que compilar.

## Fase E: Curriculum Stage 2 Real

### E1. Teacher Stage 1 no requerido

Stage 2 usa:

- Problema formal + codigo + input -> traza.

Ejecutar:

```bash
python -u run_accelerated_curriculum.py \
  --output-dir runs/stage2_real \
  --resume runs/stage1_real/stage_1_complete.silex \
  --stages 2
```

Criterio TDD Stage 2:

- `NLL_4 <= 0.120`.
- `mono <= 0.0015`.
- `latent_gain >= 0.020`.
- `Acc_var >= 0.990`.
- `Acc_line >= 0.970`.
- 3 evaluaciones consecutivas.

### E2. Metricas de traza

Verificar:

- `variable_exact_counts`.
- `line_exact_counts`.
- Parseo robusto de lineas `@pc|...`.

## Fase F: Curriculum Stage 3 Real + Teacher + SSD-F

### F1. Precomputar teacher cache

Requisito:

- Teacher de Stage 2.
- No cargar segunda copia del modelo en VRAM.
- Precomputar logits en disco registro a registro.

Verificar:

- `precompute_stage3_teacher_cache`.
- Reader/writer.
- Shape `[511,258]`.
- dtype `float16` en disco, `float32` en GPU al usar.

### F2. Stage 3 sin SSD-F

Primero ejecutar oracle synthetic puro:

```bash
python -u run_accelerated_curriculum.py \
  --output-dir runs/stage3_oracle \
  --resume runs/stage2_real/stage_2_complete.silex \
  --stages 3 \
  --disable-ssd
```

Criterios:

- NLL baja.
- Unit tests generados empiezan a pasar.

### F3. Activar SSD-F

Despues:

- `build_ssd_pool`.
- `verify_candidate_code`.
- Consensus >= 2.
- Unit tests 512.

Criterio Stage 3 final:

- `NLL_4 <= 0.180`.
- `mono <= 0.0020`.
- `latent_gain >= 0.040`.
- `compile_pass >= 0.990`.
- `unit_pass@1 >= 0.920`.
- 3 evaluaciones consecutivas.

## Fase G: Rendimiento

### G1. Reducir coste del output adapter

Ahora:

- Bootstrap output adapter usa Python/autograd.
- `~4.1 s/update` en RTX 5090.

Mejoras:

- Native output adapter forward.
- Native output adapter backward.
- Fusion con logits.
- Evitar materializar activaciones innecesarias.

Objetivo:

- `>=2x` respecto a 4.1 s/update.

### G2. Optimizar evaluacion

Problema:

- Validacion grande consume mucho.

Implementar:

- `--val-size-fast` para pruebas.
- `--val-size-full` para hitos.
- Cache de registros de validacion generados.

### G3. Reporte de velocidad

Todas las corridas deben loguear:

- `step_seconds`
- `updates_per_minute`
- `max_memory_allocated_mb`
- `gpu_name`
- `cuda_version`
- commit git
- flags usados

## Fase H: VRAM Y Robustez

### H1. Stress real

Ejecutar:

```bash
python vram_stress_test.py --steps 10
```

Condicion estricta:

- Modo TDD 8GB: `max_memory_allocated <= 7256 MB`.

Condicion experimental 5090:

- Documentar VRAM real.
- No reclamar cumplimiento 8GB si el modo usa output adapter/autograd y supera presupuesto.

### H2. Recuperacion de checkpoint

Pruebas:

1. Entrenar 10 updates.
2. Guardar checkpoint.
3. Cargar checkpoint.
4. Continuar 10 updates.
5. Verificar que no falla y metricas siguen coherentes.

## Fase I: Scripts Vast Definitivos

Actualizar scripts:

- `scripts/vast_setup.sh`
- `scripts/vast_status.sh`
- `scripts/vast_stop_training.sh`
- `scripts/vast_output_adapter_bootstrap.sh`
- `scripts/vast_stage1_after_bootstrap.sh`

Anadir:

- `scripts/vast_run_full_curriculum.sh`
- `scripts/vast_tail_metrics.sh`
- `scripts/vast_collect_artifacts.sh`

Requisitos:

- Todos exportan `CUDA_VISIBLE_DEVICES=0`.
- Todos usan `/venv/main/bin/python`.
- Todos guardan logs en `runs/*.log`.
- Todos imprimen PID, metrics path, checkpoint path.

## Fase J: Documentacion Final

Actualizar:

- `README.md`
- `ACCELERATION_STATUS.md`
- `FULL_ACCELERATION_ROADMAP.md`

Anadir:

- Estado real de implementacion.
- Que es TDD estricto.
- Que es experimental.
- Como correr local.
- Como correr Vast.
- Coste estimado.
- Riesgos conocidos.

## 5. Comandos De Validacion Final

### Local

```powershell
python -m py_compile silexcode\model.py silexcode\train.py silexcode\bootstrap.py run_bootstrap.py
python -m pytest -q
```

### Vast smoke

```bash
cd /workspace/silexcode
git pull --ff-only
/venv/main/bin/python -m pip install -e . --no-build-isolation
chmod +x scripts/*.sh
CUDA_VISIBLE_DEVICES=0 OUTPUT_ADAPTER_LR=0.003 UPDATES_PER_LEVEL=50 EVAL_EVERY=10 VAL_SIZE=4 scripts/vast_output_adapter_bootstrap.sh bootstrap_output_adapter_smoke
```

### Vast status

```bash
scripts/vast_status.sh
tail -f runs/bootstrap_output_adapter_smoke.log
/venv/main/bin/python analyze_bootstrap_metrics.py runs/bootstrap_output_adapter_smoke/bootstrap_metrics.jsonl
```

### Stop

```bash
scripts/vast_stop_training.sh
```

## 6. Criterios Para Decir Proyecto 100% Completo

No se considera completo hasta cumplir todo esto:

- Tests locales pasan.
- Build Linux/Vast pasa.
- `vram_stress_test.py --steps 10` pasa.
- Stage 1 alcanza umbrales TDD durante 3 evaluaciones consecutivas.
- Stage 2 alcanza umbrales TDD durante 3 evaluaciones consecutivas.
- Stage 3 alcanza umbrales TDD durante 3 evaluaciones consecutivas.
- SSD-F genera candidatos y solo acepta codigo compilado y verificado.
- Checkpoint final se exporta.
- Checkpoint final se puede importar.
- Una generacion de prueba usa el checkpoint final.
- README documenta comandos reales.
- No hay secretos SSH commiteados.

## 7. Orden De Ejecucion Recomendado

1. Implementar native output adapter.
2. Anadir metadata estricta de checkpoint para output adapter.
3. Ejecutar bootstrap B0 rank 64 desde repo limpio.
4. Ejecutar bootstrap B1-B4 con resume.
5. Diagnosticar KFAC interno con metricas por matriz.
6. Probar AdamW interno como baseline.
7. Corregir KFAC o usar fallback justificado.
8. Stage 1 real.
9. Stage 2 real.
10. Teacher cache Stage 3.
11. Stage 3 oracle.
12. SSD-F.
13. Stress VRAM.
14. Documentacion final.
15. Commit y push final.

## 8. Riesgos Tecnicos

### Riesgo 1: El adaptador de salida calibra B0 pero no B1-B4

Mitigacion:

- LR schedule.
- Mas updates por nivel.
- Mezcla gradual de niveles.
- Rank 64 parece mejor que 256; probar rank 32 y 128.

### Riesgo 2: KFAC interno sigue sin aprender

Mitigacion:

- Diagnostico por gradiente.
- AdamW interno baseline.
- Diagonal-KFAC fallback.
- Ajuste de trust-region.

### Riesgo 3: Stage 3 tarda demasiado

Mitigacion:

- Stage 3 oracle antes de SSD.
- SSD refresh menos frecuente al principio.
- Validacion pequena durante exploracion, validacion completa solo en hitos.

### Riesgo 4: Salirse del TDD estricto

Mitigacion:

- Mantener flags experimentales desactivados por defecto.
- Documentar claramente que ruta es TDD y que ruta es bootstrap experimental.
- No tocar el backbone ternario ni `V=258`.

## 9. Estado De Vast Actual

Ultimos endpoints usados:

```text
direct: ssh -p 57606 root@47.186.29.91
proxy:  ssh -p 18711 root@ssh9.vast.ai
```

Clave local:

```text
D:\silexcode\vast_silex_ed25519
```

Estos archivos son privados y no deben commitearse:

```text
vast_known_hosts
vast_silex_ed25519
vast_silex_ed25519.pub
```

## 10. Primera Accion Cuando Se Ordene Ejecutar Todo

Empezar por:

1. Implementar `native output adapter`.
2. Probar equivalencia Python vs nativo.
3. Push.
4. Vast smoke.
5. Bootstrap B0-B4 con checkpoint.

No empezar Stage 1 real hasta que B1-B4 este probado o se haya decidido explicitamente saltarlo.
