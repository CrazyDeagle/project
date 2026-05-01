# SilexCode Acceleration Status

## Estado Actual

El runtime CUDA/C++ está operativo:

- `silex_train_chunk_cuda` ejecuta forward, backward y update K-FAC nativo.
- El stress test previo en RTX 5090 completó 10 pasos bajo el presupuesto de VRAM.
- El modo acelerado empaqueta varios registros sintéticos por chunk de 512 tokens.
- La ruta TDD estricta sigue disponible en `run_curriculum.py`.

## Resultado Medido en Vast RTX 5090

Probe stage 1, 1000 updates, runner acelerado inicial:

| Update | Validation `NLL_4` |
| -----: | -----------------: |
|      0 |            12.5449 |
|    300 |            11.2918 |
|    600 |            11.0405 |
|    800 |            10.8570 |
|    900 |            11.0238 |

Umbral normativo stage 1:

```text
NLL_4 <= 0.080
compile_pass >= 0.995
```

Conclusión: el entrenamiento aprende algo, pero no lo suficiente para justificar una corrida larga todavía.

## Mejora Aplicada Después del Probe

Commit `5a6b207` cambia sólo el runner acelerado:

- Usa first-fit packing sobre más candidatos.
- Puede empaquetar más registros reales por chunk.
- Desactiva por defecto la pérdida sobre EOS de padding en `run_accelerated_curriculum.py`.
- Conserva el modo estricto mediante `--include-padding-loss`.

Validación local:

```text
python -m pytest -q
32 passed, 1 skipped
```

Dry-run local con la nueva ruta:

```text
run_accelerated_curriculum=PASS
stage 1 packed_records=3
stage 1 target_tokens=252
updated_matrices=64
```

## Qué Puede Mejorar Bastante

1. Señal de pérdida útil por step.
   La mejora aplicada reduce aprendizaje desperdiciado en padding y mete más ejemplos por chunk.

2. Estabilidad K-FAC.
   En Vast se observaron saltos grandes de `natural_norm`, por ejemplo `45564`, `634536`.
   Esto sugiere que el update natural puede estar demasiado agresivo para el arranque aleatorio.

3. Inicialización de adaptadores.
   El backbone determinista no es un pretraining semántico. Los adaptadores empiezan intentando aprender código desde una representación casi aleatoria.
   El siguiente salto real debe venir de un warm-start supervisado o de una fase bootstrap más estable.

4. Métrica rápida de decisión.
   No hay que lanzar 200k updates hasta que un probe de 1k-5k updates baje `NLL_4` de forma fuerte.

## Próximo Objetivo Medible

Reactivar Vast sólo para esta prueba:

```bash
cd /workspace/silexcode
git pull
/venv/main/bin/python -m pip install -e . --no-build-isolation
/venv/main/bin/python -u run_accelerated_curriculum.py \
  --output-dir runs/accelerated_signal_stage1_1000 \
  --stages 1 \
  --max-updates 1000 \
  --eval-every 100 \
  --val-size 16 \
  --max-records-per-chunk 8 \
  --candidate-multiplier 4
```

Analizar:

```bash
/venv/main/bin/python analyze_curriculum_metrics.py \
  runs/accelerated_signal_stage1_1000/accelerated_metrics.jsonl
```

Decisión:

- Si `NLL_4` sigue por encima de `~5`, no correr curriculum largo.
- Si `NLL_4` baja fuerte hacia `<=1`, probar 5k updates.
- Sólo lanzar thresholds estrictos cuando el probe corto muestre tendencia realista hacia `0.080`.
