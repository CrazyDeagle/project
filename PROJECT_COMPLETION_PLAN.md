# SilexCode-T18.6B-R64: Plan de Cierre al 100%

Este documento resume el estado actual del proyecto y define, paso por paso, lo que falta para cerrar completamente el TDD de arquitectura, entrenamiento, checkpoints y curriculum.

Nota de edicion: este archivo usa ASCII en el contenido operativo para evitar problemas de consola en Windows.

## Estado de Cierre Ejecutado

Fecha de cierre local: 2026-05-01.

Resultado final de este plan:

- Fallback nativo packed para checkpoints reales: COMPLETADO.
- Smoke curriculum end-to-end: COMPLETADO.
- Checkpoint roundtrip completo: COMPLETADO.
- Stress VRAM determinista: COMPLETADO.
- Stress VRAM packed: COMPLETADO.
- SSD-F smoke: COMPLETADO.
- Teacher cache smoke: COMPLETADO.
- Curriculum dry-run: COMPLETADO.
- CLI `run_curriculum.py --dry-run`: COMPLETADO.
- README operativo: COMPLETADO.

Validaciones ejecutadas:

```text
python -m pytest -q
28 passed, 1 skipped in 3.37s
```

```text
$env:SILEX_RUN_FULL_CHECKPOINT_TEST='1'; python -m pytest -q tests\test_checkpoint_roundtrip.py
2 passed in 13.72s
```

```text
python -u vram_stress_test.py --steps 10 --mode deterministic
max_memory_allocated = 7253.181640625 MB
```

```text
python -u vram_stress_test.py --steps 1 --mode packed
max_memory_allocated = 7253.181640625 MB
```

```text
python -u curriculum_smoke_test.py
curriculum_smoke_test=PASS
```

```text
python -u teacher_cache_smoke_test.py
teacher_cache_smoke_test=PASS
```

```text
python -u ssd_smoke_test.py
ssd_smoke_test=PASS
```

```text
python -u curriculum_dry_run.py
curriculum_dry_run=PASS
```

```text
python -u run_curriculum.py --dry-run --output-dir runs\final_dry_run
run_curriculum=PASS
```

Nota de rendimiento: el modo packed respeta checkpoints arbitrarios, pero es mucho mas lento que el fast path determinista. La ultima medicion local fue un paso packed en aproximadamente 292 s. Esta ruta existe por correccion; el rendimiento principal sigue siendo el modo determinista FWHT.

## Estado Actual Validado

### Runtime CUDA/PyTorch

- Kernels CUDA/C++ de `TLinear` packed implementados:
  - `tlinear_forward_kernel`
  - `tlinear_backward_input_kernel`
- Packing ternario obligatorio implementado:
  - 5 trits por byte
  - `S_5(d_in)` alineado a 32 bytes
  - LUT constante de trits en GPU
- Byte-level tokenizer implementado con:
  - `V=258`
  - bytes `0..255`
  - `BOS=256`
  - `EOS=257`
- Modelo PyTorch implementado:
  - Backbone recurrente de 64 capas
  - Embedding/head atados
  - RMSNorm
  - Mixer recurrente
  - Latent reasoner
  - Adaptadores plasticos `A_m/B_m/A_f/B_f`
- Ruta de inferencia nativa C++ implementada:
  - `silex_forward_cuda`
- Ruta de entrenamiento nativa C++ implementada:
  - `silex_train_chunk_cuda`
  - workspace estatico
  - backward nativo
  - actualizacion K-FAC in-place
- Curriculum sintetico implementado:
  - `dataset.py`
  - `train.py`
  - renderers F0-F8
  - validacion AST restrictiva
  - compilacion segura
  - SSD-F con filtro de tests
- Muestreo para SSD-F implementado:
  - `generate_bytes`
  - temperatura
  - top-k
  - top-p
  - seed local determinista
- Checkpoint `.silex` implementado:
  - export/import de `state_dict`
  - metadatos
  - K-FAC opcional
  - marca `deterministic_backbone`

### Validaciones Recientes

Ultima bateria de tests:

```powershell
python -m pytest -q
```

Resultado:

```text
27 passed in 3.21s
```

Ultima prueba de VRAM:

```powershell
python -u vram_stress_test.py --steps 10
```

Resultado:

```text
max_memory_allocated = 7253.181640625 MB
```

Limite teorico TDD:

```text
7256.25598526001 MB
```

Estado:

```text
7253.181640625 < 7256.25598526001
```

## Riesgo Tecnico Cerrado Recientemente

El fast path FWHT usa la formula determinista del TDD para reconstruir los pesos sin leer `Wpack`.

Esto es correcto solo para el backbone inicializado deterministicamente. Para checkpoints reales con pesos packed arbitrarios, ese fast path no puede usarse.

Estado actual:

- El modelo tiene `deterministic_backbone`.
- `forward_native()` falla explicitamente si `deterministic_backbone=False`.
- `train_chunk_cuda()` falla explicitamente si `deterministic_backbone=False`.
- Los checkpoints `.silex` guardan/importan esta marca.
- Los checkpoints por carpeta se marcan como no deterministas.

Resultado: no puede haber uso silenciosamente incorrecto del fast path con checkpoints reales.

## Lo Que Falta Para Cerrar el 100%

## 1. Fallback Nativo Packed Para Checkpoints Reales

### Objetivo

Permitir que `silex_forward_cuda` y `silex_train_chunk_cuda` funcionen correctamente con pesos packed arbitrarios cargados desde checkpoint, no solo con pesos deterministas.

### Problema Actual

El runtime nativo optimizado usa:

- `deterministic_tlinear_forward_native`
- `deterministic_tlinear_forward_multi_native`
- `deterministic_tlinear_backward_input_native`

Estas funciones ignoran `Wpack` y reconstruyen pesos desde `(layer, matrix_id)`.

Eso es correcto para inicializacion determinista, pero incorrecto para checkpoints entrenados/preentrenados arbitrarios.

### Implementacion Requerida

1. Anadir parametro C++ a:
   - `silex_forward_cuda`
   - `silex_train_chunk_cuda`
   - `silex_train_chunk_cuda_update`

   Parametro:

   ```cpp
   bool deterministic_backbone
   ```

2. Crear helper C++ de TLinear seleccionable:

   ```cpp
   torch::Tensor silex_tlinear_forward_select(
       torch::Tensor X,
       torch::Tensor Wpack,
       torch::Tensor alpha,
       int d_in,
       int d_out,
       int layer,
       int matrix_id,
       bool deterministic_backbone
   );
   ```

   Si `deterministic_backbone=True`:

   ```cpp
   deterministic_tlinear_forward_native(...)
   ```

   Si `deterministic_backbone=False`:

   ```cpp
   tlinear_forward_native(X, Wpack, alpha, d_in, d_out)
   ```

3. Crear helper equivalente para backward-input:

   ```cpp
   torch::Tensor silex_tlinear_backward_input_select(
       torch::Tensor dY,
       torch::Tensor Wpack,
       torch::Tensor alpha,
       int d_in,
       int d_out,
       int layer,
       int matrix_id,
       bool deterministic_backbone
   );
   ```

4. Sustituir todas las llamadas directas deterministas en:
   - forward backbone
   - latent forward
   - latent backward
   - backward de 64 capas

5. Para multi-output determinista:
   - Mantener `deterministic_tlinear_forward_multi_native` si `deterministic_backbone=True`.
   - Si `False`, llamar a `tlinear_forward_native` varias veces con cada `Wpack`.

6. Pasar `self.deterministic_backbone` desde `model.py`.

7. Quitar el error duro de `_require_deterministic_native_runtime()` solo cuando el fallback packed este implementado y validado.

### Tests Requeridos

1. Test de forward nativo con pesos cero packed no deterministas:
   - construir modelo o llamada C++ con `deterministic_backbone=False`
   - usar `Wpack` todo cero logico
   - verificar logits y state esperados

2. Test de equivalencia:
   - `forward_native(... deterministic_backbone=False)`
   - contra `forward_python_reference`
   - con pesos packed pequenos/estructurados

3. Test de seguridad:
   - cargar checkpoint no determinista
   - verificar que el modelo no usa fast path determinista

### Validacion

```powershell
python -m pytest -q
```

Si hay GPU suficiente:

```powershell
python -u vram_stress_test.py --steps 1 --checkpoint-mode packed
```

## 2. Smoke Test Corto del Curriculum End-to-End

### Objetivo

Validar que el pipeline completo arranca antes de lanzar entrenamiento largo.

### Implementacion Requerida

Crear script:

```text
curriculum_smoke_test.py
```

Debe ejecutar:

1. Instanciar modelo determinista.
2. Instanciar `BlockKFACOptimizer`.
3. Generar un registro por etapa:
   - stage 1
   - stage 2
   - stage 3
4. Construir:
   - `input_ids`
   - `labels`
   - `loss_mask`
5. Ejecutar un paso nativo:

   ```python
   model.train_chunk_cuda(...)
   ```

6. Confirmar que devuelve metricas:
   - `nll`
   - `nll4`
   - `mono`
   - `latent_gain`
   - `natural_norm`
   - `updated_matrices`

7. Confirmar que no hay NaN/Inf.

### Validacion

```powershell
python -u curriculum_smoke_test.py
```

Resultado esperado:

```text
stage=1 ok
stage=2 ok
stage=3 ok
curriculum_smoke_test=PASS
```

## 3. Checkpoint Roundtrip Completo

### Objetivo

Garantizar que un `.silex` exportado se puede importar y produce el mismo estado.

### Implementacion Requerida

Crear test:

```text
tests/test_checkpoint_roundtrip.py
```

Debe:

1. Instanciar modelo determinista.
2. Exportar `.silex` temporal.
3. Instanciar segundo modelo.
4. Importar `.silex`.
5. Comparar:
   - todos los tensors del `state_dict`
   - `deterministic_backbone`
   - presencia/ausencia de K-FAC

Opcional:

6. Export/import con K-FAC incluido.

### Validacion

```powershell
python -m pytest -q tests\test_checkpoint_roundtrip.py
```

## 4. Stress Test de VRAM en Dos Modos

### Objetivo

Validar memoria para:

- modo determinista rapido
- modo packed checkpoint

### Implementacion Requerida

Extender:

```text
vram_stress_test.py
```

Con argumentos:

```powershell
--steps 10
--mode deterministic
--mode packed
```

Modo determinista:

- usa fast path FWHT
- limite esperado:

```text
<= 7256.25598526001 MB
```

Modo packed:

- usa pesos `Wpack`
- puede ser mas lento
- no debe superar 8GB
- idealmente debe mantenerse cerca del limite TDD

### Validacion

```powershell
python -u vram_stress_test.py --steps 10 --mode deterministic
python -u vram_stress_test.py --steps 1 --mode packed
```

## 5. SSD-F Runtime Smoke

### Objetivo

Validar que `build_ssd_pool` funciona con `generate_bytes` y el verificador AST/tests.

### Implementacion Requerida

Crear script:

```text
ssd_smoke_test.py
```

Debe:

1. Instanciar modelo.
2. Generar pocos problemas stage 3:

   ```python
   indices = [40_000_000, 40_000_001]
   ```

3. Ejecutar `build_ssd_pool` con constantes reducidas, o anadir parametros opcionales para:
   - candidates per problem
   - max accepted
   - max bytes

4. Confirmar que:
   - no lee internet
   - no usa datasets externos
   - candidatos invalidos se rechazan
   - candidatos aceptados pasan compile + unit tests

### Validacion

```powershell
python -u ssd_smoke_test.py
```

No es obligatorio que acepte muestras con un modelo no entrenado. Si debe completar sin errores.

## 6. Teacher Cache Smoke Para Etapa 3

### Objetivo

Validar que la precomputacion teacher funciona registro a registro sin segunda copia VRAM del modelo.

### Implementacion Requerida

Crear script:

```text
teacher_cache_smoke_test.py
```

Debe:

1. Instanciar modelo.
2. Ejecutar:

   ```python
   precompute_stage3_teacher_cache(...)
   ```

   con 2-4 indices.

3. Abrir cache.
4. Hacer lookup.
5. Verificar shape:

   ```text
   [511, 258]
   ```

6. Verificar dtype guardado:

   ```text
   float16 en disco
   ```

### Validacion

```powershell
python -u teacher_cache_smoke_test.py
```

## 7. Mini Curriculum Dry Run

### Objetivo

Ejecutar el bucle real de curriculum con limites minimos para detectar fallos de integracion.

### Implementacion Requerida

Modificar `train_curriculum` para aceptar overrides opcionales:

```python
def train_curriculum(
    model,
    kfac_optimizer,
    output_dir: str,
    *,
    max_updates_override: dict[int, int] | None = None,
    eval_every_updates_override: int | None = None,
    val_size_override: int | None = None,
    require_thresholds: bool = True,
):
```

Regla:

- En entrenamiento real, defaults del TDD intactos.
- En smoke test, usar overrides.

Crear script:

```text
curriculum_dry_run.py
```

Configuracion:

```python
max_updates_override = {1: 2, 2: 2, 3: 2}
eval_every_updates_override = 1
val_size_override = 2
require_thresholds = False
```

### Validacion

```powershell
python -u curriculum_dry_run.py
```

Resultado esperado:

```text
stage=1 dry_run_complete
stage=2 dry_run_complete
stage=3 dry_run_complete
curriculum_dry_run=PASS
```

## 8. CLI de Entrenamiento Real

### Objetivo

Tener una entrada clara para iniciar entrenamiento largo.

### Implementacion Requerida

Crear:

```text
run_curriculum.py
```

Argumentos:

```powershell
--output-dir runs\silex_curriculum_001
--resume checkpoint.silex
--include-kfac
--stage 1
--max-updates
--eval-every
--val-size
--dry-run
```

Debe:

1. Crear directorio de salida.
2. Instanciar modelo.
3. Cargar checkpoint si `--resume`.
4. Instanciar K-FAC.
5. Ejecutar `train_curriculum`.
6. Guardar metricas JSONL.
7. Guardar checkpoints por etapa.

### Validacion

```powershell
python -u run_curriculum.py --dry-run --output-dir runs\dry_run
```

## 9. Documentacion de Uso

### Objetivo

Que el proyecto sea ejecutable sin recordar comandos sueltos.

### Implementacion Requerida

Crear o actualizar:

```text
README.md
```

Debe incluir:

1. Requisitos:
   - Visual Studio Developer Command Prompt
   - CUDA
   - PyTorch CUDA
   - `--no-build-isolation`

2. Instalacion:

   ```powershell
   pip install -e . --no-build-isolation
   ```

3. Tests:

   ```powershell
   python -m pytest -q
   ```

4. Stress VRAM:

   ```powershell
   python -u vram_stress_test.py --steps 10
   ```

5. Smoke curriculum:

   ```powershell
   python -u curriculum_dry_run.py
   ```

6. Entrenamiento real:

   ```powershell
   python -u run_curriculum.py --output-dir runs\silex_curriculum_001
   ```

7. Nota sobre checkpoints:
   - determinista: fast path FWHT
   - checkpoint packed arbitrario: fallback packed nativo requerido

## 10. Validacion Final Para Declarar 100%

Antes de declarar el proyecto cerrado, ejecutar en este orden:

```powershell
python -m pytest -q
```

```powershell
python -u vram_stress_test.py --steps 10 --mode deterministic
```

```powershell
python -u curriculum_smoke_test.py
```

```powershell
python -u teacher_cache_smoke_test.py
```

```powershell
python -u ssd_smoke_test.py
```

```powershell
python -u curriculum_dry_run.py
```

```powershell
python -u run_curriculum.py --dry-run --output-dir runs\final_dry_run
```

Opcional si se implementa packed mode completo:

```powershell
python -u vram_stress_test.py --steps 1 --mode packed
```

## Criterio de Cierre al 100%

El proyecto se considera cerrado al 100% cuando:

- Todos los tests pasan.
- El stress determinista de 10 pasos queda por debajo de `7256.25598526001 MB`.
- El checkpoint roundtrip pasa.
- El curriculum dry run pasa por las 3 etapas.
- SSD-F smoke completa sin errores.
- Teacher cache smoke completa sin errores.
- `run_curriculum.py --dry-run` completa.
- El runtime nativo no puede usar el fast path determinista con pesos packed arbitrarios salvo que el checkpoint se marque explicitamente determinista.
- Si se requiere soporte para checkpoints arbitrarios acelerados, el fallback packed nativo debe estar implementado y validado.

## Comando Para Pedir el Cierre Completo

Cuando quieras que se ejecute todo este plan, pide:

```text
Implementa todo lo que falta en PROJECT_COMPLETION_PLAN.md y no pares hasta que pasen todas las validaciones finales.
```
