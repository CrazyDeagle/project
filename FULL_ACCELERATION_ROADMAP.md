# SilexCode Full Acceleration Roadmap

Este documento define todo lo que falta mejorar antes de gastar otra corrida larga en GPU. El objetivo es evitar entrenar a ciegas: cada bloque debe tener una hipotesis, un cambio concreto y una prueba que diga si seguimos o paramos.

## 0. Estado Base Verificado

Runtime actual:

- CUDA/C++ extension compila en Windows y Linux Vast.
- `silex_forward_cuda` funciona como forward nativo.
- `silex_train_chunk_cuda` ejecuta forward, backward y K-FAC in-place.
- `vram_stress_test.py` ya paso bajo el presupuesto teorico de VRAM.
- Curriculum sintetico, SSD-F, checkpoint y tokenizer existen.
- Modo acelerado existe en `run_accelerated_curriculum.py`.

Resultado real medido en Vast RTX 5090:

```text
Stage 1 accelerated initial probe, 1000 updates:
NLL_4 validation: 12.5449 -> ~10.86
compile_pass: 0.0
target NLL_4 stage 1: <= 0.080
```

Conclusion:

```text
El sistema entrena, pero no converge suficientemente rapido.
No conviene gastar mas credito en una corrida larga sin mejorar aprendizaje/estabilidad.
```

## 1. Objetivos del Sprint Completo

Objetivo principal:

```text
Conseguir que un probe corto de 1000-5000 updates muestre una caida fuerte y estable de NLL_4 en Stage 1.
```

Criterio minimo para volver a Vast con una corrida seria:

```text
Stage 1 probe 1000 updates:
NLL_4 validation debe bajar claramente por debajo de 5.0.
```

Criterio fuerte:

```text
Stage 1 probe 5000 updates:
NLL_4 validation <= 1.0
latent_gain positivo estable
natural_norm sin explosiones repetidas
```

No lanzar `--require-thresholds` hasta cumplir lo anterior.

## 2. Mejora A: Diagnostico de Entrenamiento Mucho Mas Claro

Problema:

Ahora vemos `nll4`, `mono`, `latent_gain`, `natural_norm`, pero no sabemos si el modelo:

- aprende solo EOS/padding,
- aprende bytes frecuentes,
- falla por logits saturados,
- falla por update K-FAC demasiado agresivo,
- falla por poca capacidad de adaptadores,
- falla por generacion aunque el NLL mejore.

Implementar:

1. `analyze_curriculum_metrics.py` mejorado:
   - Resumen por stage.
   - Mejor `nll4`, ultimo `nll4`, pendiente aproximada.
   - Alertas si `natural_norm` explota.
   - Ratio de target tokens por chunk.

2. Log de entrenamiento ampliado:
   - `target_tokens`.
   - `packed_records`.
   - `tokens_per_second` o `updates_per_minute`.
   - `max_memory_allocated_mb`.
   - `natural_norm`.
   - `trust_chi` si el kernel lo expone.

3. Evaluacion de distribucion de bytes:
   - NLL por clases:
     - bytes codigo ASCII comunes: `def`, espacios, `\n`, digitos.
     - tags `<C>`, `</C>`, `<R>`, `<P>`.
     - EOS.
   - Top-1 token accuracy en posiciones con mascara.

Criterio de aceptacion:

```text
Un probe produce un resumen que dice por que no esta aprendiendo:
frecuencia, saturacion, padding, K-FAC o generacion.
```

## 3. Mejora B: Packing Acelerado Sin Padding Waste

Estado:

Ya aplicado parcialmente:

- First-fit packing.
- `candidate_multiplier`.
- Padding-loss desactivado por defecto.

Completar:

1. Packing por etapa:
   - Stage 1: maximizar numero de bloques `<R>...<C>code`.
   - Stage 2: no forzar muchos records si las trazas son largas.
   - Stage 3: priorizar records con codigo corto para mas ejemplos por chunk.

2. Balance de familias dentro de chunk:
   - No dejar que F0/F1 dominen por ser mas cortas.
   - En Stage 1, asegurar mezcla F0-F4.
   - En Stage 3, asegurar aparicion de F7/F8 aunque sean menos frecuentes.

3. Opciones CLI:
   - `--packing shortest`
   - `--packing balanced`
   - `--packing random-fit`

Criterio de aceptacion:

```text
Stage 1 chunks contienen >=3 records de media.
Stage 3 chunks contienen >=2 records de media cuando sea posible.
target_tokens/chunk aumenta sin meter padding artificial.
```

## 4. Mejora C: Estabilidad K-FAC y Trust Region

Problema medido:

En Vast se observaron saltos de `natural_norm`:

```text
100
45564
634536
328411
```

Esto sugiere updates demasiado agresivos o curvatura mal condicionada al inicio.

Implementar:

1. Exponer `chi` desde `block_kfac_step_param` y `silex_train_chunk_cuda`.
2. Loggear:
   - `natural_norm`.
   - `chi`.
   - numero de matrices actualizadas.
3. Warmup de K-FAC:
   - Durante N pasos, actualizar curvatura pero no parametros.
   - O usar LR reducido para las primeras N updates.
4. Clip adicional del update natural:
   - Mantener matematicamente separado como modo experimental.
   - CLI: `--kfac-warmup-updates`.
5. Grid corto de hiperparametros:
   - eta stage 1: `0.080`, `0.040`, `0.020`, `0.010`.
   - damping: `1e-3`, `3e-3`, `1e-2`.
   - delta: `1e-3`, `3e-4`, `1e-4`.

Criterio de aceptacion:

```text
natural_norm no explota repetidamente.
NLL validation baja mas suave que el baseline.
No aumenta VRAM por encima del limite.
```

## 5. Mejora D: Warm-Start Supervisado de Adaptadores

Problema:

El backbone determinista no es un modelo preentrenado semanticamente. Los adaptadores empiezan desde casi cero y solo pueden modificar localmente la red.

Objetivo:

Crear una fase bootstrap antes del curriculum largo que ponga los adaptadores en una zona util.

Opciones de implementacion:

### D1. Warm-start por tokens frecuentes

Entrenar solo Stage 1 con records muy cortos y repetidos:

- F0/F1 primero.
- Codigo corto.
- Sin trazas largas.
- Mascara solo en codigo.

Ventaja:

```text
Simple, compatible con TDD, usa el mismo loss.
```

Riesgo:

```text
Puede sobreajustar a templates simples.
```

### D2. Warm-start por curriculum micro-etapas

Subdividir Stage 1:

```text
1a: tags y estructura de funcion
1b: returns escalares
1c: loops simples
1d: if/while
```

El TDD final sigue teniendo Stage 1, pero antes se hace una fase auxiliar de arranque.

Ventaja:

```text
Reduce entropia inicial del problema.
```

Riesgo:

```text
Hay que evitar crear targets fuera de la gramatica sintetica.
```

### D3. Warm-start distillation desde renderer

Como tenemos el codigo oracle, podemos generar targets exactos y usar una mezcla con prompts mas cortos.

No cambia la matematica del modelo. Cambia el orden de presentacion de datos.

Criterio de aceptacion:

```text
Tras warm-start, Stage 1 validation NLL_4 inicial baja bastante frente a baseline.
Meta minima: pasar de ~12 a <8.
Meta fuerte: <5.
```

## 6. Mejora E: Evaluacion de Generacion Mas Barata

Problema:

`compile_pass` con generacion completa es caro y al inicio siempre da 0.

Implementar:

1. No ejecutar generacion completa en cada eval temprana.
2. Medir primero:
   - NLL.
   - token accuracy.
   - tag accuracy.
3. Activar `compile_pass` solo cuando:

```text
NLL_4 < 2.0
```

o con flag:

```text
--generate-eval-outputs
```

Criterio de aceptacion:

```text
Evaluaciones rapidas durante probes.
Generacion solo cuando el modelo tenga posibilidad real de compilar.
```

## 7. Mejora F: Checkpointing de Probes

Problema:

Si una prueba de 5000 updates mejora, hay que poder continuar sin perder credito.

Implementar:

1. Guardar checkpoint cada N evals:
   - modelo.
   - plastic adapters.
   - K-FAC opcional.
   - config del run.
2. CLI:
   - `--checkpoint-every-evals`
   - `--include-kfac`
3. Resume validado:
   - cargar `.silex`.
   - seguir desde stage/update correcto.

Criterio de aceptacion:

```text
Se puede parar Vast, reactivar y continuar desde el ultimo checkpoint.
```

## 8. Mejora G: Scripts Vast Reproducibles

Problema:

Estamos copiando comandos manualmente.

Crear:

1. `vast_setup.sh`
   - `git pull`
   - install editable
   - pytest minimo

2. `vast_probe_stage1.sh`
   - lanza probe stage 1.
   - redirige logs.
   - imprime PID.

3. `vast_status.sh`
   - `nvidia-smi`
   - proceso activo.
   - tail metrics.

4. `vast_stop_training.sh`
   - mata proceso de entrenamiento sin apagar instancia.

Criterio de aceptacion:

```text
Con 3 comandos se instala, lanza y monitorea.
```

## 9. Mejora H: Comparador de Experimentos

Crear:

```text
compare_runs.py runA/accelerated_metrics.jsonl runB/accelerated_metrics.jsonl
```

Debe mostrar:

- Mejor NLL por stage.
- Pendiente NLL.
- Tiempo aproximado.
- `target_tokens/update`.
- Si explota `natural_norm`.

Criterio de aceptacion:

```text
Podemos decidir objetivamente si una mejora vale mas credito.
```

## 10. Plan de Ejecucion Recomendado

Orden exacto:

1. Mejorar diagnosticos.
2. Completar packing balanceado.
3. Exponer `chi` y estabilizar K-FAC.
4. Implementar warm-start micro-stage.
5. Implementar checkpointing automatico.
6. Crear scripts Vast.
7. Correr local:

```powershell
python -m pytest -q
python -u run_accelerated_curriculum.py --output-dir runs\local_accel_dry --dry-run
python analyze_curriculum_metrics.py runs\local_accel_dry\accelerated_metrics.jsonl
```

8. Activar Vast solo para probe corto:

```bash
cd /workspace/silexcode
git pull
/venv/main/bin/python -m pip install -e . --no-build-isolation
/venv/main/bin/python -u run_accelerated_curriculum.py \
  --output-dir runs/stage1_probe_next \
  --stages 1 \
  --max-updates 1000 \
  --eval-every 100 \
  --val-size 16 \
  --max-records-per-chunk 8 \
  --candidate-multiplier 4
```

9. Decidir:

```text
Si NLL_4 > 5: no correr largo, seguir mejorando.
Si NLL_4 <= 5: correr 5000 updates.
Si NLL_4 <= 1: probar thresholds stage 1.
```

## 11. Riesgos Que No Hay Que Ocultar

1. El TDD presupone un checkpoint preentrenado para pesos ternarios. Sin checkpoint real, el backbone determinista es una inicializacion, no una inteligencia preentrenada.
2. Los adaptadores plasticos pueden no tener capacidad suficiente para aprender todo desde cero.
3. Mas GPU acelera steps, pero no arregla mala dinamica de aprendizaje.
4. El SSD-F no ayuda hasta que el modelo genere candidatos que compilen.
5. `compile_pass` sera 0 durante bastante tiempo si NLL sigue alto.

## 12. Definicion de Exito Antes de Gastar Mas Credito

No gastar mas credito en corrida larga hasta tener:

```text
pytest completo OK
dry-run OK
Vast setup reproducible OK
probe stage 1 1000 updates con NLL_4 <= 5
sin explosiones repetidas de natural_norm
checkpoint generado correctamente
```

Solo entonces:

```text
Lanzar 5000-20000 updates.
```

Y solo si eso confirma:

```text
Intentar curriculum estricto completo.
```
