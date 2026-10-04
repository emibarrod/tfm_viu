# Registro de interacciones con herramientas de IA generativa

Índice cronológico del uso de IA generativa en el trabajo. Cada fila resume **qué se pidió**,
**cuándo** y **qué resultado verificable** produjo. Como explica el [README](README.md), es una
reconstrucción por etapas a partir del historial de *commits* del repositorio. Las reglas que
dirigieron el trabajo están en [`instrucciones_permanentes.md`](instrucciones_permanentes.md).

Salvo indicación contraria, la herramienta es **Claude** (Anthropic) mediante Claude Code sobre
el repositorio. El reparto de tareas es el que declaran las Secciones 1.6, 5.9 y 4.1 de la
memoria. La herramienta refactoriza, documenta, prueba, revisa y verifica. Las decisiones de
diseño, la hipótesis, el análisis y las conclusiones son del autor.

## 0. Búsqueda bibliográfica inicial (febrero – abril de 2026)

Antes de crear el repositorio.

| Periodo | Herramienta | Qué se pidió | Resultado |
|---|---|---|---|
| febrero – abril | **Perplexity** | Localizar literatura sobre predicción temprana de sepsis, fusión multimodal con datos clínicos y *benchmarks* públicos de UCI, y comprobar los metadatos de cada referencia | Corpus inicial de la revisión bibliográfica |
| febrero – abril | **Claude**, interfaz conversacional | Apoyo a la búsqueda y a la ordenación de la bibliografía inicial | Corpus ordenado por las preguntas de la revisión |

Perplexity siguió usándose durante todo el trabajo para ampliar la búsqueda y localizar
trabajos que la búsqueda manual no había recuperado (Sección 4.1).

## 1. Construcción (abril – julio de 2026)

El trabajo de esta etapa es del autor. Escribe el código del *pipeline* y de los modelos y
redacta los capítulos. La etapa de extracción parte de la implementación de referencia del
*benchmark* MIMIC-Sepsis (Sección 5.2). Claude interviene después, sobre lo ya escrito, para
ordenarlo, documentarlo y revisarlo.

| Fecha | Trabajo del autor | Asistencia de Claude |
|---|---|---|
| 2026-04-19 → 05-13 | Estructura del documento LaTeX y borradores de los capítulos 1 a 5 | Revisión del estilo de los borradores y del código LaTeX |
| 2026-05-19 → 05-31 | Extracción sobre MIMIC-IV adaptada de MIMIC-Sepsis, definición de la etiqueta Sepsis-3 y documento de decisión sobre esa definición | Refactorización y documentación del código de extracción y etiquetado |
| 2026-06-04 → 06-13 | Modelos clásicos de referencia, ablaciones por modalidad y primeros resultados | Refactorización y comentarios del código de los modelos |
| 2026-07-11 | Modelo profundo GRU de fusión intermedia | Revisión y documentación de su código y revisión del LaTeX de la memoria |

## 2. Corrección del *pipeline* y verificación de cifras (agosto de 2026)

Al contrastar el código con los datos apareció un sesgo en el muestreo de controles. El autor
decidió corregirlo y regenerar todos los resultados, aunque eso empeoraba las cifras, y amplió
el diseño experimental con un comparador clínico, la comorbilidad y pruebas estadísticas. Claude
refactorizó el código afectado, escribió sus pruebas y verificó cada cifra resultante.

| Fecha | Trabajo del autor | Asistencia de Claude | Resultado verificable |
|---|---|---|---|
| 2026-08-18 | Decide migrar el entorno y congelar los resultados existentes antes de tocar nada | Migración a `uv` + Python 3.11 | Entorno reproducible fijado en `uv.lock` |
| 2026-08-19 | Fija el emparejamiento de controles por *offset* y los arreglos de preprocesado | Refactorización de la construcción de controles y del preprocesado según ese criterio, con sus pruebas | Pruebas automáticas del emparejamiento |
| 2026-08-19 | Regenera la cohorte y entrena el modelo GRU | Verificación de que los resultados se reproducen entre ejecuciones | Cohorte de 9 278 estancias y métricas del brazo profundo |
| 2026-08-22 | Compara los criterios de umbral de Youden y F1 y adopta Youden | Refactorización del ciclo para guardar las probabilidades | Comparación de umbrales en `data/05_results/` |
| 2026-08-22 | Añade al diseño el comparador clínico SOFA/qSOFA, el índice de Charlson, las ablaciones de columnas y las pruebas pareadas | Integración en el ciclo de experimentos, con sus pruebas, y verificación de las métricas | Métricas canónicas en `data/05_results/` |
| 2026-08-23 | Rehace la interpretación de los capítulos 5 a 8 con los resultados nuevos | Verificación de cada cifra contra su artefacto y revisión del LaTeX | Memoria alineada con `data/05_results/` |
| 2026-08-23 | Decide pasar la bibliografía a APA 7 | Migración del LaTeX a biblatex con estilo APA | Bibliografía en APA 7 |
| 2026-08-23 | Rehace la comparación con el estado del arte | Contraste de cada celda de la tabla con la publicación original | Celdas sin respaldo corregidas |
| 2026-08-23 | Revisa la documentación del código | Sincronización de la documentación con lo que el código hace | Un README por etapa |
| 2026-08-23 | Añade la importancia por permutación al brazo profundo | Vectorización de dos funciones lentas, con resultados idénticos a los anteriores | Pruebas que comparan las dos versiones |

## 3. Revisión editorial y tipográfica (agosto de 2026)

Punto de partida: las notas del autor sobre el PDF compilado.

| Fecha | Fases | Qué se pidió | Resultado verificable |
|---|---|---|---|
| 2026-08-25 | A – B | Corregir el motor tipográfico del preámbulo y las cajas que se salen del margen | 0 cajas desbordadas |
| 2026-08-25 | D | Revisar acrónimos y glosario | Acrónimos expandidos en su primer uso |
| 2026-08-25 | E | Reverificar contra Crossref y DataCite las entradas existentes y las nuevas que eligió el autor | Discrepancias de metadatos corregidas |
| 2026-08-25 | F – H | Revisar terminología, registro y erratas | Terminología española coherente |

## 4. Revisión de contenido, capítulo a capítulo (agosto de 2026)

El punto de partida fueron las **notas de lectura del autor sobre el PDF compilado**. Cada fase
consistió en entregar esas notas y pedir que se aplicaran una a una, verificando que ninguna
cifra se movía.

| Fecha | Fase | Capítulo revisado |
|---|---|---|
| 2026-08-26 | I | Capítulos 1 y 2 |
| 2026-08-26 | J | Capítulo 3 |
| 2026-08-29 | K | Capítulo 4 |
| 2026-08-29 | L | Capítulo 5 |
| 2026-08-29 | M | Capítulo 6 |
| 2026-08-29 | N | Capítulo 7 |
| 2026-08-29 | O | Capítulo 8 y numeración del preliminar |

## 5. Revisión final antes del depósito (agosto – septiembre de 2026)

| Fecha | Fase | Qué se pidió | Resultado verificable |
|---|---|---|---|
| 2026-08-29 | P | Contrastar la memoria con el código y señalar cada desajuste | El grupo de control estaba descrito al revés. El 40,7 % de los controles cumple Sepsis-3 fuera del horizonte, y el autor lo cuantifica y corrige |
| 2026-08-29 | Q | Comprobar que cada fuente citada sostiene lo que el texto le atribuye y que el glosario no tiene entradas huérfanas | Atribuciones sin respaldo señaladas y corregidas por el autor |
| 2026-08-29 | R | Comprobar los requisitos formales de la guía (título, portada, resúmenes, ODS, declaración de IA, metodología de la revisión, objetivos) y revisar el estilo y el LaTeX de los textos del autor | Requisitos cubiertos |
| 2026-08-31 | T | Refactorizar el script de las figuras del capítulo 6 y verificar cada valor contra los CSV canónicos | Figuras trazables a sus datos |
| 2026-08-31 | U | Revisar la tipografía del documento compuesto | PDF sin cajas desbordadas |
| 2026-08-31 → 09-01 | — | Segunda pasada sobre contradicciones internas, trazabilidad de cifras, puntuación y pies de tabla | Contradicciones señaladas y resueltas por el autor |

## 6. Revisión antes de la entrega (septiembre – octubre de 2026)

| Fecha | Qué se hizo | Resultado verificable |
|---|---|---|
| 2026-09-22 → 10-02 | Revisión crítica del autor, capítulo a capítulo, con revisión del estilo y del LaTeX de los cambios | Capítulos 1 a 8 revisados |
| 2026-10-04 | Revisión de estilo, redundancia y errores de cifras y referencias. La herramienta presentó cada hallazgo con su prueba y el autor marcó uno a uno cuáles se aplicaban | Informe de hallazgos marcado por el autor |

## Nota sobre el reparto de responsabilidad

Las [instrucciones permanentes](instrucciones_permanentes.md) fijan dos reglas que explican el
reparto declarado en las Secciones 1.6 y 5.9 de la memoria:

1. **Una fase, verificarla, informar y parar.** Ninguna fase encadena con la siguiente sin
   revisión y aprobación explícita del autor.
2. **Las cifras mandan sobre el texto.** Cada número se comprueba contra los artefactos de
   `data/05_results/`, no contra lo que diga la memoria.

Las decisiones que fijan el diseño del estudio son del autor. Entre ellas están el criterio de
sospecha de infección, la construcción del grupo de control, la política sobre variables
acopladas a la etiqueta y el criterio de umbral.