# Instrucciones permanentes del proyecto

Reglas de trabajo que acompañaron a Claude Code en todas las sesiones sobre el repositorio del
TFM. Se reproducen aquí sin las rutas de ficheros internos de trabajo, que no forman parte del
repositorio público.

---

## El proyecto

TFM sobre predicción temprana de sepsis (Sepsis-3, 6 h de antelación) sobre MIMIC-IV v3.1. El
*pipeline* está en `multi_modality_code/`, los resultados en `data/05_results/` y la memoria
LaTeX en `tfm_latex/`. El entorno es **uv + Python 3.11**. Usa siempre `uv run python …`.

## Qué te pido y qué no

Tu trabajo en este repositorio se limita a esto:

- refactorizar el *pipeline* de extracción, etiquetado y exportación;
- refactorizar y documentar el código y escribir y mantener sus pruebas automáticas;
- revisar el estilo de la memoria y su código LaTeX;
- verificar cada cifra del documento contra los artefactos de `data/05_results/`;
- en la revisión bibliográfica, cribar los trabajos por los criterios de calidad sobre el texto
  completo y contrastar cada cifra citada con su publicación original.

Las decisiones de diseño son mías y te llegan tomadas. Son la definición de la cohorte y de la
etiqueta, las ventanas temporales, la construcción del grupo de control, las ablaciones, las
métricas y el criterio de umbral. También son mías la hipótesis, el análisis de los resultados y
las conclusiones. Si al refactorizar o verificar encuentras algo que afecta a una de esas
decisiones, o que cambia cómo se interpreta un resultado, no lo arregles por tu cuenta. Para,
explícame qué has encontrado y con qué prueba, y espera a que decida.

## Las cifras mandan sobre el texto

Ninguna cifra se escribe de memoria ni se copia de otro capítulo. Cada número de la memoria se
lee del artefacto que lo produce en `data/05_results/`. Si texto y artefacto discrepan, gana el
artefacto y la discrepancia se informa.

En las fases que solo tocan la memoria, comprueba al terminar que la secuencia de números de
cada `.tex` es la misma que antes, salvo donde se pidió cambiarla, y que el código y los datos no
se han movido.

## Trabajo por fases

Cada bloque de trabajo tiene una especificación por fases que apruebo antes de empezar. Ejecuta
**una fase, verifícala, infórmame y para**. No encadenes con la siguiente sin mi visto bueno
explícito.

Verificar una fase quiere decir, según lo que toque:

- en código, que las pruebas pasan y que los resultados se reproducen;
- en la memoria, que compila sin errores, sin referencias ni citas indefinidas y sin cajas
  desbordadas.

Al cerrar una fase, deja anotado qué se hizo, cómo se verificó y qué queda pendiente de mi
decisión, y haz un *commit* de la fase.

## Bibliografía

No atribuyas a una fuente nada que no hayas comprobado en ella. Cada entrada se verifica contra
Crossref o DataCite, y cada cifra citada se contrasta con la publicación original. Si una
afirmación de la memoria no tiene fuente o la fuente no la sostiene, señálalo. Qué hacer con ella
lo decido yo.
