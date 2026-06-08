# RBC — Correlaciones cruzadas

<img width="1518" height="863" alt="image" src="https://github.com/user-attachments/assets/3b6a9998-3aa0-4c5b-a682-b8754ef12af0" />


Aplicación interactiva desarrollada en **Python** y **Streamlit** para simular un modelo **Real Business Cycle (RBC)**, comparar sus versiones con trabajo divisible e indivisible, y analizar sus propiedades cíclicas mediante volatilidades, correlaciones contemporáneas y correlaciones cruzadas.

Esta aplicación nace como material computacional complementario del artículo:

**Monrocle Arribas, P. (2026). *Modelo Neoclásico de Ciclos Económicos (I)*. Documento de Trabajo / Working Paper.**

La herramienta permite trasladar parte de la discusión teórica del artículo a un entorno interactivo, de forma que el lector pueda modificar parámetros, simular el modelo y observar cómo cambian las propiedades cíclicas del producto, el consumo, la inversión, el capital y el trabajo.

## Aplicación en línea

La aplicación puede ejecutarse en Streamlit Community Cloud:

```text
https://rbc-model-5fbtaweurbgttr4qh6rohv.streamlit.app/
```

## Objetivo

El objetivo del proyecto es ofrecer una herramienta sencilla e interactiva para estudiar las propiedades cíclicas de un modelo RBC aplicado a la economía española.

En particular, la aplicación permite analizar:

* Modelo RBC con trabajo divisible.
* Modelo RBC con trabajo indivisible.
* Simulación de series macroeconómicas artificiales.
* Extracción del componente cíclico mediante filtro Hodrick-Prescott.
* Volatilidad relativa del consumo, capital, inversión y trabajo.
* Correlaciones contemporáneas con el producto.
* Correlaciones cruzadas entre el producto y las variables adelantadas o rezagadas.
* Comparación entre distintas especificaciones del mercado de trabajo.

La aplicación está pensada como apoyo visual y computacional al estudio de la macroeconomía cuantitativa, los modelos dinámicos de equilibrio general y la literatura de ciclos económicos reales.

## Relación con el artículo

El paper desarrolla una introducción rigurosa y divulgativa al modelo RBC como extensión estocástica del modelo neoclásico de crecimiento. En concreto, analiza cómo shocks tecnológicos agregados pueden generar fluctuaciones en variables macroeconómicas reales bajo un marco de equilibrio general competitivo y expectativas racionales.

El artículo aborda:

* el contexto histórico de la literatura RBC;
* la motivación del modelo;
* el problema del hogar representativo;
* el problema de la empresa representativa;
* el equilibrio general competitivo;
* el método de resolución mediante log-linealización;
* las condiciones de estabilidad de Blanchard-Kahn;
* la calibración para la economía española;
* la simulación estocástica del modelo;
* la comparación entre trabajo divisible e indivisible;
* la evaluación empírica del modelo frente a datos de España.

Este simulador no sustituye la explicación teórica del artículo. Su finalidad es complementarla mediante una herramienta interactiva que permita visualizar algunos de los mecanismos centrales del modelo.

## Funcionalidades principales

La aplicación permite:

* simular un modelo RBC con trabajo divisible;
* simular un modelo RBC con trabajo indivisible;
* modificar el parámetro de aversión relativa al riesgo `σ`;
* modificar el parámetro de curvatura del ocio `ψ`;
* seleccionar el número máximo de lags para las correlaciones cruzadas;
* aplicar el filtro Hodrick-Prescott al componente simulado de cada variable;
* calcular volatilidades relativas respecto al producto;
* calcular correlaciones contemporáneas con el producto;
* calcular correlaciones cruzadas para distintos valores de `k`;
* comparar gráficamente consumo, capital, inversión y trabajo;
* visualizar los resultados mediante gráficos interactivos de Plotly.

## Parámetros de referencia

La aplicación utiliza una calibración base para el modelo RBC:

```text
α       = 0.43
β       = 0.997
δ       = 0.011
ρ       = 0.89
σ_ε     = 0.005
l_ss    = 0.28
T       = 50,000
seed    = 7
```

donde:

* `α` representa la elasticidad del producto respecto al capital;
* `β` es el factor de descuento intertemporal;
* `δ` es la tasa de depreciación del capital;
* `ρ` mide la persistencia del shock tecnológico;
* `σ_ε` es la desviación típica de la innovación tecnológica;
* `l_ss` representa el nivel estacionario de trabajo;
* `T` es el número de periodos simulados.

El componente cíclico de las series se extrae mediante el filtro Hodrick-Prescott con:

```text
λ = 1600
```

valor habitual para datos trimestrales.

## Metodología

La aplicación resuelve una versión log-linealizada del modelo RBC mediante una representación en espacio de estados.

El estado del sistema está formado por el capital y el shock tecnológico:

```math
s_t =
\begin{pmatrix}
k_t \\
\theta_t
\end{pmatrix}
```

El shock tecnológico sigue un proceso autorregresivo de primer orden:

```math
\tilde{\theta}_{t+1} = \rho \tilde{\theta}_t + \varepsilon_{t+1}
```

donde:

```math
\varepsilon_t \sim N(0, \sigma_\varepsilon^2)
```

La solución dinámica se obtiene mediante una descomposición QZ ordenada, imponiendo las condiciones de estabilidad asociadas al criterio de Blanchard-Kahn.

Una vez resuelto el modelo, se simulan series para:

* producto;
* consumo;
* capital;
* inversión;
* trabajo;
* tecnología.

Posteriormente, se aplica el filtro Hodrick-Prescott a cada serie simulada y se calculan las correlaciones cruzadas:

```math
corr(y_t, x_{t+k})
```

para distintos valores de `k`.

## Modelos incluidos

### 1. Trabajo divisible

En la versión con trabajo divisible, el hogar elige de forma continua la cantidad de trabajo ofrecida. Esta especificación permite analizar el ajuste del trabajo en el margen intensivo.

La aplicación permite modificar dos parámetros de preferencias:

```text
σ — aversión relativa al riesgo asociada al consumo
ψ — curvatura del ocio
```

Estos parámetros afectan a la dinámica simulada del consumo, el trabajo, la inversión y el producto.

### 2. Trabajo indivisible

En la versión con trabajo indivisible, el ajuste del trabajo se interpreta de forma más cercana al margen extensivo: trabajar o no trabajar. Esta formulación está inspirada en Hansen (1985).

En esta versión:

```text
σ = 1
```

y no se introduce un parámetro libre de curvatura del ocio.

La comparación entre ambas especificaciones permite evaluar si el trabajo indivisible mejora la capacidad del modelo RBC para reproducir ciertos hechos cíclicos observados.

## Instalación

Para ejecutar la aplicación en local, primero clona el repositorio:

```bash
git clone https://github.com/pmonrocle/RBC-Model as rbc1
cd rbc1
```

Después instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Para lanzar la aplicación, ejecuta:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador.

## Dependencias

El proyecto utiliza las siguientes librerías:

```txt
streamlit>=1.35.0
numpy>=1.26.0
scipy>=1.13.0
plotly>=5.22.0
statsmodels>=0.14.0
pandas>=2.0.0
```

## Estructura básica del repositorio

```text
RBC-Cross-Correlations/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── paper/
    └── RBC_(I).pdf
```

## Interpretación económica y estadística

La aplicación permite observar de forma intuitiva cómo los parámetros y la especificación del mercado de trabajo afectan a la dinámica cíclica del modelo.

Algunos ejemplos:

* un shock tecnológico persistente genera movimientos conjuntos en producto, consumo, inversión y trabajo;
* la inversión suele ser más volátil que el producto;
* el consumo suele ser menos volátil que el producto;
* el capital muestra una dinámica más persistente;
* el trabajo divisible e indivisible generan diferentes patrones de volatilidad y comovimiento;
* las correlaciones cruzadas permiten analizar si una variable se adelanta, coincide o se retrasa respecto al producto;
* la comparación entre ambas especificaciones ayuda a evaluar la capacidad del modelo para aproximarse a los hechos estilizados del ciclo económico.

## Autor

**Pablo Monrocle Arribas**


## Referencia del paper

Monrocle Arribas, P. (2026). *Modelo Neoclásico de Ciclos Económicos (I)*. Documento de Trabajo / Working Paper.

## Cómo citar este repositorio

Si se utiliza esta aplicación como material de apoyo, puede citarse como:

```bibtex
@misc{monrocle2026rbccorrelations,
  author = {Monrocle Arribas, Pablo},
  title = {RBC — Correlaciones cruzadas},
  year = {2026},
  note = {Aplicación computacional complementaria del working paper Modelo Neoclásico de Ciclos Económicos (I)},
  url = {https://github.com/pmonrocle/RBC-Model/tree/main}
}
```

## Licencia

Este proyecto se publica con fines educativos y divulgativos.

El paper incluido en el repositorio es un documento de trabajo del autor. Para citar o distribuir el artículo, debe respetarse la indicación establecida en el propio documento.
