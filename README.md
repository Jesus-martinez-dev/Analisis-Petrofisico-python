# 🛢️ PetroPy Suite™ - Plataforma de Análisis Petrofísico

**PetroPy Suite** es una aplicación interactiva desarrollada en Python y Streamlit para la evaluación cuantitativa de formaciones, visualización de registros geofísicos de pozo (Well Logs - Triple Combo) y cálculo de propiedades petrofísicas fundamentales de reservorios de hidrocarburos.

---

## 🚀 Características Principales

1. **Visualizador Multicanal de Registros (Composite Log / Triple Combo)**:
   - **Track 1 (Litología):** Gamma Ray (API) con sombreado de arenas limpias (amarillo) y lutitas (gris), más línea de corte dinámica.
   - **Track 2 (Resistividades):** Resistividad Profunda ($R_t$) y Somera ($R_{xo}$) en escala logarítmica (0.2 a 2000 $\Omega\cdot\text{m}$).
   - **Track 3 (Porosidad y Efecto Gas):** Curvas de Porosidad Neutrón ($\text{NPHI}$) y Densidad ($\text{RHOB}$) en escalas normalizadas con sombreado automático del **Crossover de Gas**.
   - **Track 4 (Evaluación de Fluidos):** Saturación de agua ($S_w$) y saturación de hidrocarburo ($S_h$).
   - **Track 5 (Pay Flags):** Discriminación visual de Arena Neta (*Net Sand*), Reservorio Neto (*Net Reservoir*) y Zona Productora Neta (*Net Pay*).

2. **Cálculos Petrofísicos Cuantitativos Rigurosos**:
   - **Volumen de Arcilla ($V_{sh}$):** Modelos lineal, Larionov (Rocas Terciarias), Larionov (Rocas Pre-Terciarias consolidadas) y Stieber.
   - **Porosidad Total y Efectiva ($\phi_D, \phi_{ND}, \phi_e$):** Corrección por matriz mineralógica (Arenisca, Caliza, Dolomita) y fluido (agua dulce o lodo salado).
   - **Saturación de Agua ($S_w$):** Ecuación clásica de Archie con parámetros ajustables ($a$, $m$, $n$, $R_w$).
   - **Permeabilidad ($K$):** Estimación de capacidad de flujo mediante el modelo petrofísico de Timur.
   - **Zonificación y Pay:** Filtros configurables de $V_{sh}$, $\phi_e$ y $S_w$.

3. **Gráficos Cruzados Petrofísicos (Crossplots)**:
   - **Pickett Plot ($\log R_t$ vs $\log \phi_e$):** Isolíneas teóricas de $S_w$ para calibrar $R_w$ y el exponente de cementación $m$.
   - **Neutron-Density Crossplot ($\text{NPHI}$ vs $\text{RHOB}$):** Identificación mineralógica de matriz (Arenisca, Caliza, Dolomita) y detección de fluidos ligeros (gas).
   - **Buckles Plot ($\phi_e$ vs $S_w$):** Evaluación del volumen aparente de agua ($BVW$) y condición de saturación irreducible.

4. **Entrada y Salida de Datos**:
   - Simulador geológico sintético multizona calibrado (Casquete de gas, zona de petróleo, acuífero y lutitas de sello).
   - Capacidad para cargar registros propios en formato `.csv` (ejemplo incluido: `ejemplo_pozo.csv`).
   - Exportación completa de resultados y curvas calculadas en `.csv`.

---

## 🛠️ Instalación y Ejecución

### Opción 1: Ejecución Rápida
Ejecuta el script automatizado:
```bash
./run.sh
```

### Opción 2: Ejecución Manual
```bash
# 1. Crear y activar entorno virtual (opcional pero recomendado)
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (por defecto en `http://localhost:8501`).

---

## 📁 Estructura del Proyecto

```text
├── app.py                     # Aplicación principal completa en Streamlit
├── requirements.txt           # Dependencias requeridas (streamlit, pandas, numpy, matplotlib, scipy)
├── run.sh                     # Script de inicio rápido
├── ejemplo_pozo.csv           # Archivo de prueba con datos reales/sintéticos
└── README.md                  # Documentación del proyecto
```
