"""
==============================================================================
vid_simulator_v2.py — Simulador de Enfermedades y Plagas en la Vid (v2)
==============================================================================
Versión refactorizada que sustituye el motor climático estocástico por datos 
meteorológicos históricos REALES de estaciones de Castilla y León.

Cambios respecto a v1:
  - ELIMINADO: Motor de clima sintético (Markov, radiación simulada)
  - ELIMINADO: Variables con data leakage (CO2_ppm, VOC_ppb, Temp_Hoja_C)
  - AÑADIDO:   Carga y muestreo de clima real (Open-Meteo)
  - CONSERVADO: Todos los motores biológicos de enfermedad (11 funciones)
  - CONSERVADO: Post-procesado (recorte inteligente de series)

El simulador actúa como un "etiquetador biológico": dado un bloque de clima 
real, calcula cómo habría evolucionado una enfermedad bajo esas condiciones.

Uso:
    python vid_simulator_v2.py
    
Pre-requisitos:
    python descarga_clima_historico.py
    python data_cleaner.py
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import gc
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Fijar semilla para reproducibilidad
np.random.seed(42)

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
N_SERIES_A_GENERAR = 1000   # Número editable de series a generar
PASOS_POR_DIA = 24

# 1.2 RUTAS
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

# Entrada: Datos climáticos reales (preprocesados por data_cleaner.py)
CLIMA_DIR = PROJECT_ROOT / config_yaml['paths']['clima_clean_dir']

# Salida: Dataset final
OUTPUT_FILE = PROJECT_ROOT / config_yaml['paths']['raw_data']
OUTPUT_DIR = OUTPUT_FILE.parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1.3 MAPEO DE CLASES
# ==============================================================================
CLASES_SALIDA = [
    "HEALTHY", "MILDIU", "BLACK_ROT", "OIDIO", "ERINOSIS",
    "BOTRYTIS", "LOBESIA", "ALTICA", "EMPOASCA", "RED_MITE", "ESCA"
]

# 1.4 VENTANAS DE INICIO POR ENFERMEDAD (meses válidos para muestreo)
# ==============================================================================
# Cada enfermedad se asocia a los meses del año en los que tiene sentido
# iniciar la simulación, alineados con su biología real.
MESES_INICIO = {
    "HEALTHY":   (3, 7),   # Siempre (cualquier momento del ciclo)
    "MILDIU":    (3, 5),   # Marzo - Mayo (necesita lluvia primaveral)
    "BLACK_ROT": (3, 5),   # Marzo - Mayo (idem)
    "BOTRYTIS":  (4, 6),   # Abril - Junio (humedad primaveral/estival)
    "OIDIO":     (5, 7),   # Mayo - Julio (calor seco)
    "ESCA":      (5, 8),   # Mayo - Agosto (estrés hidráulico estival)
    "LOBESIA":   (3, 5),   # Marzo - Mayo (primera generación GDD)
    "ALTICA":    (3, 5),   # Marzo - Mayo (emergencia por GDD)
    "EMPOASCA":  (5, 7),   # Mayo - Julio (calor)
    "RED_MITE":  (5, 7),   # Mayo - Julio (calor seco)
    "ERINOSIS":  (3, 4),   # Marzo - Abril (solo en brotación)
}


# ==============================================================================
# 2. CARGA Y MUESTREO DE CLIMA REAL
# ==============================================================================

def cargar_datos_clima_real(clima_dir: Path) -> dict:
    """
    Carga todos los archivos Parquet de clima limpio en un diccionario.
    Retorna: {parcela_id: DataFrame}
    """
    archivos = sorted(clima_dir.glob("clima_*_clean.parquet"))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos de clima limpio en {clima_dir}.\n"
            "Ejecuta primero:\n"
            "  python descarga_clima_historico.py\n"
            "  python data_cleaner.py"
        )

    datasets = {}
    for f in archivos:
        # Extraer ID: "clima_ribera_del_duero_clean.parquet" → "ribera_del_duero"
        parcela_id = f.stem.replace("clima_", "").replace("_clean", "")
        df = pd.read_parquet(f)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.sort_values("Fecha").reset_index(drop=True)
        datasets[parcela_id] = df
        print(f"  📂 {parcela_id}: {len(df)} registros ({df['Fecha'].min().year}-{df['Fecha'].max().year})")

    return datasets


def muestrear_clima_real(datasets_clima: dict, dias_necesarios: int, agente: str) -> tuple:
    """
    Selecciona aleatoriamente una parcela y una ventana temporal continua
    de clima real, coherente con la estación biológica de la enfermedad.

    Incluye validación de condiciones iniciales: las primeras 24h deben
    representar un estado climático "sano" (sin eventos extremos que 
    dispararían inmediatamente una enfermedad en el simulador).

    Returns:
        (bloque_clima: DataFrame, parcela_id: str, fecha_inicio: Timestamp)
    """
    n_steps = dias_necesarios * PASOS_POR_DIA
    mes_inicio_min, mes_inicio_max = MESES_INICIO.get(agente, (3, 7))

    # Intentar hasta encontrar una ventana válida
    max_intentos = 500
    parcelas = list(datasets_clima.keys())

    # Umbrales de condiciones iniciales "sanas" (primeras 24h)
    # Representan el rango normal para una vid al inicio de temporada
    VENTANA_VALIDACION_H = 24   # Horas a comprobar
    T_MIN_SANA = 3.0            # No heladas severas
    T_MAX_SANA = 38.0           # No ola de calor extrema
    LLUVIA_MAX_ACUM_24H = 30.0  # No en medio de un diluvio (mm en 24h)
    HR_MIN_SANA = 15.0          # No sequía extrema
    HR_MAX_SANA = 98.0          # No saturación total persistente

    for _ in range(max_intentos):
        # 1. Elegir parcela aleatoria
        parcela_id = np.random.choice(parcelas)
        df_parcela = datasets_clima[parcela_id]

        # 2. Filtrar fechas válidas como inicio (mes correcto)
        mask_mes = (df_parcela["Fecha"].dt.month >= mes_inicio_min) & \
                   (df_parcela["Fecha"].dt.month <= mes_inicio_max)

        # Además, asegurar que quedan suficientes registros después
        mask_espacio = df_parcela.index <= (len(df_parcela) - n_steps)
        
        indices_validos = df_parcela.index[mask_mes & mask_espacio]

        if len(indices_validos) == 0:
            continue

        # 3. Elegir fecha de inicio aleatoria
        idx_inicio = np.random.choice(indices_validos)
        bloque = df_parcela.iloc[idx_inicio: idx_inicio + n_steps].copy()

        # 4. Verificar integridad (sin NaN en variables críticas)
        cols_criticas = ["Temp_Amb_C", "Hum_Rel_Pct", "Lluvia_mm", "Viento_kmh", "ETo"]
        if bloque[cols_criticas].isnull().any().any():
            continue

        if len(bloque) < n_steps:
            continue

        # 5. VALIDACIÓN DE CONDICIONES INICIALES "SANAS"
        # Comprobamos que las primeras 24h no son un evento extremo
        primeras_horas = bloque.head(VENTANA_VALIDACION_H)
        
        t_media_inicio = primeras_horas["Temp_Amb_C"].mean()
        t_min_inicio = primeras_horas["Temp_Amb_C"].min()
        t_max_inicio = primeras_horas["Temp_Amb_C"].max()
        lluvia_acum_inicio = primeras_horas["Lluvia_mm"].sum()
        hr_media_inicio = primeras_horas["Hum_Rel_Pct"].mean()
        
        # ¿Temperatura fuera de rango razonable?
        if t_min_inicio < T_MIN_SANA or t_max_inicio > T_MAX_SANA:
            continue
        
        # ¿Lluvia acumulada excesiva? (ej: tormenta intensa al inicio)
        if lluvia_acum_inicio > LLUVIA_MAX_ACUM_24H:
            continue
        
        # ¿Humedad extrema persistente?
        if hr_media_inicio < HR_MIN_SANA or hr_media_inicio > HR_MAX_SANA:
            continue

        fecha_inicio = bloque["Fecha"].iloc[0]
        return bloque.reset_index(drop=True), parcela_id, fecha_inicio

    raise RuntimeError(
        f"No se encontró una ventana de clima válida para '{agente}' tras {max_intentos} intentos.\n"
        f"Verifica que los datos climáticos cubren los meses {mes_inicio_min}-{mes_inicio_max}."
    )


# ==============================================================================
# 3. FENOLOGÍA: Coeficiente de Cultivo (K_c) dinámico
# ==============================================================================
def calcular_curva_fenologia(n_steps, tipo_curva='LOGISTICA', agresividad=0.5):
    """
    Sustituye la curva fenológica por etapas clásica (FAO-56) por funciones 
    continuas (ej. Logística). Genera el coeficiente de cultivo dinámico (Kc) 
    para escalar la evapotranspiración sin introducir discontinuidades matemáticas.
    """
    x = np.linspace(0, 1, n_steps)
    if tipo_curva == 'LOGISTICA':
        # Simula el crecimiento rápido en primavera
        k = 10 + (agresividad * 5)
        # Centramos el crecimiento (0.5) a mitad de la temporada activa
        y = 1 / (1 + np.exp(-k * (x - 0.4))) 
    elif tipo_curva == 'EXPONENCIAL':
        y = x ** (3 + agresividad * 3)
    elif tipo_curva == 'ESCALON':
        pto = np.random.uniform(0.6, 0.85)
        y = 1 / (1 + np.exp(-50 * (x - pto)))
    else: 
        y = x
    # Normalización estricta 0-1
    return (y - y.min()) / (y.max() - y.min() + 1e-6)


# ==============================================================================
# 4. MODELO DINÁMICO DE SUELO
# ==============================================================================
def calcular_humedad_suelo_dinamico(lluvia_series, eto_series, kc_series, hum_inicial_pct):
    """
    Implementa un balance hídrico de capa única (modelo "cubo"). Calcula el 
    agotamiento de humedad diario acotado por Capacidad de Campo y Punto de 
    Marchitez (suelo franco-arcilloso). Incorpora cierre estomático dinámico 
    mediante un coeficiente de estrés (Ks) lineal cuando se supera el umbral 
    crítico de agua útil.
    """
    n_steps = len(lluvia_series)
    hum_suelo_pct = np.zeros(n_steps)

    # PARÁMETROS FÍSICOS
    PROFUNDIDAD_RAICES_MM = 800.0   
    CC_PCT = 35.0                   # Capacidad de Campo
    PMP_PCT = 18.0                  # Punto de Marchitez
    # Valores de Kc para la vid
    KC_MIN = 0.2                    # Invierno/Suelo desnudo
    KC_MAX = 0.8                    # Máximo desarrollo foliar
    
    # Conversión inicial
    agua_actual_mm = (hum_inicial_pct / 100.0) * PROFUNDIDAD_RAICES_MM
    CC_MM = (CC_PCT / 100.0) * PROFUNDIDAD_RAICES_MM
    PMP_MM = (PMP_PCT / 100.0) * PROFUNDIDAD_RAICES_MM
    
    for t in range(n_steps):
        # 1. ENTRADA
        if lluvia_series[t] > 0:
            agua_actual_mm += lluvia_series[t]
            
        # 2. SALIDA (EVAPOTRANSPIRACIÓN REAL)
        kc_actual = KC_MIN + kc_series[t] * (KC_MAX - KC_MIN)
        etc = eto_series[t] * kc_actual
        
        # Estrés hídrico: Si hay poca agua, la planta cierra estomas
        ks = 1.0
        umbral_estres = PMP_MM + 0.4 * (CC_MM - PMP_MM)
        
        if agua_actual_mm < umbral_estres:
            ks = (agua_actual_mm - PMP_MM) / (umbral_estres - PMP_MM)
            ks = max(0, ks)

        if agua_actual_mm > PMP_MM:
            agua_actual_mm -= (etc * ks)
        
        # 3. DRENAJE
        if agua_actual_mm > CC_MM:
            agua_actual_mm = CC_MM 
            
        hum_suelo_pct[t] = (agua_actual_mm / PROFUNDIDAD_RAICES_MM) * 100.0
        
    return hum_suelo_pct


# ==============================================================================
# 5. SIMULACIONES DE LAS ENFERMEDADES FÚNGICAS
# ==============================================================================

def calcular_horas_mojado(is_raining, hum_relativa):
    """
    Calcula la duración del mojado foliar (Leaf Wetness Duration - LWD).
    La hoja se considera mojada si llueve O si hay rocío (HR > 90%).
    Devuelve un array con las horas CONSECUTIVAS de mojado en cada paso t.
    """
    n_steps = len(is_raining)
    wetness_counter = np.zeros(n_steps)
    count = 0
    
    for t in range(n_steps):
        if is_raining[t] > 0 or hum_relativa[t] > 90.0:
            count += 1
        else:
            count = 0
        wetness_counter[t] = count
        
    return wetness_counter


# 5.1 MILDIU (Plasmopara viticola)
# Modelo: Regla de los 3-Dieces + Curva de Incubación de Goidanich
# ==============================================================================
def simular_dinamica_mildiu(temp_series, lluvia_series, fenologia_series, hum_relativa):
    """
    Simula el ciclo biológico del mildiu mediante el modelo empírico de los 3-Dieces
    y la ley de incubación de Goidanich. El modelo gestiona estados de incubación
    latente y dispara la esporulacion (progreso visible) bajo condiciones nocturnas 
    de alta higrometría (HR > 92%), simulando el estrés biótico progresivo.
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    incubando = False
    dias_incubacion_restantes = 0.0
    progreso_latente = 0.0
    
    PASOS_DIA = 24
    
    for t in range(PASOS_DIA, n_steps):
        lluvia_acum_24h = np.sum(lluvia_series[t-24:t])
        temp_media_24h = np.mean(temp_series[t-24:t])
        tamano_brote = fenologia_series[t]
        
        condicion_primaria = (temp_media_24h >= 10.0) and \
                             (tamano_brote >= 0.1) and \
                             (lluvia_acum_24h >= 10.0)
                             
        if condicion_primaria and not incubando and progreso_latente == 0:
            incubando = True
            temp_eff = max(6.1, temp_media_24h)
            dias_totales = 87.5 / (temp_eff - 6.0)
            dias_incubacion_restantes = dias_totales
        
        if incubando:
            factor_temp = 1.0 
            if temp_series[t] > 20: factor_temp = 1.2
            
            dias_incubacion_restantes -= (1/24.0) * factor_temp
            
            if dias_incubacion_restantes <= 0:
                incubando = False
                progreso_latente = 0.1
        
        es_noche = (t % 24) < 6 or (t % 24) > 20
        if progreso_latente > 0 and hum_relativa[t] > 92.0 and es_noche:
            progreso_latente += 0.05
            
        if progreso_latente > 1.0: progreso_latente = 1.0
        infeccion[t] = progreso_latente

    return infeccion


# 5.2 BOTRYTIS (Botrytis cinerea)
# Modelo: Ecuación Logit de Broome (1995)
# ==============================================================================
def simular_dinamica_botrytis(temp_series, lluvia_series, hum_relativa, dano_previo=False):
    """
    Simula la infección por Botrytis cinerea basándose en el modelo Logit de Broome (1995).
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    horas_mojado = calcular_horas_mojado(lluvia_series, hum_relativa)
    acumulado_danio = 0.0
    
    for t in range(n_steps):
        W = horas_mojado[t]
        T = temp_series[t]
        
        if W > 4:
            logit = -2.6478 - (0.3749 * W) + (0.0616 * T * W) - (0.0015 * W * (T**2))
            prob_infeccion = 1 / (1 + np.exp(-logit))
            
            factor_mec = 2.0 if dano_previo else 1.0
            
            if prob_infeccion > 0.4:
                tasa = 0.04 * prob_infeccion * factor_mec
                acumulado_danio += tasa
        
        infeccion[t] = min(acumulado_danio, 1.0)
        
    return infeccion


# 5.3 BLACK ROT (Guignardia bidwellii)
# Modelo: Polinomio de Spotts (Horas mínimas vs Temperatura)
# ==============================================================================
def simular_dinamica_black_rot(temp_series, lluvia_series, hum_relativa):
    """
    Simula la infección por Black Rot implementando el modelo de Spotts (1977).
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    horas_mojado = calcular_horas_mojado(lluvia_series, hum_relativa)
    acumulado_danio = 0.0
    
    for t in range(n_steps):
        T = temp_series[t]
        W_real = horas_mojado[t]
        
        if 7 < T < 32:
            W_min = (0.03 * T**2) - (1.45 * T) + 22.5
            W_min = max(6.0, W_min)
            
            if W_real > W_min:
                exceso = W_real - W_min
                tasa = 0.01 + (exceso * 0.005)
                acumulado_danio += tasa
                
        infeccion[t] = min(acumulado_danio, 1.0)
        
    return infeccion


# 5.4 OIDIO (Erysiphe necator)
# Modelo: Índice de Riesgo Gubler-Thomas (GTI)
# ==============================================================================
def simular_dinamica_oidio(temp_series, lluvia_series):
    """
    Simula la dinámica epidémica del Oídio basándose en una adaptación del 
    Índice de Riesgo de Gubler-Thomas (GTI).
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    gti_score = 0.0
    pasos_dia = 24
    dias_totales = n_steps // pasos_dia
    
    acumulado_danio = 0.0
    
    for d in range(dias_totales):
        idx_start = d * pasos_dia
        idx_end = idx_start + pasos_dia
        
        temps_dia = temp_series[idx_start:idx_end]
        lluvia_dia = np.sum(lluvia_series[idx_start:idx_end])
        t_max = np.max(temps_dia)
        
        horas_optimas = np.sum((temps_dia >= 21) & (temps_dia <= 30))
        
        score_change = 0
        if horas_optimas >= 6:
            score_change += 20
        else:
            score_change -= 5
            
        if t_max > 35.0 or lluvia_dia > 2.0:
            score_change -= 10
            
        gti_score = np.clip(gti_score + score_change, 0, 100)
        
        if gti_score > 60:
            tasa_diaria = 0.05
        elif gti_score > 30:
            tasa_diaria = 0.012
        else:
            tasa_diaria = 0.0
            
        acumulado_danio += tasa_diaria
        
        for h in range(pasos_dia):
            progreso_hora = acumulado_danio - tasa_diaria + (tasa_diaria * (h + 1) / pasos_dia)
            if idx_start + h < n_steps:
                infeccion[idx_start + h] = progreso_hora
                
    return np.clip(infeccion, 0, 1.0)


# ==============================================================================
# 6. SIMULACIÓN DE LA ESCA
# ==============================================================================
def simular_dinamica_esca(n_steps, eto_series, hum_suelo_series, temp_series, riesgo_inicial):
    """
    Simula el complejo de la Yesca como un proceso de difusión con saltos (Jump-Diffusion).
    Mecanismo: Fallo Hidráulico desencadenado por alta demanda evaporativa.
    """
    tasa_deterioro_diaria = 0.0005
    carga_fungica = np.linspace(riesgo_inicial, riesgo_inicial + (n_steps * tasa_deterioro_diaria), n_steps)
    
    conductividad_tronco = 1.0 - (carga_fungica ** 2) 
    conductividad_tronco = np.maximum(0.01, conductividad_tronco)
    
    grado_infeccion = np.copy(carga_fungica) * 0.3
    colapso_ocurrido = False
    
    for t in range(n_steps):
        if colapso_ocurrido:
            grado_infeccion[t] = 1.0
            continue
            
        demanda = eto_series[t]
        oferta = conductividad_tronco[t] * max(0.1, hum_suelo_series[t] / 30.0)
        
        hsi = demanda / (oferta + 1e-6)
        
        if temp_series[t] > 25.0 and hsi > 5.0:
            lambda_jump = 0.01 * np.exp(0.5 * (hsi - 5.0)) 
            lambda_jump = min(0.8, lambda_jump)
            
            if np.random.rand() < lambda_jump:
                colapso_ocurrido = True
                grado_infeccion[t] = 1.0
        
        else:
            grado_infeccion[t] = carga_fungica[t] * 0.2 + (hsi * 0.01)

    return np.clip(grado_infeccion, 0, 1.0)


# ==============================================================================
# 7. SIMULACIÓN DE PLAGAS
# ==============================================================================
def calcular_gdd_acumulado(temp_series, t_base=10.0):
    """
    Calcula la integral térmica (Grados-Día de Crecimiento).
    """
    n_steps = len(temp_series)
    gdd_series = np.zeros(n_steps)
    acumulado = 0.0
    
    pasos_dia = 24
    dias = n_steps // pasos_dia
    
    for d in range(dias):
        idx_start = d * pasos_dia
        idx_end = idx_start + pasos_dia
        t_media_dia = np.mean(temp_series[idx_start:idx_end])
        
        grado_dia = max(0, t_media_dia - t_base)
        acumulado += grado_dia
        
        gdd_series[idx_start:idx_end] = acumulado
        
    return gdd_series


# 7.1 POLILLA DEL RACIMO (LOBESIA BOTRANA)
# Modelo: Picos generacionales basados en GDD (Gaussianas)
# ==============================================================================
def simular_dinamica_lobesia(temp_series, lluvia_series):
    """
    Simula la dinámica poblacional mediante un modelo fenológico de Grados-Día (GDD) 
    con tres picos generacionales superpuestos.
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    gdd_acum = calcular_gdd_acumulado(temp_series)
    
    picos_gdd = [150, 450, 850]
    anchuras_gdd = [50, 80, 100]
    severidades = [0.2, 0.8, 1.5]
    
    for t in range(n_steps):
        gdd_actual = gdd_acum[t]
        
        presion_total = 0.0
        for mu, sigma, sev in zip(picos_gdd, anchuras_gdd, severidades):
            forma = np.exp(-0.5 * ((gdd_actual - mu) / sigma) ** 2)
            presion_total += forma * sev
            
        if lluvia_series[t] > 2.0:
            presion_total *= 0.2
            
        infeccion[t] = presion_total
        
    infeccion_acumulada = np.maximum.accumulate(infeccion)
    
    return np.clip(infeccion_acumulada, 0, 1.0)


# 7.2 MOSQUITO VERDE (Empoasca vitis)
# Modelo: Densidad-Dependiente del Vigor (Fenología) + Mortalidad por Sequía
# ==============================================================================
def simular_dinamica_empoasca(temp_series, hum_relativa, kc_series):
    """
    Simula un crecimiento logístico dependiente de la temperatura.
    """
    n_steps = len(temp_series)
    poblacion = np.zeros(n_steps)
    
    nivel_actual = 0.05 
    
    for t in range(1, n_steps):
        temp = temp_series[t]
        tasa_crecimiento = 0.0
        
        if 20 < temp < 32:
            tasa_crecimiento = 0.15
        elif temp >= 32:
            tasa_crecimiento = 0.005 
        
        capacidad_carga = kc_series[t] + 0.5 
        
        factor_mortalidad = 0.0
        if hum_relativa[t] < 20.0:
            factor_mortalidad = 0.1 
        
        delta = tasa_crecimiento * nivel_actual * (1 - nivel_actual / (capacidad_carga + 0.01))
        
        nivel_actual += delta - (nivel_actual * factor_mortalidad)
        nivel_actual += np.random.normal(0, 0.002)
        
        poblacion[t] = np.clip(nivel_actual, 0, 1.0)
        nivel_actual = poblacion[t]

    return poblacion


# 7.3 ALTICA (Altica ampelophaga)
# Modelo: Emergencia por Umbral GDD (Función Escalón)
# ==============================================================================
def simular_dinamica_altica(temp_series):
    """
    Simula la emergencia de adultos invernantes mediante una función escalón al 
    superar el umbral crítico de 280 GDD.
    """
    n_steps = len(temp_series)
    dano = np.zeros(n_steps)
    
    gdd_acum = calcular_gdd_acumulado(temp_series, t_base=10.0)
    UMBRAL_EMERGENCIA = 280.0
    
    adultos_emergidos = False
    nivel_dano = 0.0
    
    for t in range(n_steps):
        if not adultos_emergidos and gdd_acum[t] > UMBRAL_EMERGENCIA:
            adultos_emergidos = True
            
        if adultos_emergidos:
            tasa_ingesta = 0.0
            if temp_series[t] > 15:
                tasa_ingesta = 0.005
            
            nivel_dano += tasa_ingesta
            
        dano[t] = nivel_dano
        
    return np.clip(dano, 0, 1.0)


# 7.4 ARAÑA ROJA (RED MITE)
# Modelo: Crecimiento Exponencial inverso a la Humedad
# ==============================================================================
def simular_dinamica_arana_roja(temp_series, hum_relativa, lluvia_series):
    """
    Simula el crecimiento exponencial discreto de la Araña Roja.
    """
    n_steps = len(temp_series)
    poblacion = np.zeros(n_steps)
    
    N = 0.01
    
    for t in range(n_steps):
        T = temp_series[t]
        HR = hum_relativa[t]
        
        potencial_termico = max(0, T - 12) * 0.002
        factor_sequedad = (1 - (HR / 100.0)) ** 2 
        
        r_m = potencial_termico * factor_sequedad
        N = N * (1 + r_m)
        
        if lluvia_series[t] > 1.0:
            N = N * 0.6
            
        N = min(N, 1.0)
        poblacion[t] = N
        
    return poblacion


# 7.5 ERINOSIS (Colomerus vitis)
# Modelo: Ventana Fenológica (Solo infecta al brotar)
# ==============================================================================
def simular_dinamica_erinosis(temp_series, fenologia_series, hum_relativa):
    """
    Simula el establecimiento de agallas restringido a la ventana de brotación.
    """
    n_steps = len(temp_series)
    infeccion = np.zeros(n_steps)
    
    estado_establecido = 0.0
    
    for t in range(n_steps):
        kc = fenologia_series[t]
        T = temp_series[t]
        HR = hum_relativa[t]
        
        en_ventana = (kc >= 0.05) and (kc <= 0.40)
        
        if en_ventana:
            if 18.0 <= T <= 27.0 and HR > 65.0:
                estado_establecido += 0.003
            elif T > 15.0: 
                estado_establecido += 0.001
        
        infeccion[t] = estado_establecido
        
    return np.clip(infeccion, 0, 0.7)


# ==============================================================================
# 7.6 SIMULACIÓN DE GASES (CO2 y VOC)
# ==============================================================================
def simular_gases_planta(temp_series, viento_series, lluvia_series, grado_inf, agente, horas_dia):
    """
    Simula emisiones de CO2 y VOC evitando 'data leakage'.
    Los gases dependen fuertemente del ciclo diario, la temperatura y el viento.
    La enfermedad altera las señales, pero no es una relación lineal 1-a-1.
    """
    n_steps = len(temp_series)
    co2 = np.zeros(n_steps)
    voc = np.zeros(n_steps)
    
    # 1. Daño activo (derivada discreta)
    tasa_dano = np.zeros(n_steps)
    tasa_dano[1:] = np.maximum(0, np.diff(grado_inf))
    
    # 2. Inercia biológica del estrés (Media Móvil Exponencial)
    # Simula cómo la planta sigue emitiendo defensas químicas tras el ataque inicial
    estres_inercial = np.zeros(n_steps)
    acumulador = 0.0
    for t in range(n_steps):
        acumulador = (acumulador * 0.90) + tasa_dano[t]  # Factor de decaimiento (retardo)
        estres_inercial[t] = acumulador
    
    for t in range(n_steps):
        T = temp_series[t]
        viento = viento_series[t]
        hora = horas_dia[t]
        lluvia = lluvia_series[t]
        inf_actual = grado_inf[t]
        dano_activo = tasa_dano[t]
        
        # --- CO2 (ppm) ---
        es_dia = 7 <= hora <= 19
        co2_base = 410.0  
        
        if es_dia:
            # Fotosíntesis activa: consume CO2
            eficiencia = max(0.1, 1.0 - (inf_actual * 0.8))
            co2_local = co2_base - (30.0 * eficiencia * max(0, T - 10)/15.0)
        else:
            # Respiración nocturna: emite CO2
            estres = 1.0 + (inf_actual * 0.5)
            co2_local = co2_base + (40.0 * estres * max(0, T - 5)/15.0)
            
        # Viento dispersa y acerca al background
        factor_mezcla = min(1.0, viento / 15.0)
        co2[t] = co2_local * (1 - factor_mezcla) + co2_base * factor_mezcla
        co2[t] += np.random.normal(0, 2.0)
        
        # --- VOC (ppb) ---
        # Background natural basado en temperatura (estrés abiótico confuso para el modelo)
        voc_base = 150.0 + (max(0, T - 20) * 4.0) 
        if T > 35.0:  # Estrés térmico extremo dispara VOC natural
            voc_base += (T - 35.0) * 15.0 

        # Emisión por estrés biótico (SIN USAR LA VARIABLE 'agente')
        # Utilizamos el daño acumulado (inf_actual) y el estrés inercial reciente
        emision_biotica = (estres_inercial[t] * 2500.0) + (inf_actual * 40.0)
        
        voc_local = voc_base + emision_biotica
        
        # Dispersión atmosférica
        factor_viento = np.exp(-viento / 8.0)
        factor_lluvia = 0.4 if lluvia > 0.5 else 1.0
        
        voc[t] = voc_local * factor_viento * factor_lluvia
        
        # Ruido de lectura del sensor
        voc[t] += np.random.normal(0, 6.0)
        
    return co2, voc



# ==============================================================================
# 8. ORQUESTACIÓN Y GENERACIÓN DEL DATASET
# ==============================================================================

def generar_configuracion_serie(n_muestras):
    """
    Genera la configuración inicial para cada serie, equilibrando las clases.
    Ya no necesita parámetros climáticos (temp_target, lluvia_mult) porque
    el clima viene de datos reales.
    """
    print(f"⚙️ Generando configuración para {n_muestras} series...")
    data = []
    
    for _ in range(n_muestras):
        agente = np.random.choice(CLASES_SALIDA)
        
        data.append({
            'Etiqueta_Clase': agente,
            'Riesgo_Base': np.random.uniform(0.1, 0.5),
            'pH_Base': 6.8 + np.random.normal(0, 0.2),
            'Hum_Suelo_Ini': np.random.uniform(20.0, 35.0),
        })

    return pd.DataFrame(data)


def simular_serie(row, datasets_clima):
    """
    Ejecuta todos los motores para una sola serie temporal, usando clima REAL.
    """
    agente = row['Etiqueta_Clase']
    clase = row['Etiqueta_Clase']
    
    # 1. DURACIÓN
    if agente == "HEALTHY":
        dias = np.random.randint(30, 60)
    elif agente == "ERINOSIS":
        dias = 120
    else:
        dias = 200
        
    n_steps = dias * PASOS_POR_DIA
    
    # 2. MUESTREO DE CLIMA REAL (en lugar de generar_clima_integrado)
    bloque_clima, parcela_id, fecha_inicio = muestrear_clima_real(
        datasets_clima, dias, agente
    )
    
    # Extraer arrays numpy del bloque real
    temp = bloque_clima["Temp_Amb_C"].values.astype(np.float64)
    lluvia = bloque_clima["Lluvia_mm"].values.astype(np.float64)
    viento = bloque_clima["Viento_kmh"].values.astype(np.float64)
    hr = bloque_clima["Hum_Rel_Pct"].values.astype(np.float64)
    eto = bloque_clima["ETo"].values.astype(np.float64)
    fechas_serie = bloque_clima["Fecha"].values
    
    # 3. Variables derivadas del clima real
    kc = calcular_curva_fenologia(n_steps)
    h_suelo = calcular_humedad_suelo_dinamico(lluvia, eto, kc, row['Hum_Suelo_Ini'])
    gdd = calcular_gdd_acumulado(temp)
    mojado = calcular_horas_mojado((lluvia > 0.1).astype(int), hr)
    
    # 4. Simulación del Agente (MOTORES BIOLÓGICOS INTACTOS)
    if agente == "HEALTHY":     grado_inf = np.random.uniform(0, 0.02, n_steps)
    elif agente == "MILDIU":    grado_inf = simular_dinamica_mildiu(temp, lluvia, kc, hr)
    elif agente == "BOTRYTIS":  grado_inf = simular_dinamica_botrytis(temp, lluvia, hr)
    elif agente == "BLACK_ROT": grado_inf = simular_dinamica_black_rot(temp, lluvia, hr)
    elif agente == "OIDIO":     grado_inf = simular_dinamica_oidio(temp, lluvia)
    elif agente == "ESCA":      grado_inf = simular_dinamica_esca(n_steps, eto, h_suelo, temp, row['Riesgo_Base'])
    elif agente == "LOBESIA":   grado_inf = simular_dinamica_lobesia(temp, lluvia)
    elif agente == "EMPOASCA":  grado_inf = simular_dinamica_empoasca(temp, hr, kc)
    elif agente == "ALTICA":    grado_inf = simular_dinamica_altica(temp)
    elif agente == "RED_MITE":  grado_inf = simular_dinamica_arana_roja(temp, hr, lluvia)
    elif agente == "ERINOSIS":  grado_inf = simular_dinamica_erinosis(temp, kc, hr)
    else:                       grado_inf = np.zeros(n_steps)

    # 5. Variables complementarias (SIN data leakage)
    ph = row['pH_Base'] - (np.cumsum(lluvia) * 0.001) + np.random.normal(0, 0.02, n_steps)
    
    # 5.1 Simulamos CO2 y VOC (Modelo Guenther + estrés biótico universal)
    horas_dia = pd.Series(fechas_serie).dt.hour.values
    co2, voc = simular_gases_planta(temp, viento, lluvia, grado_inf, agente, horas_dia)
    
    # 6. GENERACIÓN DEL DATAFRAME
    df_serie = pd.DataFrame({
        'Fecha': fechas_serie,
        'Etiqueta_Clase': clase,
        'Grado_Infeccion': grado_inf.astype('float32'),
        'Temp_Amb_C': temp.astype('float32'),
        'Hum_Rel_Pct': hr.astype('float32'),
        'Lluvia_mm': lluvia.astype('float32'),
        'Viento_kmh': viento.astype('float32'),
        'Hum_Suelo_Pct': h_suelo.astype('float32'),
        'pH_Suelo': ph.astype('float32'),
        'CO2_ppm': co2.astype('float32'),
        'VOC_ppb': voc.astype('float32'),
        'Parcela_ID': parcela_id,
        # NOTA: GDD_Acumulado y Horas_Humedad_Foliar se calculan en el
        # preprocesamiento (src/data_processing/preprocess.py) a partir de
        # Temp_Amb_C, Hum_Rel_Pct y Lluvia_mm, ya que son variables derivadas
        # que no provienen directamente de sensores.
    })

    # 7. Clase de entrenamiento (misma lógica que v1)
    df_serie['Clase_Entrenamiento'] = clase
    
    UMBRAL_DETECCION = 0.05
    mask_sano = df_serie['Grado_Infeccion'] < UMBRAL_DETECCION
    
    if agente != "HEALTHY":
        df_serie.loc[mask_sano, 'Clase_Entrenamiento'] = 'HEALTHY'
    
    # Castings
    df_serie['Etiqueta_Clase'] = df_serie['Etiqueta_Clase'].astype(str)
    df_serie['Clase_Entrenamiento'] = df_serie['Clase_Entrenamiento'].astype(str)
    df_serie['Parcela_ID'] = df_serie['Parcela_ID'].astype(str)

    # =========================================================================
    # POST-PROCESADO: RECORTE INTELIGENTE DE SERIES (CONSERVADO DE v1)
    # =========================================================================
    if agente != "HEALTHY":
        # --- A) RECORTE INFERIOR ---
        UMBRAL_INICIO = 0.01
        BUFFER_INICIO_HORAS = 96

        mask_infectado = df_serie['Grado_Infeccion'].values >= UMBRAL_INICIO

        if mask_infectado.any():
            primera_pos = int(mask_infectado.argmax())
            pos_inicio = max(0, primera_pos - BUFFER_INICIO_HORAS)
            df_serie = df_serie.iloc[pos_inicio:].copy()
        else:
            return None

        # --- B) RECORTE SUPERIOR ---
        if agente != "ERINOSIS":
            UMBRAL_FIN = 0.99
            BUFFER_FIN_HORAS = 24

            mask_total = df_serie['Grado_Infeccion'].values >= UMBRAL_FIN

            if mask_total.any():
                primera_pos_fin = int(mask_total.argmax())
                pos_fin = min(len(df_serie), primera_pos_fin + BUFFER_FIN_HORAS)
                df_serie = df_serie.iloc[:pos_fin].copy()
        else:
            BUFFER_ESTABILIZACION_HORAS = 20 * 24
            valores_inf = df_serie['Grado_Infeccion'].values
            diff = np.diff(valores_inf)
            indices_creciendo = np.where(diff > 1e-6)[0]

            if len(indices_creciendo) > 0:
                ultimo_crecimiento = indices_creciendo[-1]
                pos_corte = min(len(df_serie), ultimo_crecimiento + BUFFER_ESTABILIZACION_HORAS)
                df_serie = df_serie.iloc[:pos_corte].copy()

    return df_serie


# ==============================================================================
# 9. FUNCIÓN PRINCIPAL
# ==============================================================================
def main():
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    
    # 1. CARGAR DATOS CLIMÁTICOS REALES
    print("=" * 70)
    print("  SIMULADOR DE ENFERMEDADES Y PLAGAS EN LA VID (v2)")
    print("  Motor: Clima REAL + Etiquetador Biológico")
    print("=" * 70)
    print(f"\n📂 Cargando datos climáticos reales desde: {CLIMA_DIR}")
    
    datasets_clima = cargar_datos_clima_real(CLIMA_DIR)
    print(f"\n✅ {len(datasets_clima)} parcelas cargadas en memoria.")
    
    # 2. CONFIGURAR SERIES
    df_config = generar_configuracion_serie(N_SERIES_A_GENERAR)
    
    print(f"\n🎬 Iniciando simulación de {N_SERIES_A_GENERAR} series...")
    batch_list = []
    
    parquet_writer = None
    schema = None
    
    id_serie = 0
    series_descartadas = 0
    
    for i, (idx, row) in enumerate(df_config.iterrows()):
        try:
            df_serie = simular_serie(row, datasets_clima)
        except RuntimeError as e:
            print(f"  ⚠️ Error en serie {i}: {e}")
            series_descartadas += 1
            continue
        
        if df_serie is None or df_serie.empty:
            series_descartadas += 1
            continue
        
        df_serie['ID_Serie'] = id_serie
        id_serie += 1
        batch_list.append(df_serie)
        
        # Guardado por lotes (cada 200 series)
        if len(batch_list) >= 200:
            df_batch = pd.concat(batch_list, ignore_index=True)
            df_batch = df_batch.reindex(sorted(df_batch.columns), axis=1)
            table = pa.Table.from_pandas(df_batch)
            
            if parquet_writer is None:
                schema = table.schema
                parquet_writer = pq.ParquetWriter(OUTPUT_FILE, schema, compression='snappy')
            
            parquet_writer.write_table(table)
            
            batch_list = []
            del df_batch
            del table
            gc.collect()
            print(f"  📦 Lote completado: {i+1}/{N_SERIES_A_GENERAR} (descartadas: {series_descartadas})")
            
    # Lote final
    if batch_list:
        df_batch = pd.concat(batch_list, ignore_index=True)
        df_batch = df_batch.reindex(sorted(df_batch.columns), axis=1)
        table = pa.Table.from_pandas(df_batch)
        
        if parquet_writer is None:
            schema = table.schema
            parquet_writer = pq.ParquetWriter(OUTPUT_FILE, schema, compression='snappy')
            
        parquet_writer.write_table(table)
        print(f"  📦 Lote final completado.")

    if parquet_writer is not None:
        parquet_writer.close()

    # RESUMEN FINAL
    print(f"\n{'=' * 70}")
    print(f"  ✅ Dataset v2 generado con éxito")
    print(f"     Archivo: {OUTPUT_FILE}")
    print(f"     Series válidas: {id_serie}")
    print(f"     Series descartadas: {series_descartadas}")
    print(f"{'=' * 70}")

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    main()
