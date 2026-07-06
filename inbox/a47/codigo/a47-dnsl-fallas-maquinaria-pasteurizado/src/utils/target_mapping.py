"""
Funciones centralizadas de mapeo de targets.

Convierte los valores originales del dataset UCI (Cooler_Condition, Valve_Condition,
Pump_Leakage, Hydraulic_Accumulator) a las clases internas del modelo (0=Sano, 1=Warning, 2=Crítico).

IMPORTANTE: Estas funciones son la ÚNICA fuente de verdad para el mapeo de targets.
No deben duplicarse en otros módulos.
"""

def map_cooler(val):
    """Mapea Cooler_Condition: 100→Sano(0), 20→Warning(1), resto→Crítico(2)."""
    return 0 if val == 100 else (1 if val == 20 else 2)

def map_valve(val):
    """Mapea Valve_Condition: 100→Sano(0), >=80→Warning(1), resto→Crítico(2)."""
    return 0 if val == 100 else (1 if val >= 80 else 2)

def map_pump(val):
    """Mapea Pump_Leakage: 0→Sano(0), 1→Warning(1), resto→Crítico(2)."""
    return 0 if val == 0 else (1 if val == 1 else 2)

def map_acc(val):
    """Mapea Hydraulic_Accumulator: 130→Sano(0), >=100→Warning(1), resto→Crítico(2)."""
    return 0 if val == 130 else (1 if val >= 100 else 2)
