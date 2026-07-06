import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)

def format_output(resultados: dict, cycle_id=None, real_targets=None, verbose: bool = True) -> pd.DataFrame:
    """
    Convierte el resultado de inferencia en un DataFrame y formatea a Strings.
    Si se proporciona real_targets, añade las columnas correspondientes.
    """
    if resultados is None:
        logger.error("No hay resultados para formatear.")
        return pd.DataFrame()

    components = ['Enfriador_Fouling', 'Válvula_Switch', 'Bomba_Leakage', 'Acumulador_Gas']
    states = ['SANO', 'WARNING', 'CRÍTICO']
    
    preds_int = resultados['predicciones']
    confs = resultados['confianzas']
    
    data = {"Componente": components}
    
    data["Prediccion_Numerica"] = preds_int
    data["Prediccion_Texto"] = [states[p] for p in preds_int]
    data["Confianza"] = confs
    
    if real_targets is not None:
        data["Realidad_Numerica"] = real_targets
        data["Realidad_Texto"] = [states[r] for r in real_targets]
        data["Acierto"] = [p == r for p, r in zip(preds_int, real_targets)]
        
    if cycle_id is not None:
        data["Cycle_ID"] = cycle_id

    df_out = pd.DataFrame(data)
    
    # Dashboard log
    if verbose:
        logger.info("Resultados listos:")
        if real_targets is not None:
            logger.info(f"{'COMPONENTE':<22} | {'PREDICCIÓN':<10} | {'REALIDAD':<10} | {'ACIERTO'} | {'CONF'}")
            for i, row in df_out.iterrows():
                marca = "OK " if row['Acierto'] else "ERR"
                logger.info(f"{row['Componente']:<22} | {row['Prediccion_Texto']:<10} | {row['Realidad_Texto']:<10} | {marca:<7} | {row['Confianza']:.2%}")
        else:
            for i, row in df_out.iterrows():
                logger.info(f"{row['Componente']:<22} | {row['Prediccion_Texto']:<10} | Conf: {row['Confianza']:.2%}")
        
    return df_out

def save_predictions(df_pred: pd.DataFrame, out_path: str) -> None:
    """
    Guarda las predicciones en CSV.
    """
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_pred.to_csv(out_path, index=False)
    logger.info(f"Predicciones guardadas en {out_path}.")
