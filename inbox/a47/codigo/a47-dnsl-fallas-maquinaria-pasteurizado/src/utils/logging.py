import logging
import sys
import os

def get_logger(name: str) -> logging.Logger:
    """
    Configura y devuelve un logger con salida a consola.
    """
    logger = logging.getLogger(name)
    
    # Evitar múltiples handlers si se llama varias veces
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
