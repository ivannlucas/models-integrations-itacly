def apply_threshold(proba: float, threshold: float = 0.5) -> int:
    """Convert probability to binary label using a fixed threshold."""
    return int(proba >= threshold)
