import math

def perplexity(mean_loss: float) -> float:
    if mean_loss < 0: raise ValueError("Loss cannot be negative.")
    return math.exp(mean_loss)
