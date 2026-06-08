# physics/tire_deg.py
import math

def predict_tire_degradation_penalty(
    lap: int, 
    compound: str, 
    track_temp_f: float, 
    threshold_laps: int = 12
) -> float:
    """
    Predicts the performance drop-off (in seconds) due to tire wear.
    
    Formula: delta_thermal(t) = c * e^(k * (t - t_threshold))
    """
    comp = compound.lower()
    
    # Configure compound-specific coefficients
    if comp == "soft":
        linear_rate = 0.08  # Softs wear faster
        k = 0.65            # Aggressive exponential cliff curve
        c = 0.15            # Cliff base penalty scaling
    elif comp == "medium":
        linear_rate = 0.05
        k = 0.45
        c = 0.10
    else:  # "hard"
        linear_rate = 0.02
        k = 0.30
        c = 0.05
        threshold_laps += 5  # Hards last longer before hitting the cliff
    
    # Track temperature penalty scaling (optimal is 95°F)
    temp_factor = 1.0 + max(0.0, (track_temp_f - 95.0) / 30.0)
    effective_linear_rate = linear_rate * temp_factor
    
    # Phase 1: Warmup (First 2 laps show improvement, negative penalty)
    if lap <= 2:
        return -0.25 * (3 - lap)
    
    # Phase 2: Steady-State Linear Wear
    base_wear_penalty = (lap - 2) * effective_linear_rate
    
    # Phase 3: Non-Linear Thermal Cliff
    cliff_penalty = 0.0
    if lap > threshold_laps:
        laps_past_threshold = lap - threshold_laps
        # Exponential curve: c * e^(k * t_diff)
        cliff_penalty = c * math.exp(k * laps_past_threshold) * temp_factor
        
    return round(base_wear_penalty + cliff_penalty, 3)