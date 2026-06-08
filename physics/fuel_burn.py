# D:\Chatbot\physics\fuel_burn.py

def calculate_vehicle_mass(initial_mass_kg: float, lap: int, fuel_burn_per_lap_kg: float) -> float:
    """
    Calculates the instantaneous vehicle mass at the start of a given lap.
    
    Formula: M_vehicle(t) = M_initial - (t * m_dot_fuel)
    """
    return max(0.0, initial_mass_kg - (lap * fuel_burn_per_lap_kg))

def calculate_fuel_corrected_time(
    actual_time_s: float, 
    initial_mass_kg: float, 
    current_lap: int, 
    fuel_burn_per_lap_kg: float, 
    alpha_fuel: float = 0.03
) -> float:
    """
    Corrects the actual lap time to isolate mechanical and tire grip.
    Normalizes the time to show what the pace would be at zero-fuel load.
    
    Formula: T_corrected = T_actual - alpha_fuel * [M_initial - M_vehicle(t)]
    """
    current_mass = calculate_vehicle_mass(initial_mass_kg, current_lap, fuel_burn_per_lap_kg)
    mass_delta = initial_mass_kg - current_mass
    time_correction = alpha_fuel * mass_delta
    
    # We subtract the time correction since a lighter car is naturally faster
    corrected_time = actual_time_s - time_correction
    return round(corrected_time, 3)