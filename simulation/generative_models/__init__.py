"""
Generative models for simulation-based inference.

Each model module implements a `generate` function that simulates agent behavior.
All models follow the same interface:

    generate(sim_params: np.ndarray, model_spec: ParamConfig, seed: int) -> Dict[str, Any]
    
    Returns a dictionary with:
        - trial_data: pd.DataFrame with all trial-level data (observed + latent variables)
        - sim_params: np.ndarray of free parameters (raw input)
        - params_dict: dict with all parameters (filled with defaults)
        - model_id: str model identifier
        - model_spec: ParamConfig used for simulation
        - seed: int random seed
"""
from .ruleQhyp import generate as generate_ruleQhyp
from .Qlearn import generate as generate_Qlearn

__all__ = ['generate_ruleQhyp', 'generate_Qlearn']
