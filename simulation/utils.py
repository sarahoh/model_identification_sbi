import numpy as np
from typing import Set, Optional, Dict, Any, List, Tuple
from itertools import product
import random
from .constants import TASK_RULES, N_RULES_PER_AGENT, PROB_SILENT_4TH_RULE, N_ACTIONS


def renorm(x: np.ndarray) -> np.ndarray:
    """Renormalize array to sum to 1."""
    s = np.sum(x)
    if s == 0:
        return x
    return x / s


def random_stimulus_generator(seed, L=2):
    """
    Returns a generator that yields tuples of length L.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    L : int, optional
        Length of tuple to yield (default: 2, for visual and auditory cues)
        
    Yields:
    -------
    tuple
        Tuple of length L with values 0 or 1
    """
    rng = np.random.default_rng(seed)
    while True:
        yield tuple(rng.integers(0, 2, size=L))



def generate_adaptive_rule_sequence_from_dark_behavior(
    dark_actions: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate rule sequence adaptively based on DARK session behavior.
    
    Follows experimental protocol:
    1. DARK (100% - all agents start here, already completed)
    2. LEFT or RIGHT (opposite of preferred side during DARK)
    3. AL (100%)
    4. SILENT (75%) or stop at 3 rules (25%)
    
    Parameters:
    -----------
    dark_actions : np.ndarray
        Actions taken during DARK sessions (0 or 1)
    rng : np.random.Generator
        Random number generator
        
    Returns:
    --------
    np.ndarray
        Rule sequence starting with DARK: ['DARK', 'LEFT'/'RIGHT', 'AL', 'SILENT'?]
    """
    # Count which side was used more during DARK
    n_left = np.sum(dark_actions == 0)
    n_right = np.sum(dark_actions == 1)
    
    # Select opposite of preferred side as 2nd rule
    if n_left < n_right:
        # Used LEFT less → train LEFT
        second_rule = 'LEFT'
    elif n_right < n_left:
        # Used RIGHT less → train RIGHT
        second_rule = 'RIGHT'
    else:
        # Equal usage → random choice
        second_rule = rng.choice(['LEFT', 'RIGHT'])
    
    # Build sequence: DARK → opposite side → AL → SILENT (maybe)
    rule_sequence = ['DARK', second_rule, 'AL']
    
    # 75% chance of adding SILENT as 4th rule
    if rng.random() < PROB_SILENT_4TH_RULE:
        rule_sequence.append('SILENT')
    
    return np.array(rule_sequence)


def get_rule_policy(rule: str, visual: int, auditory: int, prev_action: Optional[float]) -> np.ndarray:
    """
    Get deterministic policy for a rule. Returns probability distribution over actions.
    
    Used by both ruleQhyp and Qlearn models to determine the correct action
    based on the current rule and stimulus configuration.
    
    Parameters:
    -----------
    rule : str
        Rule name (LEFT, RIGHT, LIGHT, DARK, CLICK, SILENT, AL)
    visual : int
        Visual cue (0 or 1)
    auditory : int
        Auditory cue (0 or 1)
    prev_action : Optional[float]
        Previous action (0 or 1), required for AL rule
        
    Returns:
    --------
    np.ndarray
        Probability distribution over actions (length N_ACTIONS)
        
    Raises:
    -------
    ValueError
        If rule is invalid or if AL rule is used with NaN prev_action
    """
    action_probs = np.zeros(N_ACTIONS)
    if rule == 'LEFT':
        action_probs[0] = 1.0
    elif rule == 'RIGHT':
        action_probs[1] = 1.0
    elif rule == 'LIGHT':
        action_probs[visual] = 1.0
    elif rule == 'DARK':
        action_probs[1 - visual] = 1.0
    elif rule == 'CLICK':
        action_probs[auditory] = 1.0
    elif rule == 'SILENT':
        action_probs[1 - auditory] = 1.0
    elif rule == 'AL':
        # Alternate rule: opposite of previous action
        # Requires a valid previous action (cannot be NaN)
        if prev_action is None or np.isnan(prev_action):
            raise ValueError("AL rule requires a valid prev_action (cannot be NaN)")
        # Convert to int for indexing (handles float 0.0/1.0)
        action_probs[1 - int(prev_action)] = 1.0
    else:
        raise ValueError(f"Invalid rule: {rule}")
    return action_probs


def _option_name_part(feat: str, opt: Any) -> str:
    """Build variant name component for one (feature, option) pair."""
    if opt is None:
        return f"add{feat}"
    elif opt == "Ap":
        return f"{feat}linked"
    elif isinstance(opt, float) and (opt == 0.0 or opt == 1.0):
        return f"{feat}off"
    else:
        return f"{feat}_{opt}"


def _expand_sub_features(sub_tree: dict) -> List[Tuple[dict, List[str]]]:
    """
    Recursively expand a sub-feature tree into all (params_dict, name_parts) combos.

    Sub-features with >1 option are crossed and contribute to the variant name.
    Sub-features with exactly 1 option are implicit parameter settings — they set
    the parameter value but do not add a name component.
    """
    if not sub_tree:
        return [({}, [])]

    per_feat: List[List[Tuple[dict, List[str]]]] = []
    for feat, options in sub_tree.items():
        n_opts = len(options)
        feat_combos: List[Tuple[dict, List[str]]] = []
        for opt, sub_sub in options.items():
            base_params: dict = {feat: opt}
            name_parts: List[str] = [_option_name_part(feat, opt)] if n_opts > 1 else []
            for deep_p, deep_n in _expand_sub_features(sub_sub):
                feat_combos.append(({**base_params, **deep_p}, name_parts + deep_n))
        per_feat.append(feat_combos)

    result: List[Tuple[dict, List[str]]] = []
    for cross in product(*per_feat):
        merged_p: dict = {}
        merged_n: List[str] = []
        for p, n in cross:
            merged_p.update(p)
            merged_n.extend(n)
        result.append((merged_p, merged_n))
    return result


def _extract_params_from_tree_node(
    feature_name: str, option_value: Any, node: dict
) -> List[Tuple[dict, List[str]]]:
    """
    Extract parameter settings for one (top-level feature, option) pair.

    Returns a list of (params_dict, name_parts) tuples — one entry for most features, 
    multiple when the option has nested sub-features with multiple options (currently only E=None → Ba/Sa/Sda combos).

    Three cases:
    - E disabled: explicitly zeros Ba/Sa/Sda (absent from subtree when E=0).
    - LR/WsLs/Sr: mode-label features — option_value is not a model param;
      params come entirely from the subtree read via _expand_sub_features.
    - Everything else: option_value IS the param value; subtree expansion is
      a no-op for leaf features and produces sub-combos for E=None.
    """
    name = _option_name_part(feature_name, option_value)

    if feature_name == "E" and option_value is not None:
        # E disabled → Ba, Sa, Sda all fixed off (not present in subtree).
        return [({'E': option_value, 'Ba': 0.0, 'Sa': 0.0, 'Sda': 0.0}, [name])]

    elif feature_name in ('LR', 'WsLs', 'Sr'):
        # Mode-label: params live in the subtree, not keyed by feature_name.
        return [({**sp}, [name] + sn) for sp, sn in _expand_sub_features(node)]

    else:
        base: dict = {feature_name: option_value}
        return [({**base, **sp}, [name] + sn) for sp, sn in _expand_sub_features(node)]


def generate_variants_from_tree(feature_tree: dict, base_name: str = 'model') -> dict:
    """
    Generate all possible model variants from a feature tree.

    Nested sub-features (e.g. Ba, Sa, Sda under E=None) are fully expanded into combos with all other features, 
    so every independent choice becomes its own variant dimension.

    Parameters:
    -----------
    feature_tree : dict
        Tree structure like RULEQHYP_FEATURE_OPTIONS or QLEARN_FEATURE_OPTIONS
    base_name : str
        Base name for model variants (e.g., 'ruleQhyp_v0_minimal')

    Returns:
    --------
    variants : dict
        {variant_name: {'defaults': {...}, 'description': '...'}}
    """
    # For each top-level feature, collect ALL (params, name_parts) combos across
    # its options (including nested sub-feature expansions).
    per_feature_combos: List[List[Tuple[dict, List[str]]]] = []
    for feature, options in feature_tree.items():
        feature_combos: List[Tuple[dict, List[str]]] = []
        for opt_val, subtree in options.items():
            feature_combos.extend(
                _extract_params_from_tree_node(feature, opt_val, subtree)
            )
        per_feature_combos.append(feature_combos)

    variants: dict = {}
    for cross_combo in product(*per_feature_combos):
        defaults: dict = {}
        variant_parts: List[str] = []
        for params, name_parts in cross_combo:
            defaults.update(params)
            variant_parts.extend(name_parts)

        variant_name = (
            f"{base_name}_" + "_".join(variant_parts) if variant_parts else base_name
        )
        variants[variant_name] = {
            'defaults': defaults,
            'description': f"Features: {', '.join(variant_parts)}",
        }

    return variants


def sample_variants_for_family(
    feature_tree: dict,
    base_name: str,
    n_variants: Optional[int] = None,
    strategy: str = 'all'
) -> dict:
    """
    Sample model variants from a feature tree.
    
    Parameters:
    -----------
    feature_tree : dict
        Tree structure (RULEQHYP_FEATURE_OPTIONS or QLEARN_FEATURE_OPTIONS)
    base_name : str
        Base name for variants
    n_variants : int, optional
        Number of variants to sample. If None, returns all variants.
    strategy : str
        'all': return all variants
        'random': randomly sample n_variants
        'diverse': sample diverse variants (ensures different feature combinations)
        
    Returns:
    --------
    variants : dict
        Subset of variants from generate_variants_from_tree
    """
    all_variants = generate_variants_from_tree(feature_tree, base_name)
    
    if strategy == 'all' or n_variants is None or n_variants >= len(all_variants):
        return all_variants
    
    if strategy == 'random':
        variant_names = random.sample(list(all_variants.keys()), n_variants)
        return {name: all_variants[name] for name in variant_names}
    
    elif strategy == 'diverse':
        # TODO: Implement diverse sampling (ensure different feature combinations)
        # For now, just return random sample
        variant_names = random.sample(list(all_variants.keys()), n_variants)
        return {name: all_variants[name] for name in variant_names}
    
    return all_variants


def create_simulation_plan_for_family_comparison(
    ruleqhyp_tree: dict,
    qlearn_tree: dict,
    n_variants_per_family: int = 15,
    n_agents_per_variant: int = 667,  # ~10k total per family
    strategy: str = 'diverse'
) -> dict:
    """
    Create a simulation plan for comparing ruleQhyp vs Qlearn families.
    
    Parameters:
    -----------
    ruleqhyp_tree : dict
        RULEQHYP_FEATURE_OPTIONS tree
    qlearn_tree : dict
        QLEARN_FEATURE_OPTIONS tree
    n_variants_per_family : int
        Number of variants to sample from each family
    n_agents_per_variant : int
        Number of agents to simulate per variant
    strategy : str
        'diverse': sample diverse variants (recommended)
        'random': random sample
        'all': use all variants (may be too many)
        
    Returns:
    --------
    plan : dict
        {
            'ruleQhyp': {variant_name: {'defaults': {...}, 'n_agents': N}},
            'Qlearn': {variant_name: {'defaults': {...}, 'n_agents': N}}
        }
    """
    # Sample variants from each family
    ruleqhyp_variants = sample_variants_for_family(
        ruleqhyp_tree,
        'ruleQhyp_v0_minimal',
        n_variants=n_variants_per_family,
        strategy=strategy
    )
    
    qlearn_variants = sample_variants_for_family(
        qlearn_tree,
        'Qlearn_v0_minimal',
        n_variants=n_variants_per_family,
        strategy=strategy
    )
    
    # Add n_agents to each variant
    for variant_name in ruleqhyp_variants:
        ruleqhyp_variants[variant_name]['n_agents'] = n_agents_per_variant
    
    for variant_name in qlearn_variants:
        qlearn_variants[variant_name]['n_agents'] = n_agents_per_variant
    
    return {
        'ruleQhyp': ruleqhyp_variants,
        'Qlearn': qlearn_variants
    }
