"""
Simulation function for Q-learning model - direct state-action Q-learning.

This model learns Q-values for state-action pairs directly, where states are
combinations of visual cue, auditory cue, and optionally previous action/reward.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from helpers.params import ParamConfig
from scipy.special import softmax
from ..constants import N_ACTIONS, N_STATES, MAX_TRIALS_PER_SESSION, MAX_SESSIONS_PER_RULE, MAX_TOTAL_TRIALS, PROB_SILENT_4TH_RULE
from ..utils import random_stimulus_generator, generate_adaptive_rule_sequence_from_dark_behavior, get_rule_policy


# Required parameters for Qlearn model
REQUIRED_PARAMS = [
    'Ap',     # learning rate for positive outcomes
    'An',     # learning rate for negative outcomes
    'Agcf',   # learning rate multiplier for CF (unchosen action/state) updates
    'B',      # inv temp
    'E',      # noise
    'Ft',     # between-trial forgetting
    'Fb',     # between-block forgetting
    'L',      # lapse rate
    'R',      # recovery rate
    'Ba',     # side (action) bias weight (heuristic)
    'Ws',     # win-stay weight (contributes to heuristic when attentive)
    'Ls',     # lose-shift weight (contributes to heuristic when attentive)
    'Sa',     # sticky action weight (heuristic)
    'Sda',    # sticky action decay (heuristic)
    'Dq',     # default Q-value (uniform initialization)
]


def _get_state_index(visual: int, auditory: int, prev_action: Optional[float]) -> int:
    """
    Convert state components to a single state index.
    
    State space is fixed to VisAudAct (8 states):
    - visual (0 or 1)
    - auditory (0 or 1)  
    - prev_action (0 or 1)
    
    Returns state index: visual*4 + auditory*2 + prev_action
    """
    # Handle missing prev_action (first trial)
    if prev_action is None or np.isnan(prev_action):
        prev_action = 0  # Default to 0 if missing
    
    # Convert to int for indexing
    prev_action_int = int(prev_action)
    
    # State index: visual*4 + auditory*2 + prev_action
    state_idx = visual * 4 + auditory * 2 + prev_action_int
    
    return state_idx


def generate(
    sim_params: np.ndarray,
    model_spec: ParamConfig,
    seed: int
) -> Dict[str, Any]:
    """
    Generate responses using Q-learning model for one agent.
    
    Uses fixed state space: VisAudAct (8 states: visual × auditory × prev_action)
    
    Parameters:
    -----------
    sim_params : np.ndarray
        Array of parameter values to simulate (only free parameters)
    model_spec : ParamConfig
        Parameter configuration dictionary (should have model_name set for labels)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - trial_data : pandas DataFrame with all trial-level data (observed + latent variables)
        - sim_params : numpy array of free parameters (raw input)
        - params_dict : dictionary with all parameters (filled with defaults)
        - base_model : str base model name
        - model_id : str model identifier
        - model_spec : ParamConfig used for simulation
        - seed : int random seed
    """
    # Validate that model_spec has all required parameters
    model_spec.validate_required(REQUIRED_PARAMS)
    
    # Fill parameter values (fitted params + defaults)
    params_dict = model_spec.fill_parameter_values(sim_params)
    
    # Extract parameters (using similar naming to ruleQhyp for consistency)
    alpha_pos = params_dict['Ap']
    alpha_neg = params_dict['An']
    alpha_gain_cf = params_dict['Agcf']
    beta = params_dict['B']
    epsilon = params_dict['E']
    forget_trial = params_dict['Ft']
    forget_block = params_dict['Fb']
    lapse = params_dict['L']    # P(engaged -> random)
    recover = params_dict['R']  # P(random -> engaged)
    bias_weight = params_dict['Ba']
    ws_weight = params_dict['Ws']  # Win-stay heuristic weight (optional, attentive only)
    ls_weight = params_dict['Ls']  # Lose-shift heuristic weight (optional, attentive only)
    sticky_act_weight = params_dict['Sa']
    sticky_act_decay = params_dict['Sda']
    Q_default_value = params_dict['Dq']  # Default Q-value for all state-action pairs
    
    # Initialize Q-values: (N_STATES, N_ACTIONS)
    # State space is fixed to VisAudAct (8 states)
    Q_default = np.full((N_STATES, N_ACTIONS), Q_default_value, dtype=float)
    Q = Q_default.copy()
    
    # Constants
    stim_generator = random_stimulus_generator(seed)
    rng = np.random.default_rng(seed)
    
    # Adaptive rule sequence (experimental protocol):
    # DARK → LEFT/RIGHT (opposite of DARK preference) → AL → SILENT (75% chance)
    
    # Track session statistics per rule for criterion checking
    criterion_tracking = {}
    
    # Store all info
    all_visual_cues = []
    all_auditory_cues = []
    all_prev_actions = []
    all_prev_rewards = []
    all_correct_actions = []
    all_actions = []
    all_corrects = []
    all_rewards = []
    all_trial_idx = []
    all_q_values = []  # Will store Q-values for all state-action pairs
    all_attention_state = []
    all_rule = []
    all_session_idx = []
    all_state_indices = []  # Track which state was used on each trial
    all_new_session = []
    # Monotonic session id for the whole agent (see ruleQhyp: per-rule counter is not unique).
    absolute_session_idx = 0

    # Start with DARK, determine next rules adaptively
    current_rule = 'DARK'
    dark_actions = []  # Track DARK actions for adaptive rule selection
    hit_total_trial_cap = False
    rule_number = 0  # Track which rule we're on (0-indexed)
    
    while current_rule is not None:
        # Check total trial cap before starting new rule
        total_trials_so_far = sum(len(arr) for arr in all_actions)
        if total_trials_so_far >= MAX_TOTAL_TRIALS:
            break
        
        criterion_met = False
        n_sessions_current_rule = 0
        
        while not criterion_met and n_sessions_current_rule < MAX_SESSIONS_PER_RULE and not hit_total_trial_cap:
            # Check total trial cap before starting new session
            total_trials_so_far = sum(len(arr) for arr in all_actions)
            if total_trials_so_far >= MAX_TOTAL_TRIALS:
                hit_total_trial_cap = True
                break
            
            # Preallocate arrays
            session_visual_cues = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_auditory_cues = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_prev_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_prev_rewards = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_correct_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_corrects = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_rewards = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_q_values = np.empty((MAX_TRIALS_PER_SESSION, N_STATES, N_ACTIONS), dtype=float)
            session_attention_state = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_state_indices = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_new_session = np.zeros(MAX_TRIALS_PER_SESSION, dtype=int)
            session_new_session[0] = 1  # First trial of session is marked as new session
            
            prev_action = np.nan
            prev_reward = np.nan
            tr_this_sess = 0
            
            sticky_act_trace = np.zeros(N_ACTIONS)
            
            # Apply between-session forgetting
            Q = (1 - forget_block) * Q + forget_block * Q_default
            
            is_engaged = True  # Start engaged
            
            while tr_this_sess < MAX_TRIALS_PER_SESSION:
                # Check total trial cap during session
                total_trials_so_far = sum(len(arr) for arr in all_actions) + tr_this_sess
                if total_trials_so_far >= MAX_TOTAL_TRIALS:
                    break
                
                visual, auditory = next(stim_generator)
                
                # Get correct action based on rule policy
                if current_rule == 'AL' and np.isnan(prev_action):
                    correct_action = np.nan
                else:
                    correct_action = rng.choice(N_ACTIONS, p=get_rule_policy(current_rule, visual, auditory, prev_action))
                
                # Apply between-trial forgetting
                Q = (1 - forget_trial) * Q + forget_trial * Q_default
                
                # Get current state index (VisAudAct: 8 states)
                state_idx = _get_state_index(visual, auditory, prev_action)
                
                # Heuristic contribution (base: sticky action + side bias)
                bias_act = np.array([1.0, 0.0])  # Bias toward left
                heuristic_logits = sticky_act_weight * sticky_act_trace + bias_weight * bias_act
                # Add win-stay-lose-shift heuristic if engaged (requires attention to prev action/outcome)
                if is_engaged and not np.isnan(prev_reward) and not np.isnan(prev_action):
                    if prev_reward == 1:
                        # Win-stay: bias toward repeating previous action
                        heuristic_logits += ws_weight * np.array([prev_action == 0, prev_action == 1], dtype=float)
                    elif prev_reward == 0:
                        # Lose-shift: bias toward opposite of previous action
                        heuristic_logits += ls_weight * np.array([prev_action == 1, prev_action == 0], dtype=float)
                heuristic_policy = softmax(heuristic_logits)
                
                # Choice probabilities
                if is_engaged:
                    # Use Q-values for current state
                    q_logits = beta * Q[state_idx, :]
                    # Combine Q-learning with heuristic (similar to ruleQhyp)
                    action_probs = (1 - epsilon) * softmax(q_logits) + epsilon * heuristic_policy
                else:
                    # Inattentive: use heuristic only
                    action_probs = heuristic_policy
                
                # Select action
                action = rng.choice(N_ACTIONS, p=action_probs)
                
                # Check correctness
                if np.isnan(correct_action):
                    correct = 0
                else:
                    correct = int(action == correct_action)
                
                # Observe outcome
                reward = correct  # Deterministic feedback
                
                # Update Q-values (only if engaged)
                if is_engaged:
                    # Determine learning rate
                    alpha = alpha_pos if reward > 0 else alpha_neg
                    
                    # Update chosen state-action pair
                    delta = reward - Q[state_idx, action]
                    Q[state_idx, action] += alpha * delta
                    
                    # Counterfactual updates (unchosen action in current state)
                    alpha_cf = alpha * alpha_gain_cf
                    other_action = 1 - action
                    delta_cf = (1 - reward) - Q[state_idx, other_action]
                    Q[state_idx, other_action] += alpha_cf * delta_cf
                    
                    # Counterfactual updates (chosen action in other states)
                    other_states = np.arange(N_STATES) != state_idx
                    if np.any(other_states):
                        deltas_cf = (1 - reward) - Q[other_states, action]
                        Q[other_states, action] += alpha_cf * deltas_cf
                
                # Update sticky action trace
                sticky_act_trace = sticky_act_trace * (1 - sticky_act_decay)
                sticky_act_trace[action] = 1.0
                
                # Store info
                session_visual_cues[tr_this_sess] = visual
                session_auditory_cues[tr_this_sess] = auditory
                session_prev_actions[tr_this_sess] = prev_action
                session_prev_rewards[tr_this_sess] = prev_reward
                session_correct_actions[tr_this_sess] = correct_action
                session_actions[tr_this_sess] = action
                session_corrects[tr_this_sess] = correct
                session_rewards[tr_this_sess] = reward
                session_q_values[tr_this_sess] = Q.copy()
                session_attention_state[tr_this_sess] = is_engaged
                session_state_indices[tr_this_sess] = state_idx
                
                prev_action = action
                prev_reward = reward
                tr_this_sess += 1
                
                # Update engagement state
                if is_engaged:
                    if rng.random() < lapse:
                        is_engaged = False
                else:
                    if rng.random() < recover:
                        is_engaged = True
            
            # Truncate session arrays
            all_visual_cues.append(session_visual_cues[:tr_this_sess])
            all_auditory_cues.append(session_auditory_cues[:tr_this_sess])
            all_prev_actions.append(session_prev_actions[:tr_this_sess])
            all_prev_rewards.append(session_prev_rewards[:tr_this_sess])
            all_correct_actions.append(session_correct_actions[:tr_this_sess])
            all_actions.append(session_actions[:tr_this_sess])
            all_corrects.append(session_corrects[:tr_this_sess])
            all_rewards.append(session_rewards[:tr_this_sess])
            all_q_values.append(session_q_values[:tr_this_sess])
            all_attention_state.append(session_attention_state[:tr_this_sess])
            all_state_indices.append(session_state_indices[:tr_this_sess])
            all_trial_idx.append(np.arange(tr_this_sess))
            all_rule.append(np.full(tr_this_sess, current_rule))
            all_session_idx.append(np.full(tr_this_sess, absolute_session_idx))
            all_new_session.append(session_new_session[:tr_this_sess])
            absolute_session_idx += 1

            n_sessions_current_rule += 1
            
            # Check total trial cap after session ends
            total_trials_so_far = sum(len(arr) for arr in all_actions)
            if total_trials_so_far >= MAX_TOTAL_TRIALS:
                hit_total_trial_cap = True
                break
            
            # Check criterion (same as ruleQhyp)
            # IMPORTANT: only use the filled portion of preallocated arrays
            session_rewards_used = session_rewards[:tr_this_sess]
            if tr_this_sess == 0:
                session_acc = 0.0
                last_20_sum = 0
            else:
                session_acc = float(np.sum(session_rewards_used)) / tr_this_sess
                last_20_sum = int(np.sum(session_rewards_used[-20:]))
            criterion_tracking_prev = criterion_tracking.get(current_rule, None)
            
            met_75_percent = (session_acc >= 0.75)
            met_combined_accuracy = False
            if criterion_tracking_prev is not None:
                prev_session_acc = criterion_tracking_prev.get('accuracy', 0.0)
                if prev_session_acc >= 0.65 and session_acc >= 0.70:
                    met_combined_accuracy = True
            consecutive_18_20 = False
            if criterion_tracking_prev is not None:
                prev_last_20_sum = criterion_tracking_prev.get('last_20_sum', 0)
                if prev_last_20_sum >= 18 and last_20_sum >= 18:
                    consecutive_18_20 = True
            
            criterion_met = consecutive_18_20 or met_75_percent or met_combined_accuracy
            
            session_count = criterion_tracking.get(current_rule, {}).get('session_count', 0) + 1
            criterion_tracking[current_rule] = {
                'last_20_sum': last_20_sum,
                'accuracy': session_acc,
                'session_count': session_count
            }
            
            # Track DARK actions for adaptive rule selection
            if current_rule == 'DARK':
                dark_actions.extend(session_actions[:tr_this_sess].tolist())
        
        # Determine next rule adaptively when current rule ends
        if current_rule == 'DARK':
            # Choose opposite of DARK preference
            dark_actions_array = np.array(dark_actions)
            rule_sequence = generate_adaptive_rule_sequence_from_dark_behavior(dark_actions_array, rng)
            current_rule = rule_sequence[1]  # Second rule (LEFT or RIGHT)
        elif current_rule in ['LEFT', 'RIGHT']:
            # Go to AL
            current_rule = 'AL'
        elif current_rule == 'AL':
            # 75% chance go to SILENT, otherwise stop
            if rng.random() < PROB_SILENT_4TH_RULE:
                current_rule = 'SILENT'
            else:
                current_rule = None  # Stop
        else:
            # SILENT or any other rule → stop
            current_rule = None
        
        # Increment rule counter if we're continuing to a new rule
        if current_rule is not None:
            rule_number += 1
    
    # Create trial_data DataFrame
    trial_data = pd.DataFrame({
        'visual_cue': np.concatenate(all_visual_cues),
        'auditory_cue': np.concatenate(all_auditory_cues),
        'prev_action': np.concatenate(all_prev_actions),
        'prev_reward': np.concatenate(all_prev_rewards),
        'action': np.concatenate(all_actions),
        'reward': np.concatenate(all_rewards),
        'correct_action': np.concatenate(all_correct_actions),
        'correct': np.concatenate(all_corrects),
        'trial_idx': np.concatenate(all_trial_idx),
        'rule': np.concatenate(all_rule),
        'session_idx': np.concatenate(all_session_idx),
        'new_session': np.concatenate(all_new_session),  # 1 if first trial of session, 0 otherwise
        'attention_state': np.concatenate(all_attention_state),
        'state_idx': np.concatenate(all_state_indices)
    })
    
    # Add Q-values as separate columns (flatten state-action pairs)
    all_q_values_array = np.concatenate(all_q_values, axis=0)  # Shape: (n_trials, N_STATES, N_ACTIONS)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            trial_data[f'Q_s{s}_a{a}'] = all_q_values_array[:, s, a]
    
    # Create return dictionary
    result = {
        'trial_data': trial_data,
        'sim_params': sim_params,
        'params_dict': params_dict,
        'base_model': model_spec.base_name,
        'model_id': model_spec.model_name,
        'model_spec': model_spec,
        'seed': seed,
    }
    
    return result
