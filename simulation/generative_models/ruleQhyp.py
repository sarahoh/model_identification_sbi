"""
Simulation function for ruleQhyp model - Q-learning with rule selection and attention.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Callable
from helpers.params import ParamConfig
from scipy.special import softmax
from ..constants import RULES, N_RULES, N_ACTIONS, N_RULES_PER_AGENT, MAX_TRIALS_PER_SESSION, MAX_SESSIONS_PER_RULE, MAX_TOTAL_TRIALS, PROB_SILENT_4TH_RULE
from ..utils import random_stimulus_generator, generate_adaptive_rule_sequence_from_dark_behavior, get_rule_policy


# Required parameters for ruleQhyp model
REQUIRED_PARAMS = [
    'Ap',     # learning rate for positive outcomes
    'An',     # learning rate for negative outcomes
    'Agcf',   # learning rate multiplier for CF (unchosen rule) updates
    'B',      # inv temp
    'E',      # noise
    'Ft',     # between-trial forgetting
    'Fb',     # between-block forgetting
    'Ex',     # exploit threshold
    'L',      # lapse rate
    'R',      # recovery rate
    'Ba',     # side (action) bias weight (heuristic)
    'Ws',     # win-stay weight (contributes to heuristic when attentive)
    'Ls',     # lose-shift weight (contributes to heuristic when attentive)
    'Sa',     # sticky action weight (heuristic)
    'Sda',    # sticky action decay (heuristic)
    'Sr',     # sticky rule weight (modifies rule selection when attentive)
    'Sdr',    # sticky rule decay (modifies rule selection when attentive)
    'Dle',    # default Q-value go left
    'Dri',    # default Q-value go right
    'Dli',    # default Q-value go light
    'Dda',    # default Q-value go dark
    'Dcl',    # default Q-value go click
    'Dsi',    # default Q-value go silent
    'Dal',    # default Q-value alternate
]

def generate(
    sim_params: np.ndarray,
    model_spec: ParamConfig,
    seed: int
) -> Dict[str, Any]:
    """
    Generate responses using ruleQhyp model for one agent.
    
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
    
    alpha_pos = params_dict['Ap']
    alpha_neg = params_dict['An']
    alpha_gain_cf = params_dict['Agcf']
    beta = params_dict['B']
    epsilon = params_dict['E']
    forget_trial = params_dict['Ft']
    forget_block = params_dict['Fb']
    exploit_threshold = params_dict['Ex']
    lapse = params_dict['L']    # P(engaged -> random)
    recover = params_dict['R']  # P(random -> engaged)
    bias_weight = params_dict['Ba']
    ws_weight = params_dict['Ws']  # Win-stay heuristic weight (optional, attentive only)
    ls_weight = params_dict['Ls']  # Lose-shift heuristic weight (optional, attentive only)
    sticky_act_weight = params_dict['Sa']
    sticky_act_decay = params_dict['Sda']
    sticky_rule_weight = params_dict['Sr']
    sticky_rule_decay = params_dict['Sdr']

    # Default Q-values
    # Order must match RULES (ALL_RULES): ['LEFT', 'RIGHT', 'LIGHT', 'DARK', 'CLICK', 'SILENT', 'AL']
    Q_default = np.array([
        params_dict['Dle'],  # goLeft (index 0)
        params_dict['Dri'],  # goRight (index 1)
        params_dict['Dli'],  # goLight (index 2)
        params_dict['Dda'],  # goDark (index 3)
        params_dict['Dcl'],  # goClick (index 4)
        params_dict['Dsi'],  # goSilent (index 5)
        params_dict['Dal']   # alternate (index 6)
    ])
    rule_q_values = Q_default.copy()  # Shape: (7,) - all 7 rules available for trial selection
    
    # Constants and pre-computed indices (for performance)
    stim_generator = random_stimulus_generator(seed)
    rng = np.random.default_rng(seed)
    AL_RULE_IDX = RULES.index('AL')  # Cache this to avoid repeated lookups

    # Adaptive rule sequence (experimental protocol):
    # DARK → LEFT/RIGHT (opposite of DARK preference) → AL → SILENT (75% chance)
    
    # Track session statistics per rule for criterion checking
    criterion_tracking = {}

    # store all info
    all_visual_cues = []
    all_auditory_cues = []
    all_prev_actions = []
    all_prev_rewards = []
    all_correct_actions = []
    all_chosen_rules = []
    all_actions = []
    all_corrects = []
    all_rewards = []
    all_trial_idx = []
    all_q_values = []
    all_attention_state = []
    all_exploit_mode = []
    all_rule = []
    all_session_idx = []
    all_new_session = []
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
            break  # Stop all simulation if we've hit the cap

        criterion_met = False  # Reset for each new rule
        n_sessions_current_rule = 0

        while not criterion_met and n_sessions_current_rule < MAX_SESSIONS_PER_RULE and not hit_total_trial_cap: # while criterion not met, repeat sessions of same rule
            # Check total trial cap before starting new session
            total_trials_so_far = sum(len(arr) for arr in all_actions)
            if total_trials_so_far >= MAX_TOTAL_TRIALS:
                hit_total_trial_cap = True
                break  # Stop this rule if we've hit the cap

            # preallocate arrays 
            session_visual_cues = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_auditory_cues = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_prev_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_prev_rewards = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_correct_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)  
            session_chosen_rules = np.empty(MAX_TRIALS_PER_SESSION, dtype=object)
            session_actions = np.empty(MAX_TRIALS_PER_SESSION, dtype=float)
            session_corrects = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_rewards = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_q_values = np.empty((MAX_TRIALS_PER_SESSION, N_RULES), dtype=float)
            session_attention_state = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_exploit_mode = np.empty(MAX_TRIALS_PER_SESSION, dtype=int)
            session_new_session = np.zeros(MAX_TRIALS_PER_SESSION, dtype=int)
            session_new_session[0] = 1  # First trial of session is marked as new session

            prev_action = np.nan
            prev_reward = np.nan
            tr_this_sess = 0
            is_exploiting = False  # Track exploit mode state

            sticky_act_trace = np.zeros(N_ACTIONS)
            sticky_rule_trace = np.zeros(N_RULES)

            # apply between-session forgetting
            rule_q_values = (1 - forget_block) * rule_q_values + forget_block * Q_default
            
            is_engaged = True  # start engaged

            while tr_this_sess < MAX_TRIALS_PER_SESSION:
                # Check total trial cap during session (prevents going over limit mid-session)
                total_trials_so_far = sum(len(arr) for arr in all_actions) + tr_this_sess
                if total_trials_so_far >= MAX_TOTAL_TRIALS:
                    break  # Stop session if we've hit the cap
                
                visual, auditory = next(stim_generator)
                # Get correct action based on rule policy
                if current_rule == 'AL' and np.isnan(prev_action):
                    correct_action = np.nan # undefined if first trial of an AL session
                else:
                    correct_action = rng.choice(N_ACTIONS, p=get_rule_policy(current_rule, visual, auditory, prev_action))
                
                # apply between-trial forgetting
                rule_q_values = (1 - forget_trial) * rule_q_values + forget_trial * Q_default

                # heuristic contribution (base: sticky action + side bias)
                heuristic_logits = sticky_act_weight * sticky_act_trace + bias_weight * np.array([1, 0])
                # Add win-stay-lose-shift heuristic if engaged (requires attention to prev action/outcome)
                if is_engaged and not np.isnan(prev_reward) and not np.isnan(prev_action):
                    if prev_reward == 1:
                        # Win-stay: bias toward repeating previous action
                        heuristic_logits += ws_weight * np.array([prev_action == 0, prev_action == 1], dtype=float)
                    elif prev_reward == 0:
                        # Lose-shift: bias toward opposite of previous action
                        heuristic_logits += ls_weight * np.array([prev_action == 1, prev_action == 0], dtype=float)
                heuristic_policy = softmax(heuristic_logits)

                chosen_rule_idx = -1
                chosen_rule = None
                if is_engaged:
                    # Compute rule logits
                    rule_logits = beta * rule_q_values + sticky_rule_weight * sticky_rule_trace
                    if np.isnan(prev_action): # Exclude AL rule from selection if prev_action is NaN (can't alternate without previous action)
                        rule_logits[AL_RULE_IDX] = -np.inf  # Set to -inf so softmax gives 0 probability
                    rule_probs = softmax(rule_logits)  # Softmax automatically handles -inf and ensures probabilities sum to 1
                    
                    # Check if we should enter/exit exploit mode (max Q value > threshold)
                    # Q-values encode the agent's belief about rule quality (expected reward)
                    # High Q-value means strong belief that a rule is good/active
                    # For exploit mode, exclude AL if prev_action is NaN
                    if np.isnan(prev_action):
                        # Avoid copy - use view and temporary modification
                        temp_q = rule_q_values
                        temp_al_val = temp_q[AL_RULE_IDX]
                        temp_q[AL_RULE_IDX] = -np.inf
                        max_q_value = np.max(temp_q)
                        max_q_rule_idx = np.argmax(temp_q)
                        temp_q[AL_RULE_IDX] = temp_al_val  # Restore
                    else:
                        max_q_value = np.max(rule_q_values)
                        max_q_rule_idx = np.argmax(rule_q_values)
                    
                    if max_q_value > exploit_threshold:
                        is_exploiting = True
                        chosen_rule_idx = max_q_rule_idx
                    else:
                        is_exploiting = False
                        chosen_rule_idx = rng.choice(N_RULES, p=rule_probs)
                    
                    chosen_rule = RULES[chosen_rule_idx]
                    rule_policy = get_rule_policy(chosen_rule, visual, auditory, prev_action)
                    action_probs = (1 - epsilon) * rule_policy + epsilon * heuristic_policy
                else:
                    action_probs = heuristic_policy

                # select action
                action = rng.choice(N_ACTIONS, p=action_probs) 
                # Check correctness (handle NaN case for AL rule on first trial)
                if np.isnan(correct_action):
                    correct = 0  # Can't be correct if correct action is undefined
                else:
                    correct = int(action == correct_action)
                # observe outcome
                reward = correct # deterministic feedback
                
                if is_engaged:
                    # determine learning rate
                    alpha = alpha_pos
                    if reward <= 0:  # negative rpe
                        alpha = alpha_neg
                    # update chosen rule (toward outcome)
                    delta = reward - rule_q_values[chosen_rule_idx]
                    rule_q_values[chosen_rule_idx] += alpha * delta

                    # update unchosen rules (counterfactual; away from outcome, vectorized)
                    alpha_cf = alpha * alpha_gain_cf
                    unchosen_rule_idxs = np.arange(N_RULES) != chosen_rule_idx
                    deltas_cf = (1 - reward) - rule_q_values[unchosen_rule_idxs]
                    rule_q_values[unchosen_rule_idxs] += alpha_cf * deltas_cf

                # update sticky rule trace
                sticky_rule_trace = sticky_rule_trace * (1 - sticky_rule_decay) # decay
                if is_engaged:
                    sticky_rule_trace[chosen_rule_idx] = 1.0

                # update sticky action trace
                sticky_act_trace = sticky_act_trace * (1 - sticky_act_decay)
                sticky_act_trace[action] = 1.0
    
                # store info
                session_visual_cues[tr_this_sess] = visual
                session_auditory_cues[tr_this_sess] = auditory
                session_prev_actions[tr_this_sess] = prev_action
                session_prev_rewards[tr_this_sess] = prev_reward
                session_correct_actions[tr_this_sess] = correct_action
                session_chosen_rules[tr_this_sess] = chosen_rule
                session_actions[tr_this_sess] = action
                session_corrects[tr_this_sess] = correct
                session_rewards[tr_this_sess] = reward
                # Use direct assignment instead of copy (much faster)
                session_q_values[tr_this_sess] = rule_q_values
                session_attention_state[tr_this_sess] = is_engaged
                session_exploit_mode[tr_this_sess] = is_exploiting
                prev_action = action
                prev_reward = reward
                tr_this_sess += 1

                # update engagement state
                if is_engaged: 
                    if rng.random() < lapse:
                        is_engaged = False # switch to unengaged
                else:
                    if rng.random() < recover:
                        is_engaged = True # switch to engaged

            # truncate session arrays to the actual number of trials
            all_visual_cues.append(session_visual_cues[:tr_this_sess])
            all_auditory_cues.append(session_auditory_cues[:tr_this_sess])
            all_prev_actions.append(session_prev_actions[:tr_this_sess])
            all_prev_rewards.append(session_prev_rewards[:tr_this_sess])
            all_correct_actions.append(session_correct_actions[:tr_this_sess])
            all_chosen_rules.append(session_chosen_rules[:tr_this_sess])
            all_actions.append(session_actions[:tr_this_sess])
            all_corrects.append(session_corrects[:tr_this_sess])
            all_rewards.append(session_rewards[:tr_this_sess])
            all_q_values.append(session_q_values[:tr_this_sess, :])
            all_attention_state.append(session_attention_state[:tr_this_sess])
            all_exploit_mode.append(session_exploit_mode[:tr_this_sess])
            all_trial_idx.append(np.arange(tr_this_sess))
            all_rule.append(np.full(tr_this_sess, current_rule))
            all_session_idx.append(np.full(tr_this_sess, absolute_session_idx))
            all_new_session.append(session_new_session[:tr_this_sess])
            absolute_session_idx += 1

            n_sessions_current_rule += 1
            
            # Check total trial cap after session ends (safety net for extreme parameter combinations)
            total_trials_so_far = sum(len(arr) for arr in all_actions)
            if total_trials_so_far >= MAX_TOTAL_TRIALS:
                hit_total_trial_cap = True
                break  # Stop this rule if we've hit the cap
    
            # After session ends, check next-rule criterion
            session_rewards_used = session_rewards[:tr_this_sess]
            if tr_this_sess == 0:
                session_acc = 0.0
                last_20_sum = 0
            else:
                session_acc = float(np.sum(session_rewards_used)) / tr_this_sess
                last_20_sum = int(np.sum(session_rewards_used[-20:]))
            # Get previous session info for same rule
            criterion_tracking_prev = criterion_tracking.get(current_rule, None)
            
            # Check the three criteria:
            # 1. >=75% (3/4) accuracy in current session (standalone)
            met_75_percent = (session_acc >= 0.75)
            # 2. >=70% (7/10) accuracy in current session AND >=65% in previous session (same rule)
            met_combined_accuracy = False
            if criterion_tracking_prev is not None:
                prev_session_acc = criterion_tracking_prev.get('accuracy', 0.0)
                if prev_session_acc >= 0.65 and session_acc >= 0.70:
                    met_combined_accuracy = True
            # 3. Two consecutive same-rule sessions with last 20 >= 18/20
            consecutive_18_20 = False
            if criterion_tracking_prev is not None:
                prev_last_20_sum = criterion_tracking_prev.get('last_20_sum', 0)
                if prev_last_20_sum >= 18 and last_20_sum >= 18:
                    consecutive_18_20 = True
            
            # Criterion is met if any of the three conditions are true
            criterion_met = consecutive_18_20 or met_75_percent or met_combined_accuracy
            
            # Store info for this session (for next session of same rule)
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
    
    # Create trial_data DataFrame with all variables (observed + latent)
    # Observed variables: visual_cue, auditory_cue, action, reward, correct, etc.
    # Latent variables: chosen_rule, attention_state, exploit_mode, Q-values
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
        'rule': np.concatenate(all_rule),  # True rule in effect (experimenter-defined)
        'session_idx': np.concatenate(all_session_idx),
        'new_session': np.concatenate(all_new_session),  # 1 if first trial of session, 0 otherwise
        'chosen_rule': np.concatenate(all_chosen_rules),
        'attention_state': np.concatenate(all_attention_state),
        'exploit_mode': np.concatenate(all_exploit_mode)
    })
    # Add Q-values as separate columns
    all_q_values_array = np.concatenate(all_q_values, axis=0)  # Shape: (n_trials, N_RULES)
    for i, rule_name in enumerate(RULES):
        trial_data[f'Q_{rule_name}'] = all_q_values_array[:, i]
    
    # Create return dictionary
    # Structure: data (DataFrames) + metadata + labels (for ML training)
    result = {
        # Data (for analysis/exploration)
        'trial_data': trial_data,
        
        # Parameters (both formats for different use cases)
        'sim_params': sim_params,  # Raw array of free parameters (for optimization/SBI)
        'params_dict': params_dict,  # Filled dictionary with all parameters (for analysis)
        
        # Metadata
        'base_model': model_spec.base_name,  # Base model name
        'model_id': model_spec.model_name,  # Model identifier (canonical ID)
        'model_spec': model_spec,  # Full parameter configuration
        'seed': seed,
    }
    
    return result

