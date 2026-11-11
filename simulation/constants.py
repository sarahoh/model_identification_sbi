# All possible rules (for reference and trial-by-trial selection in generative model)
ALL_RULES = ['LEFT', 'RIGHT', 'LIGHT', 'DARK', 'CLICK', 'SILENT', 'AL']

# Rules actually used in the experimental data (for sequence generation)
# Analysis shows only these 5 rules appear: DARK, LEFT, RIGHT, AL, SILENT
# All experimental sequences start with DARK
TASK_RULES = ['DARK', 'LEFT', 'RIGHT', 'AL', 'SILENT']

# Rules available for trial-by-trial selection in generative model (all 7 rules)
RULES = ALL_RULES
N_RULES = len(RULES)  # 7 rules for trial selection
N_ACTIONS = 2
# Q-learning state space: visual × auditory × prev_action (see generative_models.Qlearn)
N_STATES = 8

N_RULES_PER_AGENT = 4
MAX_SESSIONS_PER_RULE = 9  # Covers typical learning, with headroom for rare extremes (max observed = 9)
MAX_TRIALS_PER_SESSION = 300
MAX_SESSIONS = MAX_SESSIONS_PER_RULE * N_RULES_PER_AGENT  # Max possible sessions = 36
MAX_TOTAL_TRIALS = MAX_SESSIONS * MAX_TRIALS_PER_SESSION  # = 10,800 (covers real rat max: 5,637)

# Adaptive rule sequence generation (matching experimental protocol)
# Pattern: DARK → (opposite side) → AL → SILENT (or stop at 3)
# Probability of adding SILENT as 4th rule (vs stopping at 3 rules)
PROB_SILENT_4TH_RULE = 0.75  # ~75% of rats got 4 rules, ~25% stopped at 3

