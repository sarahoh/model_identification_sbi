"""
General simulation wrapper for generating agent data with labels.
Supports multiple model types and keeps simulation logic separate from training metadata.

Key Functions:
--------------
- simulate_agent(): Generate one agent with given parameters
- simulate_agent_with_retries(): Generate one agent, resampling params until quality checks pass
- simulate_all_agents_parallel(): Generate many agents with checkpointing

Example Usage:
--------------
# Basic simulation
from simulation.simulate import simulate_agent_with_retries
from simulation.generative_models.ruleQhyp import generate
from CONSTRAINED_MODEL_SPECS import RULEQHYP_BASE_MODEL_SPEC_CONSTRAINED

all_trial_data = []
all_metadata = {}

for agent_id in range(50000):
    trial_data, metadata = simulate_agent_with_retries(
        sim_func=generate,
        model_spec=RULEQHYP_BASE_MODEL_SPEC_CONSTRAINED,
        agent_id=agent_id,
        seed=agent_id,
        max_retries=10,
        max_sessions_per_rule=9,
        min_performance=0.3,
        verbose=(agent_id < 100)
    )
    
    all_trial_data.append(trial_data)
    all_metadata[agent_id] = metadata
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional, Tuple, List
from pathlib import Path
import shutil
import pickle
import gc
from helpers.params import ParamConfig


def simulate_agent(
    sim_func: Callable,
    model_spec: ParamConfig,
    agent_id: int,
    seed: Optional[int] = None,
    param_sampler: Optional[Callable] = None,
    create_labels_func: Optional[Callable] = None,
    variant_settings: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate one agent with a specific model version.
    
    This is a general wrapper that can work with any simulation function.
    
    Parameters:
    -----------
    sim_func : Callable
        Simulation function with signature: generate(sim_params, model_spec, seed) -> Dict
        Must return a dict with 'trial_data' (DataFrame), 'sim_params', 'params_dict',
        'model_id', 'model_spec', 'seed', 'base_model'
    model_spec : ParamConfig
        Parameter configuration for this model version
    agent_id : int
        Unique agent identifier
    seed : int, optional
        Random seed (default: agent_id)
    param_sampler : Callable, optional
        Function to sample parameters. Should take (model_spec, seed) and return
        np.ndarray of parameter values. Default: uniform sampling from bounds.
    create_labels_func : Callable, optional
        Function to create labels from simulation result. If None, not used.
    variant_settings : dict, optional
        Dictionary of variant settings used to create this model_spec.
        This records which features/variants were selected from the feature tree.
        
    Returns:
    --------
    trial_data : pd.DataFrame
        DataFrame with all trial-level data (observed + latent variables),
        with agent_id, model_version, and base_model columns added
    metadata : dict
        Dictionary containing agent_id, model_id, params_dict, sim_params, seed, variant_defaults
    """
    
    if seed is None:
        seed = agent_id
    
    # Sample free parameters
    if param_sampler is None:
        # Default: uniform sampling from parameter bounds
        free_params = model_spec.get_params_to_fit()
        rng = np.random.default_rng(seed)
        sim_params = np.array([
            rng.uniform(model_spec.params[name].lb, model_spec.params[name].ub)
            for name in free_params
        ])
    else:
        sim_params = param_sampler(model_spec, seed)
    
    # Run simulation
    result = sim_func(sim_params, model_spec, seed)
    
    # Extract trial_data and add metadata columns
    trial_data = result['trial_data'].copy()
    trial_data['agent_id'] = agent_id
    trial_data['model_version'] = result['model_id']
    trial_data['base_model'] = result['base_model']
    
    # Extract metadata (everything needed for labels)
    metadata = {
        'agent_id': agent_id,  # Include agent_id for easy linking to data
        'model_id': result['model_id'],
        'model_spec': result['model_spec'],
        'base_model': result['base_model'],
        'params_dict': result.get('params_dict', {}),
        'sim_params': result.get('sim_params', sim_params),
        'seed': result.get('seed', seed),
        'variant_settings': variant_settings  # Tree settings used to create this variant
    }
    
    return trial_data, metadata


def _simulate_variant_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments for simulate_single_variant.
    """
    return simulate_single_variant(*args)


def simulate_single_variant(
    variant_idx: int,
    variant_name: str,
    variant_info: Dict,
    gen_func: Callable,
    base_spec_serialized: Dict[str, Any],  # Changed: accept serialized ParamConfig
    agents_per_variant: int,
    extra_agents: int,
    max_retries: int,
    max_sessions_per_rule: int,
    min_performance: float,
    batch_size: int = 1000
) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Simulate all agents for a single variant (for dynamic parallel execution).
    Designed to be called by multiprocessing workers; pool will dynamically distribute variants to workers as they become available.
    
    Parameters:
    -----------
    variant_idx : int
        Index of this variant (for agent_id calculation)
    variant_name : str
        Name of the variant
    variant_info : dict
        Variant configuration dictionary
    gen_func : Callable
        Generative model function
    base_spec_serialized : dict
        Serialized ParamConfig (from serialize_param_config())
    agents_per_variant : int
        Base number of agents per variant
    extra_agents : int
        Number of variants that get one extra agent
    max_retries : int
        Max retry attempts per agent
    max_sessions_per_rule : int
        Max sessions per rule for quality check
    min_performance : float
        Minimum performance for quality check
    
    Returns:
    --------
    list of tuples
        List of (trial_data, metadata) for all successfully simulated agents in this variant
    """
    # Deserialize ParamConfig
    base_spec = ParamConfig.from_serialized(base_spec_serialized)
    
    results = []
    
    # Calculate agents for this variant
    # First 'extra_agents' variants get one extra agent to distribute remainder
    n_agents_this_variant = agents_per_variant + (1 if variant_idx < extra_agents else 0)
    
    # Calculate starting agent_id for this variant as the sum of all agents from previous variants:
    # - First 'extra_agents' variants: each has (agents_per_variant + 1) agents
    # - Remaining variants: each has agents_per_variant agents
    if variant_idx < extra_agents:
        # This variant is in the first group that gets extra agents
        variant_start_id = variant_idx * (agents_per_variant + 1)
    else:
        # This variant is in the second group (no extra agents)
        variant_start_id = extra_agents * (agents_per_variant + 1) + (variant_idx - extra_agents) * agents_per_variant
    
    # Apply variant settings
    variant_defaults = variant_info.get('defaults', {})
    variant_spec = base_spec.replace_defaults(variant_defaults)
    
    # Simulate all agents for this variant
    # If any agent fails MAX_RETRIES, simulate_agent_with_retries will:
    # 1. Print detailed failure information (variant config, rejected params, bounds, thresholds)
    # 2. Raise RuntimeError which will crash this worker and stop the simulation
    for i in range(n_agents_this_variant):
        agent_id = variant_start_id + i
        trial_data, metadata = simulate_agent_with_retries(
            sim_func=gen_func,
            model_spec=variant_spec,
            agent_id=agent_id,
            seed=agent_id,
            max_retries=max_retries,
            max_sessions_per_rule=max_sessions_per_rule,
            min_performance=min_performance,
            verbose=False,  # No verbose output in parallel mode
            variant_settings=variant_info  # Pass variant info (for metadata)
        )
        
        results.append((trial_data, metadata))
    
    return results


def simulate_agent_with_retries(
    sim_func: Callable,
    model_spec: ParamConfig,
    agent_id: int,
    seed: int,
    max_retries: int = 10,
    max_sessions_per_rule: int = 9,
    min_performance: float = 0.3,
    verbose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate an agent with quality filtering and retries.
    
    If an agent fails quality checks, resample new parameters and retry
    until we get a good one (or hit max_retries).
    
    Parameters:
    -----------
    sim_func : Callable
        Simulation function
    model_spec : ParamConfig
        Model specification
    agent_id : int
        Agent ID
    seed : int
        Base random seed (incremented for each retry)
    max_retries : int, optional
        Maximum retry attempts (default: 10)
    max_sessions_per_rule : int, optional
        Max sessions per rule for timeout check (default: 9)
    min_performance : float, optional
        Minimum accuracy (default: 0.3)
    verbose : bool, optional
        Print rejection/retry messages (default: False)
    **kwargs
        Additional arguments passed to simulate_agent
        
    Returns:
    --------
    trial_data : pd.DataFrame
        Agent trial data (only if passed quality)
    metadata : dict
        Agent metadata (only if passed quality)
    
    Raises:
    -------
    RuntimeError
        If max_retries exceeded without finding good agent
        
    Example:
    --------
    >>> trial_data, metadata = simulate_agent_with_retries(
    ...     sim_func=generate_ruleQhyp,
    ...     model_spec=RULEQHYP_BASE_MODEL_SPEC,
    ...     agent_id=42,
    ...     seed=42,
    ...     max_retries=10,
    ...     max_sessions_per_rule=9,
    ...     min_performance=0.3,
    ...     verbose=True
    ... )
    """
    rejected_params = []  # Track what didn't work
    printed_details = False  # Track if we've printed rejection details
    
    # Get free params for this variant (for cleaner printing)
    free_params = model_spec.get_params_to_fit()
    
    # Print agent number (updates in place like loading bar)
    if verbose:
        print(f"\rAgent {agent_id}", end='', flush=True)
    
    for attempt in range(max_retries):
        # Use different seed for each retry (samples NEW parameters)
        retry_seed = seed + attempt * 1000000
        
        # Simulate agent (will sample new parameters with this seed)
        trial_data, metadata = simulate_agent(
            sim_func=sim_func,
            model_spec=model_spec,
            agent_id=agent_id,
            seed=retry_seed,
            **kwargs
        )
        
        # Check quality (pass verbose=False, we'll handle printing here)
        should_keep, reason = _check_agent_quality(
            trial_data=trial_data,
            metadata=metadata,
            max_sessions_per_rule=max_sessions_per_rule,
            min_performance=min_performance,
            verbose=False  # Don't print in _check_agent_quality
        )
        
        params = metadata.get('params_dict', {})
        
        if should_keep:
            # Good agent - return it
            if verbose and attempt > 0:
                # Had rejections before acceptance - print final acceptance (only free params)
                print(f"     {_format_params(params, free_params)}")
                print(f"         ACCEPTED (after {attempt + 1} attempts)\n")
            return trial_data, metadata
        else:
            # Bad parameters - track and resample
            rejected_params.append(params.copy())
            
            if verbose:
                # First rejection: print newline to preserve agent number
                if not printed_details:
                    print()  # Move to new line
                    printed_details = True
                
                print(f"     {_format_params(params, free_params)}")
                explanation = f"         REJECTED - {reason}"
                if attempt < max_retries - 1:
                    explanation += f" - Resampling params (attempt {attempt + 2}/{max_retries})..."
                print(explanation)
    
    # Get variant name from model_spec (encodes all variant info)
    variant_name = model_spec.get_model_id()
    
    # Max retries exceeded - show ALL rejected samples for this agent
    print(f"\n{'='*70}")
    print(f"AGENT {agent_id}: FAILED AFTER {max_retries} ATTEMPTS")
    print(f"   Variant: {variant_name}")
    print(f"{'='*70}")
    
    # Show all individual rejected parameter combinations (only free params)
    print(f"\nALL {len(rejected_params)} rejected parameter combinations:")
    for i, params in enumerate(rejected_params, 1):
        print(f"  {i}. {_format_params(params, free_params)}")
    
    # Show summary statistics (only for free params)
    print(f"\nParameter summary (min, max, avg) - FREE PARAMS ONLY:")
    for param_name in sorted(free_params):
        values = [p.get(param_name, None) for p in rejected_params if param_name in p]
        if values and all(isinstance(v, (int, float)) for v in values):
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            print(f"  {param_name:6s}: min={min_val:6.2f}, max={max_val:6.2f}, avg={avg_val:6.2f}")
    
    # Show parameter bounds
    print(f"\nCurrent parameter bounds (FREE PARAMS ONLY):")
    for param_name in sorted(free_params):
        param = model_spec.params[param_name]
        print(f"  {param_name:6s}: [{param.lb:6.2f}, {param.ub:6.2f}]")
    
    # Show quality thresholds
    print(f"\nQuality thresholds:")
    print(f"  - Minimum performance: {min_performance*100:.0f}%")
    print(f"  - Max sessions per rule: {max_sessions_per_rule}")
    print(f"  - Max retries: {max_retries}")

    raise RuntimeError(
        f"Agent {agent_id} failed quality checks after {max_retries} parameter samples. "
        f"Parameter bounds may be too wide or quality thresholds too strict. "
    )


def _format_params(params: Dict[str, Any], free_params: list = None) -> str:
    """Format parameter dictionary for printing.
    
    Parameters:
    -----------
    params : dict
        Full parameter dictionary
    free_params : list, optional
        List of parameter names to show. If None, show all.
    """
    if not params:
        return "No params"
    
    # Filter to only free params if specified
    if free_params is not None:
        params_to_show = {k: v for k, v in params.items() if k in free_params}
    else:
        params_to_show = params
    
    # Format each parameter with 2 decimal places
    param_strs = [f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                  for k, v in sorted(params_to_show.items())]
    return ", ".join(param_strs)


def _check_agent_quality(
    trial_data: pd.DataFrame,
    metadata: Dict[str, Any],
    max_sessions_per_rule: int = 9,
    min_performance: float = 0.3,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Quick quality check on simulated agent (internal helper).
    
    Returns whether to KEEP the agent and a reason if rejected.
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        Agent's trial data
    metadata : dict
        Agent metadata
    max_sessions_per_rule : int, optional
        Max sessions per rule (default: 9)
    min_performance : float, optional
        Minimum accuracy (default: 0.3)
    verbose : bool, optional
        Print rejection messages (default: False)
        
    Returns:
    --------
    should_keep : bool
        True if agent passes quality checks
    reason : str
        Reason for rejection (empty if kept)
    """
    
    # Check 1: Performance (accuracy)
    if 'reward' in trial_data.columns:
        accuracy = trial_data['reward'].mean()
        if accuracy < min_performance:
            reason = f"Low accuracy ({accuracy:.2f}<{min_performance})"
            if verbose:
                params = metadata.get('params_dict', {})
                print(f"  ⚠️  Agent {metadata.get('agent_id', '?')} ({_format_params(params)}): "
                      f"REJECTED - {reason}")
            return False, reason
    
    # Check 2: Timeout on all rules
    if 'new_session' in trial_data.columns:
        n_sessions = trial_data['new_session'].sum()
        max_possible = max_sessions_per_rule * 4  # 4 rules
        
        if n_sessions >= max_possible:
            reason = f"Timed out on all rules ({n_sessions} sessions)"
            if verbose:
                params = metadata.get('params_dict', {})
                print(f"  ⚠️  Agent {metadata.get('agent_id', '?')} ({_format_params(params)}): "
                      f"REJECTED - {reason}")
            return False, reason
    
    # Passed all checks
    return True, ""




_COPY_CHUNK = 16 * 1024 * 1024  # 16MB — safe for large files on network drives


def _list_batch_files(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Sorted ``(trial_data_batch_*.parquet, metadata_batch_*.pkl)`` under ``data_dir``."""
    data_dir = Path(data_dir)
    return (
        sorted(data_dir.glob("trial_data_batch_*.parquet")),
        sorted(data_dir.glob("metadata_batch_*.pkl")),
    )


def _parquet_num_rows(path: Path) -> int:
    """Row count from parquet footer (does not load row data)."""
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(path).metadata.num_rows)


def _assert_parquet_readable(path: Path, *, delete_on_fail: bool = True) -> None:
    """Confirm parquet footer is readable; optionally delete ``path`` on failure."""
    import pyarrow.parquet as pq
    path = Path(path)
    try:
        reader = pq.ParquetFile(path)
        _ = reader.metadata
        reader.close()
    except Exception:
        if delete_on_fail and path.exists():
            path.unlink()
        raise


def _replace_or_copy(src: Path, dst: Path, *, label: str = "file") -> None:
    """Atomic replace of ``src`` -> ``dst``, with streaming-copy fallback."""
    src = Path(src)
    dst = Path(dst)
    try:
        src.replace(dst)
        return
    except Exception as e:
        print(f"Warning: {label} move failed ({e}), using streaming copy...")
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=_COPY_CHUNK)
        if src.exists():
            src.unlink()


def _reconstruct_final_files_from_batches(
    data_dir: Path,
    sim_data_file: Path,
    sim_metadata_file: Path,
    checkpoint_file: Optional[Path] = None,
    desc: str = "Reconstructing"
) -> bool:
    """
    Reconstruct final files from batch files if they exist.
    
    Returns True if reconstruction succeeded, False otherwise.
    """
    trial_data_batch_files, metadata_batch_files = _list_batch_files(data_dir)
    
    if not (trial_data_batch_files and metadata_batch_files):
        return False
    
    try:
        print(f"{desc} final files from batch files using streaming merge...")
        
        # Load and merge all metadata (small, can load all at once)
        all_metadata = {}
        for metadata_file in metadata_batch_files:
            with open(metadata_file, 'rb') as f:
                batch_metadata = pickle.load(f)
                all_metadata.update(batch_metadata)
        
        # Ensure output directory exists
        sim_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge batch files using streaming (memory-efficient)
        row_count = _merge_batch_files_streaming(
            trial_data_batch_files,
            sim_data_file,
            desc=desc
        )
        
        if row_count == 0:
            raise IOError("No data found in batch files")
        
        # Ensure merge output file exists (merge may have created a temp file)
        temp_file = sim_data_file.with_suffix('.parquet.tmp')
        if not sim_data_file.exists() and temp_file.exists():
            print(f"Moving merge temp file to final location...")
            _replace_or_copy(temp_file, sim_data_file, label="Temp file")
        elif not sim_data_file.exists():
            raise FileNotFoundError(
                f"Merge completed but output file not found at {sim_data_file} "
                f"(also checked {temp_file})"
            )
        
        # Save and verify files
        try:
            _save_and_verify_final_files(sim_data_file, sim_metadata_file, all_metadata)
        except Exception as verify_error:
            print(f"\nVerification failed - batch files will NOT be deleted")
            print(f"   Error: {verify_error}")
            print(f"   You can try regenerating final files from batch files if needed")
            raise
        
        # Clean up batch files and checkpoint (only after successful verification)
        # If we reach here, verification passed, so it's safe to delete batch files
        if checkpoint_file is None:
            checkpoint_file = data_dir.parent / "sim_checkpoint.pkl"
        _cleanup_batch_files_and_checkpoint(
            data_dir,
            trial_data_batch_files=trial_data_batch_files,
            metadata_batch_files=metadata_batch_files,
            checkpoint_file=checkpoint_file if checkpoint_file.exists() else None,
            verbose=True
        )
        
        print("Data saved to disk. Load sim_data.parquet and sim_metadata.pkl separately.")
        return True
    except Exception as e:
        print(f"Failed to {desc.lower()} from batch files ({type(e).__name__}: {e})")
        return False


def _verify_final_files(
    sim_data_file: Path,
    sim_metadata_file: Path,
    expected_n_agents: Optional[int] = None
) -> None:
    """
    Verify that final files can be loaded (lightweight check - headers OK, sampled row groups)
    without loading entire files:
    - Checks parquet file headers/metadata
    - Samples first and last row groups to catch start/end corruption
    - Checks metadata file can be loaded (it's small, so loading is fine)

    Parameters:
    -----------
    sim_data_file : Path
        Path to final simulation data parquet file
    sim_metadata_file : Path
        Path to final metadata pickle file
    expected_n_agents : int, optional
        Expected number of agents in metadata (for validation)
        
    Raises:
    -------
    FileNotFoundError
        If files don't exist
    Exception
        If verification fails (files corrupted or unreadable)
    """
    from helpers import io
    
    # Verify files can be loaded successfully
    try:
        if not sim_data_file.exists():
            raise FileNotFoundError(f"Data file not found at {sim_data_file}")
        if not sim_metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {sim_metadata_file}")
        
        # Verify parquet file can be opened and metadata read (lightweight - just headers)
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(sim_data_file)
        row_count = parquet_file.metadata.num_rows
        
        # Verify parquet is readable (not just metadata): sample first/last row groups
        try:
            n_row_groups = parquet_file.num_row_groups
            if n_row_groups == 0:
                if row_count > 0:
                    raise ValueError("Parquet file has rows in metadata but no row groups")
                print(f"  Verified parquet file structure (empty file)")
            else:
                # First row group (start-of-file); last if present (truncated writes)
                first_df = parquet_file.read_row_group(0).to_pandas()
                if len(first_df) == 0 and row_count > 0:
                    raise ValueError("Parquet file has rows in metadata but first row group is empty")

                if n_row_groups > 1:
                    last_df = parquet_file.read_row_group(n_row_groups - 1).to_pandas()
                    if len(last_df) == 0:
                        raise ValueError(f"Last row group ({n_row_groups - 1}) is empty")
                    print(
                        f"  Verified parquet file is readable "
                        f"(first={len(first_df)}, last={len(last_df)}, "
                        f"total {row_count:,} rows in {n_row_groups} row groups)"
                    )
                else:
                    print(
                        f"  Verified parquet file is readable "
                        f"(1 row group: {len(first_df)} rows, total {row_count:,} rows)"
                    )
        except Exception as read_error:
            raise RuntimeError(
                f"Parquet file exists but cannot be read. File may be corrupted. "
                f"Error: {read_error}. Try regenerating from batch files."
            ) from read_error
        
        # Verify metadata file can be loaded (it's small, so loading is fine)
        verify_metadata = io.load_pickle(sim_metadata_file)
        n_agents = len(verify_metadata)
        if expected_n_agents is not None and n_agents != expected_n_agents:
            raise ValueError(
                f"Metadata mismatch: expected {expected_n_agents} agents but loaded {n_agents} agents"
            )
        print(f"  ✓ Verified metadata file is readable ({n_agents:,} agents)")
        
        print(f"  ✓ Final files verified: {row_count:,} trials from {n_agents:,} agents")
    except Exception as verify_error:
        print(f"\nVERIFICATION FAILED: {type(verify_error).__name__}: {verify_error}")
        print(f"   Final files may be corrupted. Batch files will NOT be deleted.")
        raise


def _save_and_verify_final_files(
    sim_data_file: Path,
    sim_metadata_file: Path,
    all_metadata: Dict[int, Dict[str, Any]]
) -> None:
    """
    Save metadata file and verify both data and metadata files can be loaded.
    
    Parameters:
    -----------
    sim_data_file : Path
        Path to final simulation data parquet file
    sim_metadata_file : Path
        Path to final metadata pickle file
    all_metadata : Dict[int, Dict[str, Any]]
        Dictionary of all agent metadata to save
        
    Raises:
    -------
    FileNotFoundError
        If data file doesn't exist
    Exception
        If metadata save or verification fails
    """
    from helpers import io
    
    # Save metadata file (do this before verification to ensure it's saved)
    try:
        io.save_pickle(all_metadata, sim_metadata_file, verbose=False)
        print(f"Saved metadata file: {len(all_metadata)} agents")
    except Exception as e:
        print(f"Error saving metadata file: {e}")
        raise
    
    # Use lightweight verification function
    _verify_final_files(sim_data_file, sim_metadata_file, expected_n_agents=len(all_metadata))



def _cleanup_batch_files_and_checkpoint(
    data_dir: Path,
    trial_data_batch_files: Optional[List[Path]] = None,
    metadata_batch_files: Optional[List[Path]] = None,
    checkpoint_file: Optional[Path] = None,
    verbose: bool = True
) -> None:
    """
    Clean up batch files, data directory, and checkpoint file.
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing batch files
    trial_data_batch_files : List[Path], optional
        List of trial data batch files to delete. If None, uses glob to find them.
    metadata_batch_files : List[Path], optional
        List of metadata batch files to delete. If None, uses glob to find them.
    checkpoint_file : Path, optional
        Checkpoint file to delete. If None, skips checkpoint cleanup.
    verbose : bool
        Whether to print cleanup messages
    """
    # Delete batch files
    if trial_data_batch_files is None:
        trial_data_batch_files = list(data_dir.glob("trial_data_batch_*.parquet"))
    if metadata_batch_files is None:
        metadata_batch_files = list(data_dir.glob("metadata_batch_*.pkl"))
    
    for batch_file in trial_data_batch_files:
        batch_file.unlink()
    for batch_file in metadata_batch_files:
        batch_file.unlink()
    
    # Remove data_chunks directory if empty
    try:
        data_dir.rmdir()
        if verbose:
            print("Cleaned up data_chunks directory")
    except OSError:
        pass  # Directory not empty or doesn't exist
    
    # Remove checkpoint file if provided
    if checkpoint_file is not None and checkpoint_file.exists():
        checkpoint_file.unlink()


def _merge_batch_files_streaming(
    batch_files: List[Path],
    output_file: Path,
    existing_agent_ids: Optional[set] = None,
    append_to_existing: bool = False,
    desc: str = "Merging batch files"
) -> int:
    """
    Merge multiple parquet batch files into a single file using streaming to avoid memory issues.
    
    Parameters:
    -----------
    batch_files : List[Path]
        List of parquet file paths to merge
    output_file : Path
        Path to output file (will be created/overwritten, or appended to if append_to_existing=True)
    existing_agent_ids : set, optional
        If provided, only include agents not in this set (for deduplication)
    append_to_existing : bool
        If True, append to output_file if it exists. If False, overwrite.
    desc : str
        Description for progress messages
        
    Returns:
    --------
    int
        Total number of rows written
    """
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import pyarrow as pa
    
    if not batch_files:
        return 0
    
    # Determine working file (use output_file directly if appending, otherwise use temp)
    if append_to_existing and output_file.exists() and output_file.stat().st_size > 0:
        working_file = output_file
    else:
        working_file = output_file.with_suffix('.parquet.tmp')
    
    # If we need deduplication or appending, use streaming approach with incremental writes
    if (existing_agent_ids is not None and len(existing_agent_ids) > 0) or append_to_existing:
        # Use pyarrow dataset for streaming even when deduplication is needed
        dataset = ds.dataset(batch_files, format='parquet')
        batch_size = 500000  # Large batches for faster processing (500k rows per batch - ~10x faster than 10k)
        scanner = dataset.scanner(batch_size=batch_size)
        writer = None
        
        row_count = 0
        new_agent_ids_found = set()
        
        try:
            for batch in scanner.to_batches():
                # Convert batch to pandas for filtering (only this small batch in memory)
                batch_table = pa.Table.from_batches([batch])
                batch_df = batch_table.to_pandas()
                del batch_table
                
                # Deduplicate if needed
                if existing_agent_ids is not None and len(existing_agent_ids) > 0:
                    batch_agent_ids = set(batch_df['agent_id'].unique()) if 'agent_id' in batch_df.columns else set()
                    truly_new_agent_ids = batch_agent_ids - existing_agent_ids
                    
                    if len(truly_new_agent_ids) == 0:
                        del batch_df
                        continue
                    
                    batch_df = batch_df[batch_df['agent_id'].isin(truly_new_agent_ids)]
                    new_agent_ids_found.update(truly_new_agent_ids)
                
                # Convert back to pyarrow table for writing
                chunk_table = pa.Table.from_pandas(batch_df)
                del batch_df
                
                # Initialize writer if needed
                if writer is None:
                    # If appending to existing file, read schema from existing file
                    if append_to_existing and working_file.exists() and working_file.stat().st_size > 0:
                        existing_schema = pq.read_schema(working_file)
                        # Write new data to a separate temp file first
                        writer = pq.ParquetWriter(working_file.with_suffix('.parquet.new'), existing_schema, compression='snappy')
                    else:
                        writer = pq.ParquetWriter(working_file, chunk_table.schema, compression='snappy')
                
                writer.write_table(chunk_table)
                row_count += len(chunk_table)
                del chunk_table
                
                if row_count % 500000 == 0:
                    print(f"    {desc}: {row_count} rows...")
            
            # Update existing_agent_ids if deduplication was used
            if existing_agent_ids is not None:
                existing_agent_ids.update(new_agent_ids_found)
        finally:
            # Close writer to ensure footer is written
            if writer is not None:
                writer.close()
                # Verify file is readable immediately after close
                test_file = working_file if not append_to_existing else working_file.with_suffix('.parquet.new')
                try:
                    if test_file.exists():
                        _assert_parquet_readable(test_file, delete_on_fail=True)
                except Exception as verify_err:
                    raise RuntimeError(
                        f"Parquet file corruption detected after write. "
                        f"File may be incomplete. Error: {verify_err}"
                    ) from verify_err
        
        # If we were appending to existing file, merge using streaming
        if append_to_existing and working_file.exists() and working_file.stat().st_size > 0:
            new_file = working_file.with_suffix('.parquet.new')
            if new_file.exists() and new_file.stat().st_size > 0:
                # Merge using streaming: read both files in batches
                existing_dataset = ds.dataset([working_file], format='parquet')
                new_dataset = ds.dataset([new_file], format='parquet')
                combined_dataset = ds.dataset([working_file, new_file], format='parquet')
                
                # Write merged dataset to final location using streaming
                combined_scanner = combined_dataset.scanner(batch_size=batch_size)
                merged_file = working_file.with_suffix('.parquet.merged')
                # Get schema from the combined dataset, not the scanner
                schema = combined_dataset.schema
                merged_writer = pq.ParquetWriter(merged_file, schema, compression='snappy')
                try:
                    for merged_batch in combined_scanner.to_batches():
                        merged_writer.write_batch(merged_batch)
                finally:
                    # Always close writer to ensure footer is written
                    merged_writer.close()
                    # Verify merged file is readable
                    try:
                        _assert_parquet_readable(merged_file, delete_on_fail=True)
                    except Exception as verify_err:
                        raise RuntimeError(
                            f"Merged parquet file corruption detected. Error: {verify_err}"
                        ) from verify_err
                
                # Replace old file with merged file (use streaming copy for large files)
                _replace_or_copy(merged_file, working_file, label="Merged file")
                new_file.unlink()  # Clean up temp file
        
        # Move temp file to final location if needed
        # For large files on network drives, avoid move operations that might copy entire file
        if working_file != output_file and working_file.exists():
            _replace_or_copy(working_file, output_file, label="Direct")
        
        # Verify final file exists
        if not output_file.exists() and working_file.exists():
            print("Warning: Final file missing after move, using streaming copy from temp...")
            _replace_or_copy(working_file, output_file, label="Final file")
        
        # Count rows efficiently using parquet metadata (doesn't load data)
        if output_file.exists():
            return _parquet_num_rows(output_file)
        return row_count
    else:
        # No deduplication needed - use pyarrow dataset streaming (more efficient)
        dataset = ds.dataset(batch_files, format='parquet')
        batch_size = 500000  # Large batches for faster merge (500k rows per batch - ~10x faster than 10k)
        scanner = dataset.scanner(batch_size=batch_size)
        writer = None
        
        row_count = 0
        try:
            for batch in scanner.to_batches():
                if writer is None:
                    writer = pq.ParquetWriter(working_file, batch.schema, compression='snappy')
                writer.write_batch(batch)
                row_count += len(batch)
                if row_count % 500000 == 0:
                    print(f"    {desc}: {row_count} rows...")
        finally:
            # CRITICAL: Always close writer to ensure footer is written
            if writer is not None:
                writer.close()
                # Verify file is readable immediately after close
                try:
                    if working_file.exists():
                        _assert_parquet_readable(working_file, delete_on_fail=True)
                except Exception as verify_err:
                    raise RuntimeError(
                        f"Parquet file corruption detected after write. "
                        f"File may be incomplete. Error: {verify_err}"
                    ) from verify_err
        
        if row_count > 0 and working_file.exists():
            if working_file != output_file:
                _replace_or_copy(working_file, output_file, label="Direct")
        
        # Verify final file exists
        if not output_file.exists():
            if working_file.exists():
                print("Warning: Final file missing after move, using streaming copy from temp...")
                try:
                    _replace_or_copy(working_file, output_file, label="Final file")
                except Exception as copy_error:
                    raise RuntimeError(
                        f"Failed to create output file {output_file} after merge. "
                        f"Temp file exists at {working_file} but copy failed: {copy_error}"
                    ) from copy_error
            else:
                raise RuntimeError(
                    f"Merge completed but output file not found at {output_file} "
                    f"and temp file not found at {working_file}. "
                    f"Merge may have failed silently."
                )
        
        return row_count


def merge_batches_memory_efficient(
    save_dir: Path,
    desc: str = "Merging batches"
) -> Tuple[int, int]:
    """
    Merge batch files (trial data and metadata) using memory-efficient streaming.
    
    Uses _merge_batch_files_streaming for trial data and processes metadata
    incrementally to avoid OOM errors.
    
    Parameters:
    -----------
    save_dir : Path
        Directory containing batch files and where final files will be saved
    desc : str, optional
        Description for progress messages
        
    Returns:
    --------
    Tuple[int, int]
        (n_trial_rows, n_agents) - number of trial rows and agents merged
    """
    # Merge trial data batches using streaming
    trial_data_batch_files, metadata_batch_files = _list_batch_files(save_dir)
    sim_data_file = save_dir / 'sim_data.parquet'
    
    if trial_data_batch_files:
        print(f"\n  {desc}: Merging {len(trial_data_batch_files)} trial data batches...")
        n_trial_rows = _merge_batch_files_streaming(
            trial_data_batch_files,
            sim_data_file,
            desc=f"{desc} (trial data)"
        )
        print(f"  ✓ Trial data merged: {n_trial_rows:,} rows")
    else:
        n_trial_rows = 0
        print(f"  ⚠ No trial data batch files found")
    
    # Merge metadata batches incrementally
    sim_metadata_file = save_dir / 'sim_metadata.pkl'
    
    if metadata_batch_files:
        print(f"  {desc}: Merging {len(metadata_batch_files)} metadata batches...")
        all_metadata = {}
        
        for i, meta_file in enumerate(metadata_batch_files):
            with open(meta_file, 'rb') as f:
                batch_metadata = pickle.load(f)
                all_metadata.update(batch_metadata)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(metadata_batch_files)} metadata batches...")
        
        with open(sim_metadata_file, 'wb') as f:
            pickle.dump(all_metadata, f)
        
        n_agents = len(all_metadata)
        print(f"  ✓ Metadata merged: {n_agents:,} agents")
        del all_metadata  # Free memory
    else:
        n_agents = 0
        print(f"  ⚠ No metadata batch files found")
    
    gc.collect()
    return n_trial_rows, n_agents


def simulate_all_agents_parallel(
    base_name: str,
    gen_func: Callable,
    base_spec: ParamConfig,
    variants: Dict[str, Dict[str, Any]],
    save_dir: Path,
    n_total_per_base: int = 50000,
    max_retries: int = 25,
    max_sessions_per_rule: int = 9,
    min_performance: float = 0.25,
    batch_size: int = 1000,
    n_workers: int = 20,
    restart: bool = False,
    cleanup_batches: bool = False,
) -> Tuple[int, int]:
    """
    High-level function to simulate all agents in parallel with checkpointing.
    
    This function handles the complete simulation workflow:
    - Parallel simulation across variants using multiprocessing
    - Incremental batch saving to avoid memory issues
    - Checkpointing and resuming from interruptions
    - Merging batches to final parquet/pickle files
    - Verification of final files (lightweight check)
    - Optional cleanup of batch files (default: False - batch files preserved for safety)
    
    Parameters:
    -----------
    base_name : str
        Name of the base model (e.g., 'ruleQhyp', 'Qlearn')
    gen_func : Callable
        Simulation function (e.g., generate_ruleQhyp, generate_Qlearn)
    base_spec : ParamConfig
        Base parameter configuration
    variants : Dict[str, Dict[str, Any]]
        Dictionary mapping variant names to variant configurations
    save_dir : Path
        Directory to save simulation data and checkpoints
    n_total_per_base : int, optional
        Total number of agents to simulate per base model (default: 50000)
    max_retries : int, optional
        Maximum retries per agent if quality checks fail (default: 25)
    max_sessions_per_rule : int, optional
        Maximum sessions per rule for quality filtering (default: 9)
    min_performance : float, optional
        Minimum performance threshold for quality filtering (default: 0.25)
    batch_size : int, optional
        Number of agents to accumulate before saving a batch (default: 1000)
    n_workers : int, optional
        Number of parallel workers (default: 20)
    restart : bool, optional
        If True, start fresh (ignore checkpoints). If False, resume from checkpoint (default: False)
    cleanup_batches : bool, optional
        If True, delete batch files after successful merge and verification (default: False)
        Batch files are preserved by default for safety - can be used to reconstruct final files
        
    Returns:
    --------
    Tuple[int, int]
        (n_agents, n_trial_rows) - number of agents and trial rows simulated
        
    Example:
    --------
    >>> from simulation.simulate import simulate_all_agents_parallel
    >>> from simulation.generative_models.ruleQhyp import generate
    >>> 
    >>> n_agents, n_rows = simulate_all_agents_parallel(
    ...     base_name='ruleQhyp',
    ...     gen_func=generate,
    ...     base_spec=RULEQHYP_BASE_MODEL_SPEC,
    ...     variants=ruleQhyp_variants,
    ...     save_dir=save_path / 'data' / 'ruleQhyp',
    ...     n_total_per_base=50000,
    ...     n_workers=20
    ... )
    """
    from multiprocessing import Pool, cpu_count
    import pickle
    
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = save_dir / 'sim_checkpoint.pkl'
    sim_data_file = save_dir / 'sim_data.parquet'
    sim_metadata_file = save_dir / 'sim_metadata.pkl'
    
    # Calculate distribution (needed for both early return check and actual simulation)
    variant_items = list(variants.items())
    n_variants = len(variant_items)
    agents_per_variant = n_total_per_base // n_variants
    extra_agents = n_total_per_base % n_variants
    
    # Print distribution info
    expected_total = extra_agents * (agents_per_variant + 1) + (n_variants - extra_agents) * agents_per_variant
    print(f"  Distribution: {n_variants} variants, {agents_per_variant} base agents/variant, "
          f"{extra_agents} variants get +1 agent (expected total: {expected_total:,})")
    
    # Check if final files already exist (simulation already completed)
    if not restart and sim_data_file.exists() and sim_metadata_file.exists():
        print(f"  Final files found: Checking if simulation is already complete...")
        try:
            # Verify files are readable (not corrupted)
            _verify_final_files(sim_data_file, sim_metadata_file, expected_n_agents=None)
            
            # Count agents and rows from existing files
            import pyarrow.parquet as pq
            n_trial_rows = _parquet_num_rows(sim_data_file)
            
            from helpers import io
            existing_metadata = io.load_pickle(sim_metadata_file)
            n_agents = len(existing_metadata)
            
            print(f"  Simulation already complete for {base_name}")
            print(f"    Found {n_agents:,} agents, {n_trial_rows:,} trial rows")
            print(f"    Files verified and readable")
            return n_agents, n_trial_rows
        except Exception as verify_error:
            print(f"  Final files exist but verification failed: {verify_error}")
            print(f"     Will re-run simulation (or resume from checkpoint if available)")
            # Continue to checkpoint/resume logic below
    
    # Check for existing checkpoint
    start_variant_idx = 0
    start_batch = 0
    total_agents_saved = 0
    
    if not restart and checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            start_variant_idx = checkpoint.get('last_variant_idx', 0) + 1
            start_batch = checkpoint.get('last_batch', 0) + 1
            total_agents_saved = checkpoint.get('total_agents_saved', 0)
        print(f"  Resuming from variant {start_variant_idx} (batch {start_batch}, {total_agents_saved:,} agents already saved)")
    else:
        if checkpoint_file.exists():
            print(f"  RESTART=True: Starting fresh")
        else:
            print(f"  No checkpoint found: Starting fresh")
        # Clear any existing batch files
        trial_batches, meta_batches = _list_batch_files(save_dir)
        for batch_file in trial_batches + meta_batches:
            batch_file.unlink()
    
    # Note: Distribution already calculated above (before early return check)
    # Calculate how often to save
    variants_per_batch = max(1, batch_size // agents_per_variant)
    print(f"  Saving every ~{variants_per_batch} variants (~{variants_per_batch * agents_per_variant} agents)")
    
    # Serialize base_spec for multiprocessing
    base_spec_serialized = base_spec.serialize()
    
    # Prepare arguments for each variant (starting from checkpoint)
    # Note: Pass base agents_per_variant (not calculated total) - function will calculate n_agents_this_variant internally
    variant_args = [
        (
            variant_idx,
            variant_name,
            variant_info,
            gen_func,
            base_spec_serialized,
            agents_per_variant,  # Base number - function will add +1 for first 'extra_agents' variants
            extra_agents,
            max_retries,
            max_sessions_per_rule,
            min_performance,
            batch_size
        )
        for variant_idx, (variant_name, variant_info) in enumerate(variant_items[start_variant_idx:], start=start_variant_idx)
    ]
    
    # Run parallel simulation with incremental saves
    print(f"\n  Starting parallel simulation...\n")
    
    all_results = []
    variants_completed = start_variant_idx
    batch_counter = start_batch
    
    with Pool(n_workers) as pool:
        # Use imap_unordered for incremental results
        for variant_results in pool.imap_unordered(_simulate_variant_wrapper, variant_args, chunksize=1):
            all_results.extend(variant_results)
            
            # Print completion for this variant
            print(f"\r  ✓ Variant completed: {len(variant_results):2d} agents (buffer: {len(all_results):4d}/{batch_size} agents, {variants_completed}/{n_variants} variants)", end='', flush=True)
            variants_completed += 1
            
            # Save when we have ~batch_size agents or at the end
            if len(all_results) >= batch_size or variants_completed == n_variants:
                # Separate trial data and metadata
                batch_trial_data = [r[0] for r in all_results]
                batch_metadata = {total_agents_saved + i: r[1] 
                                for i, r in enumerate(all_results)}
                
                # Save trial data batch
                df_batch = pd.concat(batch_trial_data, ignore_index=True)
                batch_data_file = save_dir / f'trial_data_batch_{batch_counter:04d}.parquet'
                df_batch.to_parquet(batch_data_file, engine='pyarrow', compression='snappy')
                
                # Save metadata batch
                batch_meta_file = save_dir / f'metadata_batch_{batch_counter:04d}.pkl'
                with open(batch_meta_file, 'wb') as f:
                    pickle.dump(batch_metadata, f)
                
                # Print batch save message
                print(f"\r  SAVED BATCH {batch_counter}: {len(all_results):,} agents | Total: {total_agents_saved + len(all_results):,}/{n_total_per_base:,}", flush=True)
                
                # Update checkpoint
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'last_variant_idx': variants_completed - 1,
                        'last_batch': batch_counter,
                        'total_agents_saved': total_agents_saved + len(all_results)
                    }, f)
                
                # Update counters and clear memory
                total_agents_saved += len(all_results)
                all_results = []
                batch_counter += 1
                gc.collect()
    
    # Clear final progress line
    print(f"\r{' '*120}\r")
    print(f"\n  Simulation complete: {total_agents_saved:,} agents in {batch_counter} batches")
    
    # Merge batches using memory-efficient streaming method
    n_trial_rows, n_agents = merge_batches_memory_efficient(save_dir, desc=f"Merging {base_name} batches")
    
    # Verify final files (lightweight check headers OK without loading large files into memory)
    try:
        print(f"\n  Verifying final files...")
        _verify_final_files(sim_data_file, sim_metadata_file, expected_n_agents=n_agents)
    except Exception as verify_error:
        print(f"\nVerification failed - batch files will NOT be deleted")
        print(f"   Error: {verify_error}")
        print(f"   Batch files are preserved - you can reconstruct final files using _reconstruct_final_files_from_batches()")
        raise
    
    # Clean up checkpoint file (no longer needed after successful completion)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"  Cleaned up checkpoint file")
    
    # Clean up batch files if requested (default: False - batch files preserved for safety)
    if cleanup_batches:
        print(f"\n  Cleaning up batch files...")
        trial_batches, meta_batches = _list_batch_files(save_dir)
        for batch_file in trial_batches + meta_batches:
            batch_file.unlink()
        print(f"  Cleaned up batch files")
    else:
        print(f"  Batch files preserved in {save_dir} (can be manually deleted if final files are verified)")
    
    return n_agents, n_trial_rows
