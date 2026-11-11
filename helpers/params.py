# helpers/params.py
from dataclasses import dataclass, replace
from typing import Optional, Union, Dict, List, Set, Tuple, Any
import numpy as np


@dataclass
class Param:
    """Single parameter definition."""
    name: str
    lb: float
    ub: float
    x0: Optional[float] = None  # Optional starting value for optimization (not needed for simulation or DE)
    default: Optional[Union[float, str]] = None
    
    def is_free(self) -> bool:
        return self.default is None
    
    def is_reference(self) -> bool:
        return isinstance(self.default, str)


class ParamConfig:
    """Manages parameter definitions and provides easy access."""
    
    def __init__(self, params: List[Param], base_name: str = ""):
        self.params = {p.name: p for p in params}
        self.param_names = [p.name for p in params]
        self._param_list = params  # Keep original list for iteration
        self.base_name = base_name  # Base model name (e.g., "ruleQhyp")
        self.model_name = self.get_model_id()  # Optional model identifier (auto-computed if None)
        self._validate()
    
    def __iter__(self):
        """Allow iteration over parameters in order."""
        return iter(self._param_list)
    
    def _validate(self):
        """Check that all references (for copying defaults from another param) point to valid parameters."""
        for param in self.params.values():
            if param.is_reference():
                if param.default not in self.params:
                    raise ValueError(f"Parameter '{param.name}' references unknown parameter '{param.default}'")
    
    def fill_parameter_values(self, fit_params: np.ndarray) -> Dict[str, float]:
        """
        Fill in parameter values: use fitted params and fill rest with defaults.
        
        Parameters:
        -----------
        fit_params : np.ndarray
            Array of fitted parameter values (in order of get_params_to_fit())
        
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping all parameter names to their values
        """
        params_to_fit = self.get_params_to_fit()
        if len(fit_params) != len(params_to_fit):
            raise ValueError(f"Expected {len(params_to_fit)} fitted params, got {len(fit_params)}")
        
        # Initialize with fitted parameters
        params_dict = dict(zip(params_to_fit, fit_params))
        
        # Fill in defaults and resolve references
        for name in self.param_names:
            if name not in params_dict:
                # Params with default values that are not being fitted
                param = self.params[name]
                if param.is_reference():
                    params_dict[name] = self._copy_reference(param.default, params_dict)
                else:
                    params_dict[name] = param.default
        
        return params_dict
    
    def get_params_to_fit(self) -> List[str]:
        """Get list of parameter names that should be fitted."""
        return [name for name, p in self.params.items() if p.is_free()]
    
    def get_model_id(self) -> str:
        """
        Generate a unique model ID string for this parameter configuration.
        
        This creates a canonical representation that handles equivalent
        specifications (e.g., Ap=1.0, An=Ap vs An=1.0, Ap=An produce the same ID).
        
        Returns:
        --------
        str
            Unique model ID string with format:
            base_name__FIT_param1_param2__LINK_source_deps__FIX_value_params
            
        Raises:
        ------
        ValueError
            If any parameter names don't follow the capital first letter,
            lowercase remainder format, ensuring the concatenated LINK format is unambiguous.
        """
        # Validate parameter name format (Xxxxx: capital first, lowercase remainder)
        invalid_params = []
        for param_name in self.param_names:
            if not param_name or not param_name[0].isupper():
                invalid_params.append(param_name)
            elif len(param_name) > 1 and not param_name[1:].islower():
                invalid_params.append(param_name)
        
        if invalid_params:
            raise ValueError(
                f"Parameter names must follow capital first letter lowercase remainder format, for unambiguous LINK concatenation. "
                f"Invalid params: {invalid_params}"
            )
        
        name_parts = [self.base_name]
        
        # Find equivalence classes (parameters linked by references)
        equiv_classes = self._find_equivalence_classes()
        
        # Build canonical representation
        canonical_fit = set()
        canonical_links = []
        canonical_fixed = {}  # value -> set of param names
        
        # Process each equivalence class
        for equiv_class in equiv_classes:
            sorted_class = sorted(equiv_class)
            
            # Check if any member has a fixed value
            fixed_value = None
            for param_name in sorted_class:
                param = self.params[param_name]
                if not param.is_free() and not param.is_reference():
                    # This param has a fixed value
                    fixed_value = param.default
                    break
                elif param.is_reference():
                    # Follow reference chain to find fixed value
                    resolved = self._resolve_to_fixed_value(param_name)
                    if resolved is not None:
                        fixed_value = resolved
                        break
            
            if fixed_value is not None:
                # All members of this class are effectively fixed to this value
                val_str = self._normalize_value_str(fixed_value)
                if val_str not in canonical_fixed:
                    canonical_fixed[val_str] = set()
                canonical_fixed[val_str].update(sorted_class)
            else:
                # No fixed value in this class - pick lexicographically first as canonical source for fitting
                canonical_source = min(sorted_class)  # Already sorted, but use min for clarity
                canonical_fit.add(canonical_source)
                dependents = sorted([p for p in sorted_class if p != canonical_source])
                if dependents:
                    canonical_links.append((canonical_source, dependents))
        
        # Add standalone fitted parameters (not in any equivalence class)
        for name, param in self.params.items():
            if param.is_free() and name not in canonical_fit:
                # Check if it's in any equivalence class
                in_equiv_class = any(name in equiv_class for equiv_class in equiv_classes)
                if not in_equiv_class:
                    canonical_fit.add(name)
        
        # Add standalone fixed parameters (not in any equivalence class, not referenced)
        for name, param in self.params.items():
            if not param.is_free() and not param.is_reference():
                # Check if it's in any equivalence class
                in_equiv_class = any(name in equiv_class for equiv_class in equiv_classes)
                if not in_equiv_class:
                    val_str = self._normalize_value_str(param.default)
                    if val_str not in canonical_fixed:
                        canonical_fixed[val_str] = set()
                    canonical_fixed[val_str].add(name)
        
        # Build output string components
        if canonical_fit:
            fit_str = '_'.join(sorted(canonical_fit))
            name_parts.append(f"FIT_{fit_str}")
        
        if canonical_links:
            link_parts = []
            for source, deps in sorted(canonical_links):
                # Concatenate linked params within a group (assumes params start with capital letter)
                # Underscores separate different link groups
                all_params = sorted([source] + deps)
                link_parts.append(''.join(all_params))
            link_str = '_'.join(sorted(link_parts))
            name_parts.append(f"LINK_{link_str}")
        
        if canonical_fixed:
            fixed_parts = []
            for val_str in sorted(canonical_fixed.keys()):
                params_with_val = sorted(canonical_fixed[val_str])
                if len(params_with_val) == 1:
                    fixed_parts.append(f"{val_str}_{params_with_val[0]}")
                else:
                    fixed_parts.append(f"{val_str}_{'_'.join(params_with_val)}")
            if fixed_parts:
                fixed_str = '_'.join(fixed_parts)
                name_parts.append(f"FIX_{fixed_str}")
        
        if len(name_parts) == 1:
            name_parts.append("default")
        
        return "__".join(name_parts)
    
    def _normalize_value_str(self, value: Union[int, float]) -> str:
        """Normalize value to string representation."""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        elif isinstance(value, (float, np.floating)):
            if float(value).is_integer():
                return str(int(value))
            else:
                return str(float(value))
        else:
            return str(value)
    
    def _resolve_to_fixed_value(self, param_name: str) -> Optional[Union[int, float]]:
        """Resolve a parameter reference chain to find a fixed value, if any."""
        visited = set()
        current = param_name
        
        while current not in visited:
            visited.add(current)
            param = self.params[current]
            
            if not param.is_free() and not param.is_reference():
                # Found a fixed value
                return param.default
            elif param.is_reference():
                # Follow reference
                current = param.default
            else:
                # Free parameter - no fixed value in this chain
                return None
        
        # Circular reference without fixed value
        return None
    
    def _find_equivalence_classes(self) -> List[Set[str]]:
        """
        Find equivalence classes of linked parameters.
        
        Parameters that reference each other (directly or transitively) form
        an equivalence class. This handles cases like:
        - Ap=1.0, An=Ap -> {Ap, An} (An references Ap)
        - Ap=An, An=1.0 -> {Ap, An} (Ap references An)
        - Ap=An, An=Ap -> {Ap, An} (mutual reference)
        
        Returns:
        --------
        List[Set[str]]
            List of equivalence classes (sets of parameter names)
        """
        # Build reference graph
        ref_graph = {}
        for name, param in self.params.items():
            if param.is_reference():
                ref_graph[name] = param.default
        
        # Find all equivalence classes using union-find
        # Always make lexicographically smallest the root
        parent = {name: name for name in self.param_names}
        
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: str, y: str):
            px, py = find(x), find(y)
            if px != py:
                # Always make lexicographically smallest the root
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py
        
        # Union all parameters that are linked
        for name, ref_name in ref_graph.items():
            union(name, ref_name)
        
        # Group by root
        groups: Dict[str, Set[str]] = {}
        for name in self.param_names:
            root = find(name)
            if root not in groups:
                groups[root] = set()
            groups[root].add(name)
        
        # Only return groups with more than one member (actual links)
        return [group for group in groups.values() if len(group) > 1]
    
    def replace_defaults(
        self, 
        defaults: Dict[str, Optional[Union[float, str]]]
    ) -> 'ParamConfig':
        """
        Create a new ParamConfig with replaced default values.
        
        Parameters:
        -----------
        defaults : Dict[str, Optional[Union[float, str]]]
            Dictionary mapping parameter names to new default values.
            Use None to make a parameter free (override its base default).
            
        Returns:
        --------
        ParamConfig
            New ParamConfig with updated defaults.
            Inherits base_name from self, and auto-computes model_name.
        """
        new_params = []
        for param in self:
            if param.name in defaults:
                new_default = defaults[param.name]
                # Use dataclasses.replace for clean copying
                new_param = replace(param, default=new_default)
            else:
                # Create a copy to avoid modifying original
                new_param = replace(param)
            new_params.append(new_param)
        
        # Create new config inheriting base_name from self
        # model_name will be auto-computed in __init__
        return ParamConfig(
            new_params, 
            base_name=self.base_name
        )
    
    def _copy_reference(self, ref_name: str, params_dict: Dict[str, float]) -> float:
        """Resolve a parameter reference (handles chains)."""
        if ref_name not in self.params:
            raise ValueError(f"Unknown parameter reference: {ref_name}")
        
        # If already in dict, return it
        if ref_name in params_dict:
            return params_dict[ref_name]
        
        # Otherwise resolve the reference
        ref_param = self.params[ref_name]
        if ref_param.is_reference():
            return self._copy_reference(ref_param.default, params_dict)
        else:
            return ref_param.default
    
    def validate_required(self, required_params: List[str]) -> None:
        """
        Validate that all required parameters are present in the config.
        
        Parameters:
        -----------
        required_params : List[str]
            List of required parameter names
            
        Raises:
        -------
        ValueError
            If any required parameters are missing
        """
        missing = [p for p in required_params if p not in self.params]
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Available parameters: {list(self.params.keys())}"
            )
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize ParamConfig to a dictionary for multiprocessing/pickling.
        
        Returns:
        --------
        dict
            Dictionary containing all information needed to reconstruct ParamConfig
        """
        return {
            'params': [
                {
                    'name': p.name,
                    'lb': p.lb,
                    'ub': p.ub,
                    'x0': p.x0,
                    'default': p.default
                }
                for p in self._param_list
            ],
            'base_name': self.base_name,
            'model_name': self.model_name
        }
    
    @classmethod
    def from_serialized(cls, serialized: Dict[str, Any]) -> 'ParamConfig':
        """
        Reconstruct ParamConfig from serialized dictionary.
        
        Parameters:
        -----------
        serialized : dict
            Dictionary from serialize() method
        
        Returns:
        --------
        ParamConfig
            Reconstructed ParamConfig object
        """
        params = [Param(**p) for p in serialized['params']]
        spec = cls(params, base_name=serialized.get('base_name', ''))
        if 'model_name' in serialized:
            spec.model_name = serialized['model_name']
        return spec

