"""
Contract-Focused Script Analyzer

Analyzes Python scripts for contract alignment validation.
Focuses on main function signature and parameter usage patterns.

Based on analysis of actual scripts:
- currency_conversion.py
- xgboost_training.py
"""

import ast
from typing import Dict, List, Any, Optional


class ScriptAnalyzer:
    """
    Contract alignment focused script analyzer.

    Validates:
    - Main function signature compliance
    - Parameter usage patterns (input_paths, output_paths, environ_vars, job_args)
    - Contract alignment validation
    """

    def __init__(self, script_path: str):
        self.script_path = script_path
        self.script_content = self._read_script()
        self.ast_tree = self._parse_script()

    def _read_script(self) -> str:
        """Read script content from file."""
        with open(self.script_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_script(self) -> ast.AST:
        """Parse script content into AST."""
        return ast.parse(self.script_content)

    def validate_main_function_signature(self) -> Dict[str, Any]:
        """
        Validate main function has correct signature.

        Expected signature:
        def main(input_paths: Dict[str, str], output_paths: Dict[str, str],
                 environ_vars: Dict[str, str], job_args: argparse.Namespace) -> Any
        """
        main_function = self._find_main_function()
        if not main_function:
            return {
                "has_main": False,
                "issues": ["No main function found"],
                "signature_valid": False,
            }

        # Check parameter names and types
        expected_params = ["input_paths", "output_paths", "environ_vars", "job_args"]
        actual_params = self._extract_function_parameters(main_function)

        signature_valid = self._validate_signature(expected_params, actual_params)
        issues = self._get_signature_issues(expected_params, actual_params)

        return {
            "has_main": True,
            "signature_valid": signature_valid,
            "actual_params": actual_params,
            "expected_params": expected_params,
            "issues": issues,
        }

    def extract_parameter_usage(self) -> Dict[str, List[str]]:
        """
        Extract how script uses main function parameters.

        Returns:
            Dictionary with parameter usage patterns:
            - input_paths_keys: Keys used in input_paths["key"] or input_paths.get("key")
            - output_paths_keys: Keys used in output_paths["key"] or output_paths.get("key")
            - environ_vars_keys: Keys used in environ_vars.get("key")
            - job_args_attrs: Attributes used in job_args.attribute
        """
        main_function = self._find_main_function()
        if not main_function:
            return {
                "input_paths_keys": [],
                "output_paths_keys": [],
                "environ_vars_keys": [],
                "job_args_attrs": [],
            }

        return {
            "input_paths_keys": self._find_parameter_usage(
                main_function, "input_paths"
            ),
            "output_paths_keys": self._find_parameter_usage(
                main_function, "output_paths"
            ),
            "environ_vars_keys": self._find_parameter_usage(
                main_function, "environ_vars"
            ),
            "job_args_attrs": self._find_parameter_usage(main_function, "job_args"),
        }

    def validate_contract_alignment(self, contract: Dict) -> List[Dict]:
        """
        Validate script usage aligns with contract declarations.

        Args:
            contract: Contract dictionary with expected_input_paths, expected_output_paths, etc.

        Returns:
            List of validation issues
        """
        issues = []
        parameter_usage = self.extract_parameter_usage()

        # Validate input paths alignment
        script_input_keys = parameter_usage.get("input_paths_keys", [])
        contract_input_keys = list(contract.get("expected_input_paths", {}).keys())

        for key in script_input_keys:
            if key not in contract_input_keys:
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "undeclared_input_path",
                        "message": f"Script uses input_paths['{key}'] but contract doesn't declare it",
                        "recommendation": f"Add '{key}' to contract expected_input_paths",
                    }
                )

        # Validate output paths alignment
        script_output_keys = parameter_usage.get("output_paths_keys", [])
        contract_output_keys = list(contract.get("expected_output_paths", {}).keys())

        for key in script_output_keys:
            if key not in contract_output_keys:
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "undeclared_output_path",
                        "message": f"Script uses output_paths['{key}'] but contract doesn't declare it",
                        "recommendation": f"Add '{key}' to contract expected_output_paths",
                    }
                )

        # Validate environment variables alignment
        script_env_keys = parameter_usage.get("environ_vars_keys", [])
        contract_required_env = contract.get("required_env_vars", [])
        contract_optional_env = list(contract.get("optional_env_vars", {}).keys())
        contract_all_env = contract_required_env + contract_optional_env

        for key in script_env_keys:
            if key not in contract_all_env:
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "undeclared_env_var",
                        "message": f"Script uses environ_vars.get('{key}') but contract doesn't declare it",
                        "recommendation": f"Add '{key}' to contract required_env_vars or optional_env_vars",
                    }
                )

        # Validate job arguments alignment
        script_job_attrs = parameter_usage.get("job_args_attrs", [])
        contract_args = list(contract.get("expected_arguments", {}).keys())

        for attr in script_job_attrs:
            # Convert job_args.attr to --attr format for comparison
            arg_name = attr.replace("_", "-")
            if arg_name not in contract_args:
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "undeclared_job_arg",
                        "message": f"Script uses job_args.{attr} but contract doesn't declare --{arg_name}",
                        "recommendation": f"Add '--{arg_name}' to contract expected_arguments",
                    }
                )

        return issues

    # ------------------------------------------------------------------
    # Reverse-direction checks (contract/builder DECLARES -> script PARSES/READS?)
    #
    # The forward checks above catch "script uses X but the contract doesn't declare it".
    # These catch the opposite, empirically the more dangerous direction (FZ 29d14m Cat 4/5):
    #   Cat 4: the builder passes a CLI arg (e.g. --job_type) but the script's argparse never
    #          declares it -> the script crashes reading job_args.<x> at runtime.
    #   Cat 5: the contract declares a REQUIRED env var but the script never reads it
    #          ("simplified=don't need" fallacy).
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_flag(s: str) -> str:
        """Normalize a CLI flag / arg key to argparse's implicit dest form: strip leading dashes,
        then '-' -> '_'. So '--job-type', '--job_type', and 'job_type' all collapse to 'job_type'
        (argparse derives the attr name the same way), giving one comparison key for both sides."""
        return s.lstrip("-").replace("-", "_")

    def extract_argparse_flags(self) -> List[Dict[str, Any]]:
        """Extract every ``parser.add_argument(...)`` declared ANYWHERE in the module.

        Walks the WHOLE module (not just ``main``) because the parser lives in a module-level
        ``if __name__ == "__main__":`` block (often inside a ``try:``), and may sit in a helper.
        Returns one record per add_argument: ``{flag, canonical, dest, required, choices,
        has_default, dynamic}`` where ``canonical`` (the normalized flag) is the comparison key and
        ``dynamic`` marks a non-constant flag (built from a variable — absence can't be proven)."""
        flags: List[Dict[str, Any]] = []
        for node in ast.walk(self.ast_tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            # First positional string is the option/flag (or a bare positional name).
            flag: Optional[str] = None
            dynamic = False
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if flag is None or (arg.value.startswith("--")):
                        flag = arg.value
                else:
                    # first positional is a non-constant (variable/f-string) -> can't prove
                    dynamic = dynamic or not node.args[0:1] or arg is node.args[0]
            if (
                flag is None
                and node.args
                and not any(isinstance(a, ast.Constant) for a in node.args)
            ):
                dynamic = True
            required = False
            choices: Optional[List[Any]] = None
            dest: Optional[str] = None
            has_default = False
            for kw in node.keywords:
                if kw.arg == "required" and isinstance(kw.value, ast.Constant):
                    required = bool(kw.value.value)
                elif kw.arg == "choices" and isinstance(
                    kw.value, (ast.List, ast.Tuple)
                ):
                    choices = [
                        e.value for e in kw.value.elts if isinstance(e, ast.Constant)
                    ]
                elif kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = kw.value.value
                elif kw.arg == "default":
                    has_default = True
            canonical = self._normalize_flag(flag) if flag else None
            flags.append(
                {
                    "flag": flag,
                    "canonical": canonical,
                    "dest": dest,
                    "required": required,
                    "choices": choices,
                    "has_default": has_default,
                    "dynamic": dynamic,
                }
            )
        return flags

    def extract_env_reads(self) -> Dict[str, set]:
        """Find env-var keys the script READS, in two tiers (the Cat-5 two-tier rule):

        - ``all``: every key referenced anywhere via ``os.environ.get`` / ``os.getenv`` /
          ``os.environ[...]`` (Load) / ``environ_vars.get`` / ``environ_vars[...]`` (Load).
        - ``consuming``: keys read in a CONSUMING position — an ``environ_vars.*`` access inside
          ``main()``'s body, OR an ``os.environ.*``/``os.getenv`` read that is NOT merely a value
          harvested into a module-level ``environ_vars = {...}`` dict literal. A var that is only
          harvested into that dict but never consumed in main() is exactly the Cat-5 bug, so it must
          NOT count as ``consuming``.

        Env writes (``os.environ['K'] = ...`` — Store ctx) are excluded from both tiers.
        """
        all_keys: set = set()
        consuming: set = set()

        main_fn = self._find_main_function()
        main_nodes = set(id(n) for n in ast.walk(main_fn)) if main_fn else set()

        # Identify os.environ.* nodes that are values harvested into a module-level
        # `environ_vars = {...}` dict literal — these contribute to `all` only, not `consuming`.
        harvest_node_ids: set = set()
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "environ_vars" for t in node.targets
            ):
                if isinstance(node.value, ast.Dict):
                    for v in ast.walk(node.value):
                        harvest_node_ids.add(id(v))

        def _record(key: str, node: ast.AST, is_environ_vars: bool) -> None:
            all_keys.add(key)
            if is_environ_vars:
                # environ_vars.* is a consuming read only when it is inside main()'s body.
                if id(node) in main_nodes:
                    consuming.add(key)
            else:
                # os.environ.* / os.getenv: consuming unless it's a __main__ dict-harvest value.
                if id(node) not in harvest_node_ids:
                    consuming.add(key)

        for node in ast.walk(self.ast_tree):
            # os.environ.get("K") / os.getenv("K") / environ_vars.get("K")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                fn = node.func
                key = (
                    node.args[0].value
                    if node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    else None
                )
                if key is None:
                    continue
                # os.environ.get(...)
                if (
                    fn.attr == "get"
                    and isinstance(fn.value, ast.Attribute)
                    and fn.value.attr == "environ"
                    and isinstance(fn.value.value, ast.Name)
                    and fn.value.value.id == "os"
                ):
                    _record(key, node, is_environ_vars=False)
                # os.getenv(...)
                elif (
                    fn.attr == "getenv"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "os"
                ):
                    _record(key, node, is_environ_vars=False)
                # environ_vars.get(...)
                elif (
                    fn.attr == "get"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "environ_vars"
                ):
                    _record(key, node, is_environ_vars=True)
            # os.environ["K"] / environ_vars["K"] (Load only — excludes writes)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                key = (
                    node.slice.value
                    if isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    else None
                )
                if key is None:
                    continue
                tgt = node.value
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "environ"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "os"
                ):
                    _record(key, node, is_environ_vars=False)
                elif isinstance(tgt, ast.Name) and tgt.id == "environ_vars":
                    _record(key, node, is_environ_vars=True)

        return {"all": all_keys, "consuming": consuming}

    def _reverse_contract_view(self, contract: Dict) -> tuple:
        """Shape-tolerant accessor -> (declared_arg_flags: set[canonical], required_env: set[str]).

        Cat-4 source priority: job_arguments[].flag (the .step.yaml shape — the real builder --flags)
        -> expected_arguments keys -> arguments keys. Cat-5 source: env_vars.required ->
        environment_variables.required -> required_env_vars. Optional env is never pulled."""
        declared_args: set = set()
        ja = contract.get("job_arguments")
        if ja:
            declared_args = {
                self._normalize_flag(a["flag"])
                for a in ja
                if isinstance(a, dict) and a.get("flag")
            }
        elif contract.get("expected_arguments"):
            declared_args = {
                self._normalize_flag(k) for k in contract["expected_arguments"]
            }
        elif contract.get("arguments"):
            declared_args = {self._normalize_flag(k) for k in contract["arguments"]}

        required_env: set = set()
        for key in ("env_vars", "environment_variables"):
            ev = contract.get(key)
            if isinstance(ev, dict) and ev.get("required"):
                required_env = set(ev["required"])
                break
        else:
            if contract.get("required_env_vars"):
                required_env = set(contract["required_env_vars"])

        return declared_args, required_env

    def validate_reverse_alignment(
        self, contract: Dict, sagemaker_step_type: Optional[str] = None
    ) -> List[Dict]:
        """Reverse-direction alignment: does the script PARSE the args / READ the required env the
        contract+builder declare? Returns issues in the same schema the forward checker emits, so a
        caller can ``.extend()`` them into one ``issues[]``. Additive — the forward methods the B1
        validator consumes are untouched."""
        issues: List[Dict] = []
        declared_args, required_env = self._reverse_contract_view(contract)

        # --- Cat 4: declared CLI args must be parsed --- (Processing only) ---
        # Training/Transform/CreateModel Estimators receive hyperparameters via JSON, NOT argv —
        # they legitimately declare job_arguments but build args=Namespace() with no add_argument.
        # So the arg check is GATED to Processing; other types get an INFO note, never an ERROR.
        if declared_args:
            if sagemaker_step_type == "Processing":
                flags = self.extract_argparse_flags()
                parsed = {f["canonical"] for f in flags if f["canonical"]}
                parsed |= {
                    self._normalize_flag(f["dest"]) for f in flags if f.get("dest")
                }
                has_dynamic = any(f.get("dynamic") for f in flags)
                if not flags:
                    issues.append(
                        {
                            "severity": "ERROR",
                            "category": "unparsed_declared_arg",
                            "message": (
                                f"Script defines no argparse parser but the contract declares "
                                f"{len(declared_args)} job argument(s) {sorted(declared_args)} that "
                                f"the builder passes on argv — the script will crash at runtime."
                            ),
                            "recommendation": "Add a parser.add_argument('--<flag>', ...) for each in "
                            'the `if __name__ == "__main__":` block, or remove the flag from the '
                            ".step.yaml contract.job_arguments if the builder no longer passes it.",
                        }
                    )
                else:
                    for d in sorted(declared_args - parsed):
                        sev = "WARNING" if has_dynamic else "ERROR"
                        issues.append(
                            {
                                "severity": sev,
                                "category": "unparsed_declared_arg",
                                "message": (
                                    f"Contract declares CLI argument '--{d}' (builder passes it on "
                                    f"argv) but the script's argparse defines no matching "
                                    f"add_argument; script will crash reading job_args.{d}."
                                ),
                                "recommendation": f"Add parser.add_argument('--{d}', ...) in the "
                                "__main__ block, or remove it from contract.job_arguments.",
                            }
                        )
            else:
                issues.append(
                    {
                        "severity": "INFO",
                        "category": "reverse_arg_check_skipped_non_processing",
                        "message": (
                            f"argparse reverse check skipped: {sagemaker_step_type or 'unknown'} "
                            "steps receive arguments via hyperparameters/JSON, not argv."
                        ),
                        "recommendation": "",
                    }
                )

        # --- Cat 5: required env vars must be read (all script-bearing step types) ---
        if required_env:
            read = self.extract_env_reads()["consuming"]
            for k in sorted(required_env - read):
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "unread_required_env_var",
                        "message": (
                            f"Contract declares required env var '{k}' but the script never reads "
                            f"it (no environ_vars.get('{k}') in main() / os.environ read) — the "
                            f"value is silently ignored."
                        ),
                        "recommendation": f"Read '{k}' via environ_vars.get('{k}') inside main(), or "
                        "move it to contract.env_vars.optional if the script genuinely does not use it.",
                    }
                )

        return issues

    def _find_main_function(self) -> Optional[ast.FunctionDef]:
        """Find main function in AST."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return node
        return None

    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from function definition."""
        return [arg.arg for arg in func_node.args.args]

    def _validate_signature(self, expected: List[str], actual: List[str]) -> bool:
        """Validate the signature has the 4 REQUIRED params (in order) as a prefix.

        Trailing OPTIONAL params are allowed — the de-facto testability convention adds an optional
        ``logger=None`` (e.g. ``main(input_paths, output_paths, environ_vars, job_args, logger=None)``,
        used by 13 shipped scripts), so an exact-length match would wrongly reject them. We require
        the first 4 to be exactly the expected names; extras beyond them are accepted."""
        return len(actual) >= len(expected) and actual[: len(expected)] == expected

    def _get_signature_issues(
        self, expected: List[str], actual: List[str]
    ) -> List[str]:
        """Get list of signature validation issues (the 4 required params must be the prefix;
        trailing optional params like ``logger=None`` are allowed)."""
        issues = []
        if len(actual) < len(expected):
            issues.append(
                f"Expected at least {len(expected)} parameters, got {len(actual)}"
            )

        for i, (exp, act) in enumerate(zip(expected, actual)):
            if exp != act:
                issues.append(f"Parameter {i + 1}: expected '{exp}', got '{act}'")

        return issues

    def _find_parameter_usage(
        self, func_node: ast.FunctionDef, param_name: str
    ) -> List[str]:
        """Find usage patterns for a specific parameter."""
        usage_keys = []

        # First, collect all string literals that might be used as keys
        potential_keys = self._collect_string_literals(func_node)

        for node in ast.walk(func_node):
            # Look for param_name["key"] or param_name.get("key") patterns
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == param_name:
                    # Handle direct string literals (modernized for Python 3.8+)
                    if isinstance(node.slice, ast.Constant) and isinstance(
                        node.slice.value, str
                    ):
                        key = node.slice.value
                        if key not in usage_keys:
                            usage_keys.append(key)
                    # Handle variable subscripts - check if we can find the variable's value
                    elif isinstance(node.slice, ast.Name):
                        # Look for patterns like: for key in ["train", "validation"]: ... param_name[key]
                        var_name = node.slice.id
                        keys_from_loops = self._find_keys_from_loops(
                            func_node, var_name, potential_keys
                        )
                        for key in keys_from_loops:
                            if key not in usage_keys:
                                usage_keys.append(key)

            elif isinstance(node, ast.Call):
                # Look for param_name.get("key") patterns
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == param_name
                    and node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    key = node.args[0].value
                    if key not in usage_keys:
                        usage_keys.append(key)

            elif isinstance(node, ast.Attribute):
                # Look for job_args.attribute patterns
                if (
                    param_name == "job_args"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == param_name
                    and node.attr not in usage_keys
                ):
                    usage_keys.append(node.attr)

        return usage_keys

    def _collect_string_literals(self, func_node: ast.FunctionDef) -> List[str]:
        """Collect all string literals in the function that could be used as keys."""
        string_literals = []

        for node in ast.walk(func_node):
            # Only use ast.Constant for Python 3.8+ compatibility
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.append(node.value)

        return string_literals

    def _find_keys_from_loops(
        self, func_node: ast.FunctionDef, var_name: str, potential_keys: List[str]
    ) -> List[str]:
        """Find keys that might be assigned to a variable in loops or assignments."""
        keys = []

        for node in ast.walk(func_node):
            # Look for: for var_name in ["key1", "key2", ...]:
            if isinstance(node, ast.For):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == var_name
                    and isinstance(node.iter, (ast.List, ast.Tuple))
                ):
                    for elt in node.iter.elts:
                        # Only use ast.Constant for Python 3.8+ compatibility
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            keys.append(elt.value)

            # Look for: var_name = "key" or similar assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        # Only use ast.Constant for Python 3.8+ compatibility
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            keys.append(node.value.value)

        return keys
