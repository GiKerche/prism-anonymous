"""Adapter for the pinned official BFCL AST checker; no provider SDKs required."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType,SimpleNamespace
from typing import Any
import os,sys,ast,json,re

CHECKER_PROFILE="m1-qwen35-9b-fc"
OFFICIAL_ROOT=Path(os.environ.get("PRISM_BFCL_ROOT","external/bfcl/berkeley-function-call-leaderboard"))

@lru_cache(maxsize=1)
def load_prompt_decoder():
    """Use the same four pinned upstream functions as the recorded evaluation.

    Generated Python expressions are parsed with ast.literal_eval; they are
    never executed. No API-provider modules are imported.
    """
    load_official_checker()
    from bfcl_eval.constants.enums import ReturnFormat
    path=OFFICIAL_ROOT/'bfcl_eval/model_handler/utils.py'
    wanted={'default_decode_ast_prompting','ast_parse','resolve_ast_call','resolve_ast_by_type'}
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:
        raise ValueError('The supplied BFCL snapshot lacks the pinned prompting decoder')
    ns=dict(json=json,ast=ast,re=re,ReturnFormat=ReturnFormat,eval=ast.literal_eval)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['default_decode_ast_prompting']

def load_official_checker(checker_profile: str = CHECKER_PROFILE) -> tuple[Any, Any]:
    """Load the official checker without importing unused API provider SDKs."""
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    stub_name = "bfcl_eval.constants.model_config"
    if stub_name not in sys.modules:
        stub = ModuleType(stub_name)
        stub.MODEL_CONFIG_MAPPING = {}
        sys.modules[stub_name] = stub
    sys.modules[stub_name].MODEL_CONFIG_MAPPING[checker_profile] = SimpleNamespace(
        underscore_to_dot=True
    )
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    return ast_checker, Language

def normalized_expected_names(expected: list[dict[str, Any]]) -> list[str]:
    return [next(iter(answer)).replace(".", "_") for answer in expected]

def evaluate_item(
    original_case: dict[str, Any], gold_case: dict[str, Any], calls: list[dict[str, Any]],
    checker_profile: str = CHECKER_PROFILE,
) -> dict[str, Any]:
    stratum = gold_case["stratum"]
    names = [str(call.get("name", "")) for call in calls]
    if stratum == "no_tool":
        return {
            "official_valid": len(calls) == 0,
            "call_decision_correct": len(calls) == 0,
            "function_names_correct": len(calls) == 0,
            "arguments_correct": None,
            "execution_success": None,
            "execution_status": "NOT_DEFINED_FOR_BFCL_V4_NONLIVE_IRRELEVANCE",
            "error": None if not calls else {"error_type": "irrelevance:unexpected_tool_call"},
        }
    expected = gold_case["expected"]
    expected_names = normalized_expected_names(expected)
    function_names_correct = Counter(names) == Counter(expected_names)
    model_output = [{str(call.get("name", "")): call.get("arguments", {})} for call in calls]
    descriptions = [tool["function"] for tool in original_case["tools"]]
    checker, language = load_official_checker(checker_profile)
    result = checker(
        descriptions, model_output, expected, language.PYTHON,
        original_case["category"], checker_profile,
    )
    official_valid = bool(result.get("valid"))
    return {
        "official_valid": official_valid,
        "call_decision_correct": len(calls) > 0,
        "function_names_correct": function_names_correct,
        "arguments_correct": official_valid,
        "execution_success": None,
        "execution_status": "NOT_DEFINED_FOR_BFCL_V4_NONLIVE_AST_CATEGORIES",
        "error": None if official_valid else {
            "error_type": result.get("error_type"),
            "error": result.get("error"),
        },
    }
