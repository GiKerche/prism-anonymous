# Reference: vLLM v0.18.2 documentation, Qwen3CoderToolParser._convert_param_value
# https://docs.vllm.ai/en/v0.18.2/api/vllm/tool_parsers/qwen3coder_tool_parser/
# SPDX-License-Identifier: Apache-2.0; Copyright contributors to the vLLM project
# Pinned schema conversion used by the versioned BFCL decoder.
from __future__ import annotations
import ast
import json
import logging
from typing import Any
logger = logging.getLogger(__name__)

def _convert_param_value(
    self, param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """Convert parameter value based on its type in the schema."""
    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        if param_config != {}:
            logger.debug(
                "Parsed parameter '%s' is not defined in the tool "
                "parameters for tool '%s', directly returning the "
                "string value.",
                param_name,
                func_name,
            )
        return param_value

    if (
        isinstance(param_config[param_name], dict)
        and "type" in param_config[param_name]
    ):
        param_type = str(param_config[param_name]["type"]).strip().lower()
    elif (
        isinstance(param_config[param_name], dict)
        and "anyOf" in param_config[param_name]
    ):
        # anyOf has no top-level "type"; treat as object to trigger json.loads.
        param_type = "object"
    else:
        param_type = "string"
    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not an "
                "integer in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not a float "
                "in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        if param_value not in ["true", "false"]:
            logger.debug(
                "Parsed value '%s' of parameter '%s' is not a boolean "
                "(`true` or `false`) in tool '%s', degenerating to "
                "false.",
                param_value,
                param_name,
                func_name,
            )
        return param_value == "true"
    else:
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                param_value = json.loads(param_value)
                return param_value
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "parsed with json.loads in tool '%s', will try "
                    "other methods to parse it.",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            param_value = ast.literal_eval(param_value)  # safer
        except (ValueError, SyntaxError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' cannot be "
                "converted via Python `ast.literal_eval()` in tool "
                "'%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
        return param_value

