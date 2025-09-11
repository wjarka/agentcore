"""Utility functions for telemetry operations."""

from typing import Any


def merge_usage(
    current_usage: dict[str, Any], new_usage: dict[str, int | dict[str, int]]
) -> dict[str, Any]:
    """
    Merge new usage data into existing usage data.

    Args:
        current_usage: The existing usage dictionary to update
        new_usage: New usage data to merge in

    Returns:
        Updated usage dictionary with merged data

    Example:
        >>> current = {"tokens": 10, "cost": {"input": 0.01}}
        >>> new = {"tokens": 5, "cost": {"output": 0.02}}
        >>> merge_usage(current, new)
        {"tokens": 15, "cost": {"input": 0.01, "output": 0.02}}
    """
    updated_usage = current_usage.copy()

    for key, value in new_usage.items():
        if isinstance(value, dict):
            if key not in updated_usage:
                updated_usage[key] = {}
            for subkey, subvalue in value.items():
                if subkey in updated_usage[key]:
                    updated_usage[key][subkey] += subvalue
                else:
                    updated_usage[key][subkey] = subvalue
        else:
            if key in updated_usage:
                updated_usage[key] += value
            else:
                updated_usage[key] = value

    return updated_usage
