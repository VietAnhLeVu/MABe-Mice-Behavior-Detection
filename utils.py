"""
Utility functions for MABe Mouse Behavior Detection
"""
import re
import ast
from typing import List, Tuple


def parse_behaviors_safely(behaviors_str: str) -> List[Tuple[str, str, str]]:
    """
    Parse behaviors_labeled column safely
    
    Args:
        behaviors_str: String containing list of tuples
        
    Returns:
        List of tuples (agent, target, behavior)
    """
    if behaviors_str is None or behaviors_str == "":
        return []
    try:
        return ast.literal_eval(behaviors_str)
    except (ValueError, SyntaxError):
        print(f"Warning: Failed to parse behaviors: {behaviors_str}")
        return []


def extract_mouse_id(mouse_str: str) -> int:
    """
    Convert mouse string to numeric ID
    
    Args:
        mouse_str: "mouse1", "mouse2", or "self"
        
    Returns:
        int: Mouse ID (1, 2, ...) or -1 for "self"
    """
    if mouse_str == "self":
        return -1
    match = re.search(r"mouse(\d+)", mouse_str)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid mouse ID format: {mouse_str}")


def mouse_id_to_string(mouse_id: int) -> str:
    """
    Convert mouse ID to string format
    
    Args:
        mouse_id: -1 or positive number
        
    Returns:
        "self" if -1, "mouseN" otherwise
    """
    return "self" if mouse_id == -1 else f"mouse{mouse_id}"

