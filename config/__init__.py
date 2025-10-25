"""
Configuration utilities for loading agent instructions
"""
from pathlib import Path

def load_instructions(filename: str) -> str:
    """
    Load agent instructions from config file.
    
    Args:
        filename: Name of the instruction file in config folder
        
    Returns:
        The instruction text as a string
    """
    config_dir = Path(__file__).parent
    instruction_file = config_dir / filename
    
    if not instruction_file.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    with open(instruction_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

