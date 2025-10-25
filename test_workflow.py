"""
Quick test script to verify the workflow
"""
import asyncio
import os
from dotenv import load_dotenv
from agile_workflow import run_agile_workflow

load_dotenv(override=True)

async def quick_test():
    """Run a quick test with a simple BRD"""
    brd_text = """
    Build a simple user authentication system with:
    1. User registration with email validation
    2. User login with password hashing
    """
    
    await run_agile_workflow(brd_text)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(quick_test())

