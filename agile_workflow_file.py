import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from agents import Runner
from agileagents.decomposer_agent import decomposer_agent, TaskItem
from agileagents.developer_agent import developer_agent
from agileagents.tester_agent import tester_agent

# Load environment variables
load_dotenv(override=True)

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key before running this script.")
    exit(1)

# Create output directory
output_dir = Path("generated_tasks")
output_dir.mkdir(exist_ok=True)

async def process_single_task(task_item: TaskItem, task_number: int):
    """Process a single task: develop code and test it"""
    print(f"\n{'='*70}")
    print(f"🔨 Task {task_number}: {task_item.task}")
    print(f"{'='*70}")
    
    # Step 1: Generate code with developer agent
    print(f"\n📝 Developer Agent working on task {task_number}...")
    dev_context = (
        f"Epic: {task_item.epic}\n"
        f"User Story: {task_item.user_story}\n"
        f"Task: {task_item.task}\n"
        f"Required dependencies: Auth Library, Database Connector"
    )
    
    dev_result = await Runner.run(developer_agent, dev_context)
    dev_response = dev_result.final_output
    print(f"✅ Code generated for task {task_number}")
    
    # Step 2: Test the code with tester agent
    print(f"\n🧪 Tester Agent working on task {task_number}...")
    test_context = (
        f"Generate comprehensive unit tests for the following code:\n\n"
        f"Task: {task_item.task}\n\n"
        f"Code:\n{dev_response.code_snippet}"
    )
    
    test_result = await Runner.run(tester_agent, test_context)
    test_response = test_result.final_output
    print(f"✅ Tests generated for task {task_number}")
    
    # Step 3: Save to individual files
    # Save the main code
    code_filename = f"task_{task_number:02d}_{sanitize_filename(task_item.task)}.py"
    code_filepath = output_dir / code_filename
    
    with open(code_filepath, "w", encoding="utf-8") as f:
        f.write(f'"""\n')
        f.write(f"Task {task_number}\n")
        f.write(f"Epic: {task_item.epic}\n")
        f.write(f"User Story: {task_item.user_story}\n")
        f.write(f"Task: {task_item.task}\n")
        f.write(f'"""\n\n')
        f.write(dev_response.code_snippet)
        f.write(f'\n\n# Explanation:\n# {dev_response.explanation}\n')
    
    # Save the test file
    test_filename = f"test_task_{task_number:02d}_{sanitize_filename(task_item.task)}.py"
    test_filepath = output_dir / test_filename
    
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write(f'"""\n')
        f.write(f"Unit Tests for Task {task_number}\n")
        f.write(f"Task: {task_item.task}\n")
        f.write(f'"""\n\n')
        f.write("import unittest\n\n")
        for i, test_case in enumerate(test_response.test_cases, 1):
            f.write(f"# Test Case {i}:\n")
            f.write(f"{test_case}\n\n")
        f.write(f"\n# Coverage Summary:\n# {test_response.coverage_summary}\n")
    
    print(f"💾 Saved: {code_filename}")
    print(f"💾 Saved: {test_filename}")
    
    return {
        "task_item": task_item,
        "code": dev_response,
        "tests": test_response,
        "code_file": code_filename,
        "test_file": test_filename
    }

def sanitize_filename(text: str) -> str:
    """Convert text to a valid filename"""
    # Remove special characters and replace spaces with underscores
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    filename = ''.join(c if c in valid_chars else '_' for c in text)
    # Replace multiple underscores with single underscore
    filename = '_'.join(filter(None, filename.split('_')))
    # Limit length
    return filename[:50]

async def run_agile_workflow(brd_text: str):
    """Complete agile workflow: decompose, develop, and test"""
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Agile Workflow")
    print(f"{'='*70}")
    
    # Step 1: Decompose BRD into tasks
    print(f"\n📋 Decomposer Agent analyzing BRD...")
    decompose_result = await Runner.run(decomposer_agent, f"BRD Text:\n{brd_text}")
    decomposition = decompose_result.final_output
    print(f"✅ Decomposed into {len(decomposition.items)} tasks\n")
    
    # Display tasks
    for i, task in enumerate(decomposition.items, 1):
        print(f"{i}. {task.task} (Epic: {task.epic})")
    
    # Step 2: Process each task (develop + test) concurrently
    print(f"\n{'='*70}")
    print(f"🔄 Processing all tasks concurrently...")
    print(f"{'='*70}")
    
    results = await asyncio.gather(
        *(process_single_task(task, i) 
          for i, task in enumerate(decomposition.items, 1))
    )
    
    # Step 3: Generate summary report
    print(f"\n{'='*70}")
    print(f"📊 Generating Summary Report")
    print(f"{'='*70}")
    
    summary_file = output_dir / "workflow_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Agile Workflow Summary\n\n")
        f.write(f"## BRD Input\n\n```\n{brd_text}\n```\n\n")
        f.write(f"## Generated Tasks\n\n")
        
        for i, result in enumerate(results, 1):
            task = result["task_item"]
            f.write(f"### Task {i}: {task.task}\n\n")
            f.write(f"**Epic:** {task.epic}\n\n")
            f.write(f"**User Story:** {task.user_story}\n\n")
            f.write(f"**Files Generated:**\n")
            f.write(f"- Code: `{result['code_file']}`\n")
            f.write(f"- Tests: `{result['test_file']}`\n\n")
            f.write(f"**Code Explanation:** {result['code'].explanation}\n\n")
            f.write(f"**Test Coverage:** {result['tests'].coverage_summary}\n\n")
            f.write(f"---\n\n")
    
    print(f"💾 Saved: workflow_summary.md")
    print(f"\n{'='*70}")
    print(f"✅ Workflow Complete!")
    print(f"{'='*70}")
    print(f"\n📁 All files saved in: {output_dir.absolute()}")
    print(f"📄 Total tasks processed: {len(results)}")
    print(f"📄 Total files created: {len(results) * 2 + 1}")

def read_brd_file(file_path: str) -> str:
    """Read BRD from a text file"""
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"❌ Error: File not found: {file_path}")
            sys.exit(1)
        
        if not file_path_obj.suffix.lower() in ['.txt', '.md', '.text']:
            print(f"⚠️ Warning: File should be a .txt or .md file")
        
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"❌ Error: File is empty")
            sys.exit(1)
        
        print(f"✅ Successfully loaded BRD from: {file_path}")
        print(f"📄 File size: {len(content)} characters\n")
        return content
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

async def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("📋 BRD File Input Required")
        print("="*70)
        print("\nUsage:")
        print("  python agile_workflow_file.py <path_to_brd_file.txt>")
        print("\nExample:")
        print("  python agile_workflow_file.py my_brd.txt")
        print("  python agile_workflow_file.py C:\\path\\to\\requirements.txt")
        print("\n" + "="*70)
        print("\n💡 Tip: Create a .txt file with your BRD and pass it as an argument")
        sys.exit(1)
    
    # Read BRD from file
    brd_file = sys.argv[1]
    brd_text = read_brd_file(brd_file)
    
    # Run the workflow
    await run_agile_workflow(brd_text)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())

