import gradio as gr
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from agents import Runner
from agileagents.decomposer_agent import decomposer_agent, TaskItem
from agileagents.developer_agent import developer_agent
from agileagents.tester_agent import tester_agent

# Load environment variables
load_dotenv(override=True)

# Create output directory
output_dir = Path("generated_tasks")
output_dir.mkdir(exist_ok=True)

def sanitize_filename(text: str) -> str:
    """Convert text to a valid filename"""
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    filename = ''.join(c if c in valid_chars else '_' for c in text)
    filename = '_'.join(filter(None, filename.split('_')))
    return filename[:50]

async def process_single_task(task_item: TaskItem, task_number: int):
    """Process a single task: develop code and test it"""
    
    # Generate code
    dev_context = (
        f"Epic: {task_item.epic}\n"
        f"User Story: {task_item.user_story}\n"
        f"Task: {task_item.task}\n"
        f"Required dependencies: Auth Library, Database Connector"
    )
    
    dev_result = await Runner.run(developer_agent, dev_context)
    dev_response = dev_result.final_output
    
    # Generate tests
    test_context = (
        f"Generate comprehensive unit tests for the following code:\n\n"
        f"Task: {task_item.task}\n\n"
        f"Code:\n{dev_response.code_snippet}"
    )
    
    test_result = await Runner.run(tester_agent, test_context)
    test_response = test_result.final_output
    
    # Save files
    code_filename = f"task_{task_number:02d}_{sanitize_filename(task_item.task)}.py"
    test_filename = f"test_task_{task_number:02d}_{sanitize_filename(task_item.task)}.py"
    
    code_filepath = output_dir / code_filename
    test_filepath = output_dir / test_filename
    
    # Save code file
    with open(code_filepath, "w", encoding="utf-8") as f:
        f.write(f'"""\n')
        f.write(f"Task {task_number}\n")
        f.write(f"Epic: {task_item.epic}\n")
        f.write(f"User Story: {task_item.user_story}\n")
        f.write(f"Task: {task_item.task}\n")
        f.write(f'"""\n\n')
        f.write(dev_response.code_snippet)
        f.write(f'\n\n# Explanation:\n# {dev_response.explanation}\n')
    
    # Save test file
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
    
    return {
        "task_item": task_item,
        "task_number": task_number,
        "code": dev_response,
        "tests": test_response,
        "code_file": code_filename,
        "test_file": test_filename
    }

def run_complete_workflow(brd_text: str):
    """Complete workflow: decompose -> develop -> test"""
    
    if not brd_text or brd_text.strip() == "":
        return "❌ Please enter a BRD text", ""
    
    try:
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        
        # Step 1: Decompose BRD
        status = "📋 Step 1/3: Decomposing BRD into tasks...\n"
        
        async def workflow():
            decompose_result = await Runner.run(decomposer_agent, f"BRD Text:\n{brd_text}")
            decomposition = decompose_result.final_output
            
            status = f"✅ Found {len(decomposition.items)} tasks\n\n"
            
            task_list = ""
            for i, task in enumerate(decomposition.items, 1):
                task_list += f"{i}. {task.task} (Epic: {task.epic})\n"
            
            status += task_list + "\n📝 Step 2/3: Generating code and tests for all tasks...\n"
            
            # Step 2: Process all tasks
            results = await asyncio.gather(
                *(process_single_task(task, i) 
                  for i, task in enumerate(decomposition.items, 1))
            )
            
            status += f"✅ All {len(results)} tasks completed\n\n"
            status += "📊 Step 3/3: Generating summary...\n"
            
            # Step 3: Generate summary
            summary = "# 🎯 Workflow Summary\n\n"
            summary += f"**Total Tasks:** {len(results)}\n\n"
            summary += f"**Output Directory:** `{output_dir.absolute()}`\n\n"
            summary += "---\n\n"
            
            for result in results:
                task = result["task_item"]
                num = result["task_number"]
                summary += f"## Task {num}: {task.task}\n\n"
                summary += f"**Epic:** {task.epic}\n\n"
                summary += f"**User Story:** {task.user_story}\n\n"
                summary += f"### 📄 Generated Files:\n\n"
                summary += f"- **Code:** `{result['code_file']}`\n"
                summary += f"- **Tests:** `{result['test_file']}`\n\n"
                summary += f"### 💡 Code Explanation:\n\n"
                summary += f"{result['code'].explanation}\n\n"
                summary += f"### 🧪 Test Coverage:\n\n"
                summary += f"{result['tests'].coverage_summary}\n\n"
                summary += "---\n\n"
            
            status += "✅ **Workflow Complete!**\n\n"
            status += f"📁 Files saved in: `{output_dir.absolute()}`\n"
            status += f"📄 Total files created: {len(results) * 2}\n"
            
            return status, summary
        
        result = loop.run_until_complete(workflow())
        return result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n"
        return error_msg, ""

# Create Gradio interface using Interface (simpler than Blocks)
demo = gr.Interface(
    fn=run_complete_workflow,
    inputs=gr.Textbox(
        label="Business Requirements Document",
        placeholder="Enter your BRD here...",
        lines=10,
        value="""Build an e-commerce platform with the following features:
1. User Management: Users should be able to register, login, and manage their profiles
2. Shopping Cart: Users should be able to add items to cart and view cart contents
3. Checkout: Users should be able to proceed to checkout and complete payment"""
    ),
    outputs=[
        gr.Textbox(label="Status & Progress", lines=15),
        gr.Markdown(label="Detailed Summary")
    ],
    title="🚀 AgileMaster - Complete Agile Workflow",
    description="""
    Upload or enter a Business Requirements Document (BRD) to automatically:
    1. 📋 **Decompose** into Epics, User Stories, and Tasks
    2. 💻 **Generate** Python code for each task
    3. 🧪 **Create** unit tests for the code
    4. 💾 **Save** separate files for each task
    
    **Note:** Processing time is approximately 20-30 seconds per task.
    """,
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=False)

