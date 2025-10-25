"""
Gradio Interface for BRD File Upload
Upload a BRD text file and get generated code!
"""
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

def process_brd_file(file_obj):
    """Process uploaded BRD file"""
    
    if file_obj is None:
        return "❌ Please upload a BRD file (.txt or .md)", ""
    
    try:
        # Read the uploaded file
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = file_obj
        
        with open(file_path, 'r', encoding='utf-8') as f:
            brd_text = f.read().strip()
        
        if not brd_text:
            return "❌ Error: Uploaded file is empty", ""
        
        # Run the workflow
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        
        async def workflow():
            status = "📋 Step 1/3: Decomposing BRD into tasks...\n"
            
            # Decompose
            decompose_result = await Runner.run(decomposer_agent, f"BRD Text:\n{brd_text}")
            decomposition = decompose_result.final_output
            
            status += f"✅ Found {len(decomposition.items)} tasks\n\n"
            
            task_list = ""
            for i, task in enumerate(decomposition.items, 1):
                task_list += f"{i}. {task.task} (Epic: {task.epic})\n"
            
            status += task_list + "\n📝 Step 2/3: Generating code and tests for all tasks...\n"
            status += "⏳ This may take 2-4 minutes...\n\n"
            
            # Process all tasks
            results = await asyncio.gather(
                *(process_single_task(task, i) 
                  for i, task in enumerate(decomposition.items, 1))
            )
            
            status += f"✅ All {len(results)} tasks completed!\n\n"
            status += "📊 Step 3/3: Generating summary...\n"
            
            # Generate summary
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
            
            # Save summary
            summary_file = output_dir / "workflow_summary.md"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            
            status += "✅ **Workflow Complete!**\n\n"
            status += f"📁 Files saved in: `{output_dir.absolute()}`\n"
            status += f"📄 Total files created: {len(results) * 2 + 1}\n"
            status += f"\n💾 Summary saved to: `workflow_summary.md`\n"
            
            return status, summary
        
        result = loop.run_until_complete(workflow())
        return result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n"
        error_msg += "Please make sure:\n"
        error_msg += "- File is a valid text file (.txt or .md)\n"
        error_msg += "- File contains your BRD text\n"
        error_msg += "- OPENAI_API_KEY is set in .env\n"
        return error_msg, ""

# Create Gradio Interface
demo = gr.Interface(
    fn=process_brd_file,
    inputs=gr.File(
        label="📁 Upload BRD File (.txt or .md)",
        file_types=[".txt", ".md"],
        type="filepath"
    ),
    outputs=[
        gr.Textbox(
            label="📊 Status & Progress",
            lines=20,
            show_copy_button=True
        ),
        gr.Markdown(label="📄 Detailed Summary")
    ],
    title="🚀 AgileMaster - BRD File Upload",
    description="""
    ## Upload Your Business Requirements Document
    
    **Steps:**
    1. 📝 Upload a .txt or .md file with your BRD
    2. ⏳ Wait 2-4 minutes for processing
    3. 📁 Check `generated_tasks/` folder for output files
    
    **What You'll Get:**
    - ✅ Python code for each task
    - ✅ Unit tests for each task
    - ✅ Complete summary report
    
    **Example BRD File:**
    ```
    Build a Task Manager
    
    Features:
    1. User authentication
    2. Create and manage tasks
    3. Task categories
    ```
    
    **Processing time:** ~20-30 seconds per task (all tasks run in parallel)
    """,
    examples=[
        ["2_openai/Agilepilot/sample_brd.txt"]
    ],
    allow_flagging="never",
    theme="soft"
)

if __name__ == "__main__":
    print("🚀 Starting Gradio Interface...")
    print("📁 Upload your BRD file to generate code!")
    demo.launch(share=False, server_port=7860)

