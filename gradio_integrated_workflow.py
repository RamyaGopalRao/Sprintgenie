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
        "code_file": str(code_filepath),
        "test_file": str(test_filepath)
    }

async def run_complete_workflow(brd_text: str):
    """Complete workflow: decompose -> develop -> test"""
    
    if not brd_text or brd_text.strip() == "":
        yield "❌ Please enter a BRD text", [], ""
        return
    
    try:
        # Step 1: Decompose BRD
        status = "📋 Step 1/3: Decomposing BRD into tasks...\n"
        yield status, [], ""
        
        decompose_result = await Runner.run(decomposer_agent, f"BRD Text:\n{brd_text}")
        decomposition = decompose_result.final_output
        
        status += f"✅ Found {len(decomposition.items)} tasks\n\n"
        
        # Prepare task rows for display
        task_rows = []
        for i, task in enumerate(decomposition.items, 1):
            task_rows.append([
                i,
                task.epic,
                task.user_story,
                task.task,
                "⏳ Pending"
            ])
        
        yield status + "📝 Step 2/3: Generating code and tests for all tasks...\n", task_rows, ""
        
        # Step 2: Process all tasks
        results = await asyncio.gather(
            *(process_single_task(task, i) 
              for i, task in enumerate(decomposition.items, 1))
        )
        
        # Update task status
        for i, result in enumerate(results):
            task_rows[i][4] = "✅ Complete"
        
        status += f"✅ All {len(results)} tasks completed\n\n"
        yield status + "📊 Step 3/3: Generating summary...\n", task_rows, ""
        
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
            summary += f"- **Code:** `{Path(result['code_file']).name}`\n"
            summary += f"- **Tests:** `{Path(result['test_file']).name}`\n\n"
            summary += f"### 💡 Code Explanation:\n\n"
            summary += f"{result['code'].explanation}\n\n"
            summary += f"### 🧪 Test Coverage:\n\n"
            summary += f"{result['tests'].coverage_summary}\n\n"
            summary += "---\n\n"
        
        status += "✅ **Workflow Complete!**\n\n"
        status += f"📁 Files saved in: `{output_dir.absolute()}`\n"
        status += f"📄 Total files created: {len(results) * 2}\n"
        
        yield status, task_rows, summary
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n"
        yield error_msg, task_rows if 'task_rows' in locals() else [], ""

def sync_wrapper(brd_text):
    """Synchronous wrapper for Gradio"""
    import nest_asyncio
    nest_asyncio.apply()
    
    loop = asyncio.get_event_loop()
    
    # Create async generator
    async_gen = run_complete_workflow(brd_text)
    
    # Get final result
    result = None
    async def get_result():
        nonlocal result
        async for r in async_gen:
            result = r
        return result
    
    loop.run_until_complete(get_result())
    return result

# Create Gradio interface
with gr.Blocks(title="AgileMaster - Complete Workflow", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 AgileMaster - Complete Agile Workflow
    
    Upload or enter a Business Requirements Document (BRD) to automatically:
    1. 📋 **Decompose** into Epics, User Stories, and Tasks
    2. 💻 **Generate** Python code for each task
    3. 🧪 **Create** unit tests for the code
    4. 💾 **Save** separate files for each task
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            brd_input = gr.Textbox(
                label="Business Requirements Document",
                placeholder="Enter your BRD here or use the example below...",
                lines=10,
                value="""Build an e-commerce platform with the following features:
1. User Management: Users should be able to register, login, and manage their profiles
2. Shopping Cart: Users should be able to add items to cart and view cart contents
3. Checkout: Users should be able to proceed to checkout and complete payment"""
            )
            
            submit_btn = gr.Button("🚀 Start Workflow", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Status",
                lines=15,
                interactive=False
            )
    
    gr.Markdown("## 📋 Decomposed Tasks")
    
    task_table = gr.Dataframe(
        headers=["#", "Epic", "User Story", "Task", "Status"],
        label="Tasks",
        interactive=False,
        wrap=True
    )
    
    gr.Markdown("## 📊 Detailed Summary")
    
    summary_output = gr.Markdown()
    
    submit_btn.click(
        fn=sync_wrapper,
        inputs=[brd_input],
        outputs=[status_output, task_table, summary_output]
    )
    
    gr.Markdown("""
    ---
    ### 📁 Output Files
    All generated files will be saved in the `generated_tasks/` directory:
    - `task_XX_<task_name>.py` - Implementation code
    - `test_task_XX_<task_name>.py` - Unit tests
    """)

if __name__ == "__main__":
    demo.launch(share=False)

