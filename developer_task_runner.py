import asyncio
import os
from dotenv import load_dotenv
from agents import Runner
from agileagents.decomposer_agent import TaskItem, DecompositionResponse
from agileagents.developer_agent import developer_agent

# Load environment variables
load_dotenv(override=True)

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key before running this script.")
    exit(1)
else:
    print(f"✅ API key loaded (starts with: {api_key[:10]}...)")

# Sample decomposition response from AgileMaster
decomposition = DecompositionResponse(items=[
    TaskItem(
        epic="User Management",
        user_story="As a user, I want to register and log in so I can save my order history and preferences.",
        task="Develop the user registration and login system."
    ),
    TaskItem(
        epic="Shopping Cart",
        user_story="As a user, I want to view my cart so I can see the items I am about to purchase.",
        task="Implement the cart display functionality."
    ),
    TaskItem(
        epic="Checkout",
        user_story="As a user, I want to proceed to checkout so I can enter my payment and shipping details.",
        task="Create the checkout page layout."
    )
])

# Async function to run DeveloperAgent for each task
async def run_developer_task(task_item: TaskItem):
    context = (
        f"Epic: {task_item.epic}\n"
        f"User Story: {task_item.user_story}\n"
        f"Task: {task_item.task}\n"
        f"Required dependencies: Auth Library, Database Connector"
    )
    
    result = await Runner.run(developer_agent, context)
    print(f"✅ Completed: {task_item.task}")
    return task_item, result

# Main function to run all tasks concurrently and write to file
async def main():
    print(f"\n🚀 Starting development tasks for {len(decomposition.items)} items...")
    results = await asyncio.gather(*(run_developer_task(item) for item in decomposition.items))

    with open("agile_generated_code.py", "w", encoding="utf-8") as f:
        for i, (task_item, result) in enumerate(results, start=1):
            f.write(f"# Task {i}\n")
            f.write(f"# Epic: {task_item.epic}\n")
            f.write(f"# User Story: {task_item.user_story}\n")
            f.write(f"# Task: {task_item.task}\n\n")
            # Access the DevelopmentResponse from the RunResult
            dev_response = result.final_output
            f.write(dev_response.code_snippet + "\n\n")
            f.write(f"# Explanation: {dev_response.explanation}\n")
            f.write("#" + "-" * 70 + "\n\n")

    print("✅ Developer tasks completed. Code written to 'agile_generated_code.py'.")

# Run the script
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())