import gradio as gr
from agileagents.decomposer_agent import TaskItem, BRDRequest, decomposer_agent
from agents import Runner

async def decompose_brd(file):
    if file is None:
        return [["Error", "Please upload a document", ""]]

    # Read file content
    brd_text = str(file)

    # Create request and run agent
    request = BRDRequest(brd_text=brd_text)

    # Optional override for testing
    brd_text = "Build an e-commerce platform with cart, checkout, and user login."
    request = BRDRequest(brd_text=brd_text)

    classify_result = await Runner.run(decomposer_agent, f"BRD Text:\n{brd_text}")
    doc_type = classify_result.final_output
    print(type(doc_type))

    # Prepare grid rows
    rows = []
    for item in doc_type.items:
        its = item  # Assuming item[1] is a TaskItem
        rows.append([its.epic, its.user_story, its.task])

    return rows
demo = gr.Interface(
    fn=decompose_brd,
    inputs=gr.File(label="Upload BRD (.txt)", file_types=[".txt"]),
    outputs=gr.Dataframe(
        headers=["Epic", "User Story", "Task"],
        label="Decomposed Agile Tasks",
        type="array"
    ),
    title="AgileMaster: BRD Decomposer",
    description="Upload a plain text BRD to extract epics, user stories, and tasks using an agentic AI pipeline."
)

demo.launch()