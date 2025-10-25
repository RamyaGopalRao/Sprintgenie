# 📁 Config Folder - Agent Instructions

This folder contains the instruction prompts for each AI agent in the SprintGenie workflow.

## 📄 Files

### `decomposer_instructions.txt`
Instructions for the **Decomposer Agent** that breaks down BRDs into:
- Epics (high-level features)
- User Stories (user-focused requirements)
- Tasks (implementable work items)

### `developer_instructions.txt`
Instructions for the **Developer Agent** that generates:
- Production-ready Python code
- Comprehensive documentation
- Best practices implementation
- Type hints and error handling

### `tester_instructions.txt`
Instructions for the **Tester Agent** that creates:
- Unit tests
- Integration tests
- Edge case tests
- Security tests
- Coverage summaries

## 🔧 How It Works

Each agent file in `agileagents/` folder loads its instructions using:

```python
from pathlib import Path

def load_instructions(filename: str) -> str:
    """Load agent instructions from config file."""
    config_dir = Path(__file__).parent.parent / "config"
    instruction_file = config_dir / filename
    
    with open(instruction_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Usage
decomposer_agent = Agent(
    name="DecomposerAgent",
    instructions=load_instructions("decomposer_instructions.txt"),
    model="gpt-4o",
    output_type=DecompositionResponse
)
```

## ✨ Benefits

1. **Separation of Concerns**
   - Instructions separated from code logic
   - Easier to read and maintain

2. **Easy Customization**
   - Edit .txt files without touching Python code
   - No need to escape quotes or format strings
   - Version control friendly

3. **Team Collaboration**
   - Different team members can work on instructions
   - Clear ownership of prompt engineering
   - Easy to review changes

4. **Reusability**
   - Instructions can be shared across projects
   - Easy to A/B test different prompts
   - Simple to rollback changes

## 🎯 Customizing Instructions

### To modify an agent's behavior:

1. Open the corresponding `.txt` file
2. Edit the instructions
3. Save the file
4. Restart the workflow

**No code changes required!**

### Example Customization:

**Before:**
```txt
Write clean, documented Python code for the given task.
```

**After:**
```txt
Write clean, documented Python code for the given task.
Use FastAPI for all API endpoints.
Include Pydantic models for request/response validation.
Add OpenAPI documentation.
```

## 📝 Instruction Writing Tips

1. **Be Specific**
   - Clear requirements
   - Concrete examples
   - Expected formats

2. **Provide Context**
   - Explain the role
   - Describe the goal
   - Give guidelines

3. **Include Examples**
   - Show desired output
   - Demonstrate format
   - Illustrate best practices

4. **Set Standards**
   - Quality criteria
   - Naming conventions
   - Code organization

## 🔄 Version Control

Instructions are now tracked in Git:
- Easy to see what changed
- Can revert to previous versions
- Collaborate with team members

## 🚀 Future Enhancements

Potential additions to this folder:
- `model_config.yaml` - Model parameters (temperature, max_tokens)
- `workflow_config.yaml` - Workflow settings
- `output_templates/` - Output format templates
- Different instruction sets for different project types

## 💡 Best Practices

1. **Keep Instructions Focused**
   - One clear purpose per file
   - Avoid mixing concerns

2. **Use Clear Language**
   - Simple, direct instructions
   - Avoid ambiguity

3. **Provide Examples**
   - Show expected format
   - Illustrate edge cases

4. **Test Changes**
   - Run workflow after modifications
   - Verify output quality

## 📚 Related Files

- `../agileagents/decomposer_agent.py` - Loads decomposer_instructions.txt
- `../agileagents/developer_agent.py` - Loads developer_instructions.txt
- `../agileagents/tester_agent.py` - Loads tester_instructions.txt

---

**Easily customize your AI agents by editing these instruction files!** 🎨

