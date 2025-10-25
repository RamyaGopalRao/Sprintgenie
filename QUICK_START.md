# 🚀 Quick Start Guide - AgileMaster

## What You Have Now

✅ **Complete Agile Workflow System** with 3 AI agents:
- 📋 **Decomposer Agent** - Breaks BRD into tasks
- 💻 **Developer Agent** - Writes Python code
- 🧪 **Tester Agent** - Creates unit tests

## 🎯 Choose Your Interface

### Option 1: Web Interface (Easiest!) 🌐

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\gradio_integrated_workflow.py
```

**What You Get:**
- Beautiful web interface
- Real-time progress updates
- Interactive tables
- Easy BRD input

### Option 2: Command Line 💻

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow.py
```

**What You Get:**
- Fast execution
- Console output
- Good for automation

## 📝 Quick Example

### Input (BRD):
```
Build an e-commerce platform with:
1. User authentication
2. Shopping cart
3. Checkout system
```

### Output (6 files + summary):
```
generated_tasks/
├── task_01_user_authentication.py           ← Code
├── test_task_01_user_authentication.py      ← Tests
├── task_02_shopping_cart.py                 ← Code
├── test_task_02_shopping_cart.py            ← Tests
├── task_03_checkout_system.py               ← Code
├── test_task_03_checkout_system.py          ← Tests
└── workflow_summary.md                       ← Report
```

## ⚡ What Happens

```
Your BRD
    ↓
[Step 1] Decomposer breaks it into 3 tasks (10 sec)
    ↓
[Step 2] Developer writes code for all 3 tasks (30 sec, parallel)
    ↓
[Step 3] Tester creates tests for all 3 tasks (30 sec, parallel)
    ↓
7 files created in generated_tasks/
```

**Total Time:** ~60-90 seconds for 3 tasks

## 🎮 Try It Now!

### Web Interface Steps:

1. **Run the command** (from agents directory):
   ```bash
   .venv\Scripts\python.exe 2_openai\Agilepilot\gradio_integrated_workflow.py
   ```

2. **Open browser** to http://localhost:7860

3. **Enter your BRD** or use the example provided

4. **Click "Start Workflow"**

5. **Watch the magic happen!** ✨

6. **Check `generated_tasks/`** folder for your files

## 📁 File Structure

### What Each File Contains:

**Code Files** (`task_XX_name.py`):
```python
"""
Task metadata (Epic, User Story, Task)
"""

# Your generated Python code here
class UserAuth:
    def login(self): ...
    def register(self): ...

# Explanation:
# Detailed explanation of the code
```

**Test Files** (`test_task_XX_name.py`):
```python
"""
Test metadata
"""

import unittest

# Test Case 1:
def test_login(): ...

# Test Case 2:  
def test_register(): ...

# Coverage Summary:
# What the tests cover
```

## 🔧 Configuration

Your `.env` file should have:
```
OPENAI_API_KEY=your-api-key-here
```

## ❓ FAQ

**Q: How long does it take?**
A: ~20-30 seconds per task (development + testing)

**Q: Can I modify the agents?**
A: Yes! Edit files in `agileagents/` folder

**Q: Where are files saved?**
A: In `2_openai/Agilepilot/generated_tasks/`

**Q: Can I use different models?**
A: Yes! Change `model="gpt-4o"` to `model="gpt-4o-mini"` in agent files

**Q: Does it really work?**
A: Yes! Try it and see 🚀

## 🎯 Next Steps

1. ✅ Run the Gradio interface
2. ✅ Enter your own BRD
3. ✅ Review generated code
4. ✅ Check the tests
5. ✅ Customize agents if needed

## 💡 Pro Tips

- Start with simple BRDs to test
- Be specific in your requirements
- Each task generates ~50-200 lines of code
- Tests are comprehensive but may need tweaking
- All processing happens in parallel for speed

## 🆘 Help

**Problem**: Can't find generated files
- **Solution**: Check `2_openai/Agilepilot/generated_tasks/`

**Problem**: API key error
- **Solution**: Create `.env` file with `OPENAI_API_KEY=...`

**Problem**: Slow response
- **Solution**: Normal! AI takes time, be patient ⏱️

## 🎉 You're Ready!

Everything is set up and ready to go. Just run the command and start building! 🚀

```bash
# Start the web interface now:
.venv\Scripts\python.exe 2_openai\Agilepilot\gradio_integrated_workflow.py
```

Happy coding! 🎨

