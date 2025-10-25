# 📁 How to Use File Input for BRD

## 🎯 New Feature: Upload BRD as Text File!

You can now provide your Business Requirements Document as a `.txt` or `.md` file!

## 🚀 Quick Start

### Step 1: Create Your BRD File

Create a text file with your requirements:

**Example: `my_project.txt`**
```
Business Requirements Document
My Awesome Project

Features:
1. User authentication with email verification
2. Dashboard with analytics
3. Data export functionality
...
```

### Step 2: Run the Workflow

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py my_project.txt
```

That's it! The workflow will:
1. ✅ Read your BRD file
2. ✅ Decompose into tasks
3. ✅ Generate code for each task
4. ✅ Create unit tests
5. ✅ Save everything to `generated_tasks/`

## 📝 Example Usage

### Example 1: Simple Project

**File: `auth_system.txt`**
```
Build a user authentication system with:
- Registration with email verification
- Login with password hashing
- Password reset functionality
- User profile management
```

**Run:**
```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py auth_system.txt
```

### Example 2: E-Learning Platform (Provided!)

We've included a sample BRD for you:

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py sample_brd.txt
```

This will process a complete e-learning platform with:
- User management
- Course management
- Enrollment system
- Learning experience features
- Communication tools

### Example 3: Full Path

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py "C:\Users\YourName\Documents\my_brd.txt"
```

## 📄 Supported File Formats

- ✅ `.txt` (Plain text)
- ✅ `.md` (Markdown)
- ✅ `.text` (Plain text)

## 🎨 BRD File Tips

### Good BRD Structure:

```
Project Title

Overview:
Brief description of what you're building

Features:
1. Feature Category 1
   - Specific requirement
   - Specific requirement
   
2. Feature Category 2
   - Specific requirement
   - Specific requirement

Technical Requirements:
- Requirement 1
- Requirement 2
```

### Be Specific!

**❌ Too vague:**
```
Build a website with user features
```

**✅ Better:**
```
Build a website with:
1. User registration with email verification
2. User login with remember me option
3. Profile management with avatar upload
4. Password reset via email
```

## 📊 Output

After running, check the `generated_tasks/` folder:

```
generated_tasks/
├── task_01_*.py              # Code for task 1
├── test_task_01_*.py         # Tests for task 1
├── task_02_*.py              # Code for task 2
├── test_task_02_*.py         # Tests for task 2
├── ...
└── workflow_summary.md       # Complete summary
```

## 🔄 Complete Workflow

```
Your BRD File (my_project.txt)
    ↓
Read File
    ↓
Decomposer Agent → Break into tasks
    ↓
Developer Agent → Generate code (parallel)
    ↓
Tester Agent → Create tests (parallel)
    ↓
Save Files → generated_tasks/
```

## 💡 Pro Tips

1. **Keep BRDs Clear**
   - Use bullet points or numbered lists
   - Group related features together
   - Be specific about requirements

2. **File Size**
   - Recommended: 500-5000 words
   - Too short: May not generate enough tasks
   - Too long: May take longer to process

3. **Processing Time**
   - ~20-30 seconds per task
   - 5 tasks = ~2 minutes total
   - All tasks processed in parallel!

4. **Multiple BRDs**
   - Process different BRDs sequentially
   - Files in `generated_tasks/` will be overwritten
   - Consider backing up previous outputs

## 🎯 Comparison

### File Input vs Hardcoded:

**File Input (`agile_workflow_file.py`):**
- ✅ Easy to change BRD (just edit the file)
- ✅ Can version control your BRDs
- ✅ Can share BRD files with team
- ✅ Command line argument

**Hardcoded (`agile_workflow.py`):**
- ✅ No external files needed
- ✅ Good for testing
- ✅ BRD is in the script

**Both produce the same output!**

## 🐛 Troubleshooting

**Error: "File not found"**
- Check the file path
- Use quotes for paths with spaces
- Use absolute path if needed

**Error: "File is empty"**
- Make sure your BRD file has content
- Check file encoding (should be UTF-8)

**Slow processing**
- Normal! AI generation takes time
- ~20-30 seconds per task
- Be patient 😊

## 📚 Examples Provided

We've included `sample_brd.txt` - try it now:

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py sample_brd.txt
```

This will generate tasks for a complete e-learning platform!

## 🎊 You're Ready!

1. Create your BRD file
2. Run the command with your file
3. Check `generated_tasks/` for results
4. Start coding! 🚀

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py YOUR_FILE.txt
```

