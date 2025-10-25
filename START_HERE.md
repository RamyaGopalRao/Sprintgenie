# 🚀 START HERE - Your Agile Workflow

## ✨ What This Does

Converts your Business Requirements Document (BRD) into:
- ✅ Python code files (one per task)
- ✅ Unit test files (one per task)
- ✅ Complete documentation

**All automatically using AI agents!** 🤖

## ⚡ Quick Start (3 Steps)

### Step 1: Create Your BRD File

Create a text file (e.g., `my_project.txt`) with your requirements:

```txt
Build a Blog Platform

Features:
1. User authentication
2. Create and edit blog posts
3. Comment system
4. User profiles
```

### Step 2: Run the Command

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py my_project.txt
```

### Step 3: Get Your Code

Check the `generated_tasks/` folder for:
- Python code files
- Unit test files
- Summary report

**That's it!** 🎉

## 🎯 Try the Example

We included a sample e-learning platform BRD:

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py 2_openai\Agilepilot\sample_brd.txt
```

## 📁 What You'll Get

```
generated_tasks/
├── task_01_user_registration.py      ← Implementation
├── test_task_01_user_registration.py ← Tests
├── task_02_course_creation.py        ← Implementation
├── test_task_02_course_creation.py   ← Tests
├── ... more tasks ...
└── workflow_summary.md                ← Complete report
```

## ⏱️ How Long?

- Small project (3-5 features): ~2 minutes
- Medium project (5-10 features): ~3-4 minutes
- Large project (10+ features): ~5-6 minutes

**All tasks are processed in parallel for speed!**

## 💡 BRD Writing Tips

**Good BRD Example:**
```txt
Build a Task Manager

Features:
1. User Management
   - Registration with email
   - Login with password
   - Profile editing

2. Task Management
   - Create tasks
   - Edit tasks
   - Delete tasks
   - Mark as complete

3. Categories
   - Create categories
   - Assign tasks to categories
```

**Be Specific = Better Code!**

## 🆘 Need Help?

- **Full Guide:** Read `HOW_TO_USE_FILE_INPUT.md`
- **Quick Reference:** Read `QUICK_REFERENCE.md`
- **Complete Docs:** Read `README.md`

## 🎊 Ready to Build!

1. Create your BRD file
2. Run the command
3. Start using your generated code!

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py YOUR_BRD.txt
```

**Happy Coding!** 🎨✨

