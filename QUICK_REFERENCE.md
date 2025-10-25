# ⚡ Quick Reference Card

## 🎯 One Command to Run

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py YOUR_BRD.txt
```

## 📝 Create BRD File

```txt
# my_project.txt

Build [Your Project Name]

Features:
1. Feature 1
   - Detail
   - Detail
2. Feature 2
   - Detail
   - Detail
```

## 🚀 Complete Steps

1. Create BRD file: `my_project.txt`
2. Run: `.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py my_project.txt`
3. Wait: ~2-3 minutes
4. Check: `generated_tasks/` folder

## 📁 Output Files

```
generated_tasks/
├── task_01_*.py          ← Code
├── test_task_01_*.py     ← Tests
├── task_02_*.py          ← Code
├── test_task_02_*.py     ← Tests
└── workflow_summary.md   ← Report
```

## 🎨 What You Get

- ✅ Python code for each task
- ✅ Unit tests for each task
- ✅ Complete documentation
- ✅ Summary report

## ⚡ Try the Sample

```bash
.venv\Scripts\python.exe 2_openai\Agilepilot\agile_workflow_file.py 2_openai\Agilepilot\sample_brd.txt
```

## 💡 Tips

- Be specific in your BRD
- Use bullet points
- ~20-30 seconds per task
- All tasks run in parallel

## 🆘 Help

**File not found?**
- Check file path
- Use full path if needed

**Empty output?**
- Check your BRD file has content
- Make sure API key is set

**Slow?**
- Normal! AI takes time
- Be patient ⏱️

---

**That's it!** Create BRD → Run command → Get code! 🚀

