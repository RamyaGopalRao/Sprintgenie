@echo off
echo ============================================================
echo TARENTO INTERVIEW QUESTIONS - PDF CONVERSION HELPER
echo ============================================================
echo.
echo This script will help you convert all interview questions to PDF
echo.
echo OPTION 1: Open in VS Code (Recommended if you have it)
echo OPTION 2: Open CloudConvert website for batch conversion
echo.
choice /C 12 /M "Choose your option (1 or 2)"

if errorlevel 2 goto cloudconvert
if errorlevel 1 goto vscode

:vscode
echo.
echo Opening files in VS Code...
echo.
echo After opening, for each file:
echo 1. Press Ctrl+Shift+P
echo 2. Type "Markdown PDF"
echo 3. Select "Markdown PDF: Export (pdf)"
echo.
pause
code "TARENTO_VS_UST_COMPARISON.md"
code "TARENTO_BHASHINI_INTERVIEW_QUESTIONS.md"
code "TARENTO_CODING_QUESTIONS.md"
code "PYTHON_ARCHITECT_QUESTIONS.md"
code "ARCHITECT_INTERVIEW_QUESTIONS.md"
code "PYTHON_EVALUATION_SYSTEMS.md"
code "DEPENDENCY_INJECTION_FASTAPI.md"
goto end

:cloudconvert
echo.
echo Opening CloudConvert website...
echo.
echo Steps:
echo 1. The website will open in your browser
echo 2. Click "Select Files"
echo 3. Select all 7 .md files from this folder
echo 4. Click "Convert"
echo 5. Download the PDFs
echo.
start https://cloudconvert.com/md-to-pdf
echo.
echo The website is now open. Follow the steps above.
echo.
echo Current folder: %CD%
echo.
pause

:end
echo.
echo ============================================================
echo All Interview Files:
echo ============================================================
echo 1. TARENTO_VS_UST_COMPARISON.md
echo 2. TARENTO_BHASHINI_INTERVIEW_QUESTIONS.md
echo 3. TARENTO_CODING_QUESTIONS.md
echo 4. PYTHON_ARCHITECT_QUESTIONS.md
echo 5. ARCHITECT_INTERVIEW_QUESTIONS.md
echo 6. PYTHON_EVALUATION_SYSTEMS.md
echo 7. DEPENDENCY_INJECTION_FASTAPI.md
echo ============================================================
echo.
echo For more conversion methods, see: HOW_TO_CREATE_PDFS.md
echo.
pause

