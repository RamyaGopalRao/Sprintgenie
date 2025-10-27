"""
Create a single combined PDF from all Tarento interview markdown files
Uses reportlab for PDF generation (lighter dependency)
"""

import os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors
except ImportError:
    print("[!] reportlab not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors

# Interview files to include
INTERVIEW_FILES = [
    ("TARENTO_VS_UST_COMPARISON.md", "Tarento vs UST - Career Comparison"),
    ("TARENTO_BHASHINI_INTERVIEW_QUESTIONS.md", "Tarento Bhashini Interview Questions"),
    ("TARENTO_CODING_QUESTIONS.md", "Tarento Coding Interview Questions"),
    ("PYTHON_ARCHITECT_QUESTIONS.md", "Python Architect Questions"),
    ("ARCHITECT_INTERVIEW_QUESTIONS.md", "FastAPI & Airflow Architecture Questions"),
    ("PYTHON_EVALUATION_SYSTEMS.md", "Python Evaluation Systems"),
    ("DEPENDENCY_INJECTION_FASTAPI.md", "Dependency Injection in FastAPI"),
]

def create_styles():
    """Create custom styles for the PDF"""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        bold=True
    ))
    
    # Heading styles
    styles.add(ParagraphStyle(
        name='CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=20,
        spaceAfter=12,
        bold=True
    ))
    
    styles.add(ParagraphStyle(
        name='CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceBefore=16,
        spaceAfter=10,
        bold=True
    ))
    
    styles.add(ParagraphStyle(
        name='CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#555'),
        spaceBefore=12,
        spaceAfter=8,
        bold=True
    ))
    
    # Code style
    styles.add(ParagraphStyle(
        name='CustomCode',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backgroundColor=colors.HexColor('#f8f8f8'),
        leftIndent=20,
        rightIndent=20,
        spaceBefore=6,
        spaceAfter=6
    ))
    
    # Body text
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    ))
    
    return styles

def simple_markdown_to_flowables(md_content, styles):
    """Convert simple markdown to reportlab flowables"""
    flowables = []
    lines = md_content.split('\n')
    
    i = 0
    in_code_block = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.startswith('```'):
            if in_code_block:
                # End of code block
                code_text = '\n'.join(code_lines)
                flowables.append(Preformatted(code_text, styles['CustomCode']))
                code_lines = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Headings
        if line.startswith('# '):
            text = line[2:].strip()
            # Remove markdown formatting
            text = text.replace('**', '').replace('*', '').replace('`', '')
            flowables.append(Paragraph(text, styles['CustomH1']))
            flowables.append(Spacer(1, 0.2*inch))
        elif line.startswith('## '):
            text = line[3:].strip()
            text = text.replace('**', '').replace('*', '').replace('`', '')
            flowables.append(Paragraph(text, styles['CustomH2']))
            flowables.append(Spacer(1, 0.1*inch))
        elif line.startswith('### '):
            text = line[4:].strip()
            text = text.replace('**', '').replace('*', '').replace('`', '')
            flowables.append(Paragraph(text, styles['CustomH3']))
            flowables.append(Spacer(1, 0.1*inch))
        elif line.startswith('#### '):
            text = line[5:].strip()
            text = text.replace('**', '').replace('*', '').replace('`', '')
            flowables.append(Paragraph(text, styles['Heading4']))
        elif line.strip() == '---' or line.strip() == '***':
            # Horizontal rule
            flowables.append(Spacer(1, 0.2*inch))
        elif line.strip() and not line.startswith('|'):
            # Regular text
            text = line.strip()
            # Basic markdown cleanup
            text = text.replace('**', '<b>').replace('**', '</b>')
            text = text.replace('*', '<i>').replace('*', '</i>')
            text = text.replace('`', '<font name="Courier">')
            text = text.replace('`', '</font>')
            try:
                flowables.append(Paragraph(text, styles['CustomBody']))
            except:
                # If paragraph fails, add as preformatted
                flowables.append(Preformatted(line, styles['Code']))
        elif line.strip() == '':
            flowables.append(Spacer(1, 0.1*inch))
        
        i += 1
    
    return flowables

def create_pdf():
    """Create a single PDF with all interview questions"""
    print("=" * 70)
    print("TARENTO INTERVIEW QUESTIONS - PDF CREATOR")
    print("=" * 70)
    print()
    
    current_dir = Path(__file__).parent
    pdf_dir = current_dir / "interview_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    # Create individual PDFs
    styles = create_styles()
    
    print("[*] Creating individual PDFs...")
    print()
    
    created_pdfs = []
    
    for md_file, title in INTERVIEW_FILES:
        md_path = current_dir / md_file
        
        if not md_path.exists():
            print(f"[!] {md_file} - NOT FOUND, skipping...")
            continue
        
        pdf_filename = md_file.replace('.md', '.pdf')
        pdf_path = pdf_dir / pdf_filename
        
        print(f"[*] Creating: {pdf_filename}")
        
        try:
            # Read markdown content
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create PDF
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )
            
            # Build content
            story = []
            story.append(Paragraph(title, styles['CustomTitle']))
            story.append(Spacer(1, 0.5*inch))
            
            # Convert markdown to flowables (simplified)
            flowables = simple_markdown_to_flowables(content, styles)
            story.extend(flowables)
            
            # Build PDF
            doc.build(story)
            
            size_kb = pdf_path.stat().st_size / 1024
            print(f"    [+] Success! ({size_kb:.1f} KB)")
            created_pdfs.append((pdf_filename, size_kb))
            
        except Exception as e:
            print(f"    [-] Error: {e}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[+] Created {len(created_pdfs)} PDFs")
    print()
    print("[*] Generated PDFs:")
    total_size = 0
    for filename, size_kb in created_pdfs:
        print(f"    - {filename} ({size_kb:.1f} KB)")
        total_size += size_kb
    print(f"\n    Total size: {total_size/1024:.2f} MB")
    print()
    print(f"[*] All PDFs saved in: {pdf_dir.absolute()}")
    print()
    print("=" * 70)
    print("[*] PDF creation complete!")
    print("=" * 70)

if __name__ == "__main__":
    create_pdf()

