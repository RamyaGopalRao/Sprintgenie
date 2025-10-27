"""
Generate PDF files from Tarento interview question markdown files
"""

import os
from pathlib import Path
import subprocess
import sys

# List of files to convert to PDF
INTERVIEW_FILES = [
    "TARENTO_BHASHINI_INTERVIEW_QUESTIONS.md",
    "TARENTO_CODING_QUESTIONS.md",
    "TARENTO_VS_UST_COMPARISON.md",
    "PYTHON_ARCHITECT_QUESTIONS.md",
    "ARCHITECT_INTERVIEW_QUESTIONS.md",
    "PYTHON_EVALUATION_SYSTEMS.md",
    "DEPENDENCY_INJECTION_FASTAPI.md",
]

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import markdown
        print("[+] markdown installed")
    except ImportError:
        print("[-] markdown not installed")
        print("[*] Installing markdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    
    try:
        from weasyprint import HTML
        print("[+] weasyprint installed")
    except ImportError:
        print("[-] weasyprint not installed")
        print("[*] Installing weasyprint...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "weasyprint"])
    
    try:
        import pygments
        print("[+] pygments installed")
    except ImportError:
        print("[-] pygments not installed")
        print("[*] Installing pygments...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygments"])

def markdown_to_pdf_simple(md_file, pdf_file):
    """Convert markdown to PDF using markdown2 and weasyprint"""
    try:
        import markdown2
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML with extras
        html_content = markdown2.markdown(md_content, extras=[
            'fenced-code-blocks',
            'tables',
            'header-ids',
            'toc',
            'code-friendly',
            'break-on-newline'
        ])
        
        # Add professional CSS styling
        css_style = """
        @page {
            size: A4;
            margin: 2cm;
            @bottom-right {
                content: counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
            font-size: 28pt;
            page-break-before: always;
        }
        
        h1:first-of-type {
            page-break-before: avoid;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 25px;
            font-size: 20pt;
            page-break-after: avoid;
        }
        
        h3 {
            color: #555;
            margin-top: 20px;
            font-size: 16pt;
            page-break-after: avoid;
        }
        
        h4 {
            color: #666;
            margin-top: 15px;
            font-size: 14pt;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
            color: #c7254e;
        }
        
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }
        
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        td {
            border: 1px solid #ddd;
            padding: 10px;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            background-color: #f9f9f9;
            padding: 15px 20px;
            border-radius: 4px;
        }
        
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 8px 0;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        hr {
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
        }
        
        .page-break {
            page-break-after: always;
        }
        
        strong {
            color: #2c3e50;
            font-weight: 600;
        }
        
        em {
            color: #555;
        }
        """
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{Path(md_file).stem}</title>
            <style>{css_style}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        font_config = FontConfiguration()
        HTML(string=full_html).write_pdf(
            pdf_file,
            font_config=font_config
        )
        
        return True
    except Exception as e:
        print(f"Error with weasyprint: {e}")
        return False

def convert_to_pdf_using_pandoc(md_file, pdf_file):
    """Fallback: Convert using pandoc if available"""
    try:
        subprocess.check_call([
            'pandoc',
            md_file,
            '-o', pdf_file,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--highlight-style=tango'
        ])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    """Main function to convert all markdown files to PDF"""
    print("=" * 70)
    print("TARENTO INTERVIEW QUESTIONS - PDF GENERATOR")
    print("=" * 70)
    print()
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Check and install dependencies
    print("[*] Checking dependencies...")
    try:
        import markdown2
        print("[+] markdown2 installed")
    except ImportError:
        print("[-] markdown2 not installed")
        print("[*] Installing markdown2...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown2"])
    
    try:
        from weasyprint import HTML
        print("[+] weasyprint installed")
    except ImportError:
        print("[-] weasyprint not installed")
        print("[*] Installing weasyprint (this may take a minute)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "weasyprint"])
    
    print()
    
    # Create PDFs directory
    pdf_dir = current_dir / "interview_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    print(f"[*] PDFs will be saved to: {pdf_dir}")
    print()
    
    # Convert each file
    success_count = 0
    failed_files = []
    
    for md_filename in INTERVIEW_FILES:
        md_path = current_dir / md_filename
        
        if not md_path.exists():
            print(f"[!] {md_filename} - NOT FOUND, skipping...")
            failed_files.append(md_filename)
            continue
        
        # Create PDF filename
        pdf_filename = md_filename.replace('.md', '.pdf')
        pdf_path = pdf_dir / pdf_filename
        
        print(f"[*] Converting: {md_filename}")
        print(f"    -> {pdf_filename}")
        
        # Try weasyprint first
        success = markdown_to_pdf_simple(md_path, pdf_path)
        
        if not success:
            # Try pandoc as fallback
            print("    [*] Trying pandoc as fallback...")
            success = convert_to_pdf_using_pandoc(md_path, pdf_path)
        
        if success:
            file_size = pdf_path.stat().st_size / 1024  # KB
            print(f"    [+] Success! ({file_size:.1f} KB)")
            success_count += 1
        else:
            print(f"    [-] Failed to convert")
            failed_files.append(md_filename)
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[+] Successfully converted: {success_count}/{len(INTERVIEW_FILES)} files")
    
    if failed_files:
        print(f"\n[-] Failed files:")
        for f in failed_files:
            print(f"    - {f}")
    
    print(f"\n[*] All PDFs saved in: {pdf_dir.absolute()}")
    print()
    
    # List all generated PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if pdf_files:
        print("[*] Generated PDFs:")
        total_size = 0
        for pdf_file in sorted(pdf_files):
            size_kb = pdf_file.stat().st_size / 1024
            total_size += size_kb
            print(f"    - {pdf_file.name} ({size_kb:.1f} KB)")
        print(f"\n    Total size: {total_size/1024:.2f} MB")
    
    print()
    print("=" * 70)
    print("[*] PDF generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

