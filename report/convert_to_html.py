"""
Simple script to convert markdown report to HTML and PDF formats.
Uses minimal dependencies - markdown for HTML, then browser print for PDF.
"""

import re
import os
from pathlib import Path

def escape_html(text):
    """Escape HTML special characters."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def process_markdown_to_html(md_content):
    """Convert markdown to HTML (basic implementation)."""
    lines = md_content.split('\n')
    html_lines = []
    in_code_block = False
    in_table = False
    in_math_block = False
    math_content = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle math blocks ($$..$$)
        if line.strip().startswith('$$'):
            if not in_math_block:
                in_math_block = True
                math_content = ['$$']
                i += 1
                continue
            else:
                math_content.append('$$')
                # Close the math block
                html_lines.append('<div class="math-block">')
                html_lines.append('\n'.join(math_content))
                html_lines.append('</div>')
                in_math_block = False
                math_content = []
                i += 1
                continue
        
        if in_math_block:
            math_content.append(line)
            i += 1
            continue
        
        # Skip empty lines
        if not line.strip():
            html_lines.append('<br>')
            i += 1
            continue
        
        # Headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            html_lines.append(f'<h{level}>{text}</h{level}>')
        
        # Images
        elif '<img' in line:
            html_lines.append(line)
        
        # Tables
        elif '|' in line and not line.strip().startswith('$$'):
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            
            # Check if it's a separator row
            if re.match(r'^\|[\s\-:]+\|', line):
                i += 1
                continue
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            # Determine if header row (check if next line is separator)
            is_header = False
            if i + 1 < len(lines) and re.match(r'^\|[\s\-:]+\|', lines[i + 1]):
                is_header = True
            
            if is_header:
                html_lines.append('<thead><tr>')
                for cell in cells:
                    html_lines.append(f'<th>{cell}</th>')
                html_lines.append('</tr></thead><tbody>')
            else:
                html_lines.append('<tr>')
                for cell in cells:
                    # Handle inline math in cells
                    cell = re.sub(r'\$([^\$]+)\$', r'<span class="math-inline">$\1$</span>', cell)
                    html_lines.append(f'<td>{cell}</td>')
                html_lines.append('</tr>')
            
            # Check if next line is still a table
            if i + 1 < len(lines) and '|' not in lines[i + 1]:
                html_lines.append('</tbody></table>')
                in_table = False
        
        # Blockquotes
        elif line.startswith('>'):
            text = line.lstrip('>').strip()
            html_lines.append(f'<blockquote>{text}</blockquote>')
        
        # Bold text
        elif '**' in line:
            text = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', line)
            # Handle inline math
            text = re.sub(r'\$([^\$]+)\$', r'<span class="math-inline">$\1$</span>', text)
            html_lines.append(f'<p>{text}</p>')
        
        # Regular paragraphs
        else:
            # Handle inline math
            text = re.sub(r'\$([^\$]+)\$', r'<span class="math-inline">$\1$</span>', line)
            # Handle links
            text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)
            html_lines.append(f'<p>{text}</p>')
        
        i += 1
    
    if in_table:
        html_lines.append('</tbody></table>')
    
    return '\n'.join(html_lines)


def create_html_document(md_file, html_file):
    """Create a complete HTML document from markdown."""
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_body = process_markdown_to_html(md_content)
    
    # Create full HTML document
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokemon In-Game Party Optimization: Generations 1 - 3</title>
    
    <!-- MathJax for LaTeX rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']]
            }
        };
    </script>
    
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .content {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
        }
        
        h4 {
            color: #666;
            margin-top: 20px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: inline-block;
            margin: 10px 5px;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 15px 20px;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .math-block {
            margin: 20px 0;
            text-align: center;
            font-size: 110%;
        }
        
        .math-inline {
            display: inline;
        }
        
        p {
            margin: 10px 0;
        }
        
        @media print {
            body {
                background-color: white;
            }
            .content {
                box-shadow: none;
            }
            .no-print {
                display: none;
            }
        }
        
        @page {
            size: letter;
            margin: 0.75in;
        }
    </style>
</head>
<body>
    <div class="content">
        <div class="no-print">
            <p><em>To save as PDF: Press Ctrl+P (or Cmd+P on Mac), then select "Save as PDF"</em></p>
            <hr>
        </div>
        ''' + html_body + '''
    </div>
</body>
</html>'''
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✓ HTML file created: {html_file}")
    print(f"\nTo create PDF:")
    print(f"1. Open {html_file} in your web browser")
    print(f"2. Press Ctrl+P (or Cmd+P on Mac)")
    print(f"3. Select 'Save as PDF' as the destination")
    print(f"4. Click Save and choose 'pokemon_optimization_report.pdf' as the filename")
    
    return html_file


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    md_file = script_dir / "pokemon_optimization_report.md"
    html_file = script_dir / "pokemon_optimization_report.html"
    
    print("Converting markdown report to HTML...\n")
    create_html_document(md_file, html_file)
    print("\n✓ Conversion complete!")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(str(html_file))
        print(f"\n✓ Opened {html_file} in your default browser")
    except:
        print(f"\nPlease open {html_file} in your browser manually")


if __name__ == "__main__":
    main()
