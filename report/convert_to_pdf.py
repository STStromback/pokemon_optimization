"""
Generate PDF from HTML using Playwright (headless browser).
This is more reliable on Windows than weasyprint.
"""

from pathlib import Path
import subprocess
import sys
import time

def install_playwright():
    """Install playwright and chromium browser."""
    print("Installing playwright for PDF generation...")
    try:
        # Install playwright
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
        print("✓ playwright installed")
        
        # Install chromium browser
        print("Installing chromium browser (this may take a minute)...")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("✓ chromium browser installed")
        return True
    except Exception as e:
        print(f"✗ Failed to install playwright: {e}")
        return False

def create_pdf_with_playwright(html_file, pdf_file):
    """Create PDF using playwright headless browser."""
    try:
        from playwright.sync_api import sync_playwright
        
        print(f"Generating PDF from {html_file}...")
        print("(Waiting for MathJax to render equations...)")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load HTML file
            page.goto(f'file:///{html_file.as_posix()}')
            
            # Wait for MathJax to render
            time.sleep(3)
            
            # Generate PDF
            page.pdf(
                path=str(pdf_file),
                format='Letter',
                margin={
                    'top': '0.75in',
                    'right': '0.75in',
                    'bottom': '0.75in',
                    'left': '0.75in'
                },
                print_background=True
            )
            
            browser.close()
        
        print(f"✓ PDF file created: {pdf_file}")
        return True
        
    except ImportError:
        print("\n✗ playwright is not installed.")
        print("\nAttempting to install playwright...")
        if install_playwright():
            # Try again after installation
            try:
                from playwright.sync_api import sync_playwright
                
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    page.goto(f'file:///{html_file.as_posix()}')
                    time.sleep(3)
                    page.pdf(
                        path=str(pdf_file),
                        format='Letter',
                        margin={'top': '0.75in', 'right': '0.75in', 'bottom': '0.75in', 'left': '0.75in'},
                        print_background=True
                    )
                    browser.close()
                
                print(f"✓ PDF file created: {pdf_file}")
                return True
            except Exception as e:
                print(f"✗ Error after installation: {e}")
                return False
        return False
        
    except Exception as e:
        print(f"✗ Error creating PDF: {e}")
        return False


def create_pdf_via_browser_instructions():
    """Provide instructions for creating PDF via browser."""
    print("\n" + "="*70)
    print("ALTERNATIVE: Create PDF via Browser (Manual Method)")
    print("="*70)
    print("\n1. The HTML file should have opened in your browser automatically")
    print("2. If not, open this file manually:")
    print(f"   {Path(__file__).parent / 'pokemon_optimization_report.html'}")
    print("\n3. Press Ctrl+P (or Cmd+P on Mac) to open the print dialog")
    print("\n4. In the print dialog:")
    print("   - Destination: Select 'Save as PDF' or 'Microsoft Print to PDF'")
    print("   - Paper size: Letter")
    print("   - Margins: Default or Custom (0.75 inches all sides)")
    print("   - Background graphics: ON (to include images)")
    print("\n5. Click 'Save' and save as: pokemon_optimization_report.pdf")
    print("\n6. Choose this directory as the save location:")
    print(f"   {Path(__file__).parent}")
    print("\n" + "="*70)
    print("\nNote: Make sure to wait a few seconds after the page loads")
    print("      to allow mathematical equations to render before printing.")
    print("="*70)


def main():
    """Main PDF creation function."""
    script_dir = Path(__file__).parent
    html_file = script_dir / "pokemon_optimization_report.html"
    pdf_file = script_dir / "pokemon_optimization_report.pdf"
    
    if not html_file.exists():
        print(f"✗ HTML file not found: {html_file}")
        print("Please run convert_to_formats.py first to create the HTML file.")
        return
    
    print("="*70)
    print("PDF GENERATION")
    print("="*70)
    print("\nCreating PDF from HTML using playwright (headless browser)...\n")
    
    success = create_pdf_with_playwright(html_file, pdf_file)
    
    if success:
        print("\n" + "="*70)
        print("✓ SUCCESS!")
        print("="*70)
        print(f"\nBoth files have been created:")
        print(f"  - HTML: {html_file}")
        print(f"  - PDF: {pdf_file}")
        print("\n" + "="*70)
    else:
        print("\n⚠ Automated PDF generation failed.")
        create_pdf_via_browser_instructions()


if __name__ == "__main__":
    main()
