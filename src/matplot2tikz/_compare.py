import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ._save import get_tikz_code


def compare(figure):
    """
    Compare the matplotlib figure with the rendered TikZ code.

    This function generates the TikZ code for the given matplotlib figure,
    compiles it using pdflatex, converts the PDF to a PNG image, and
    displays it alongside the original matplotlib figure.

    :param figure: The matplotlib figure to process.
    """
    if figure is plt:
        figure = plt.gcf()

    # Check if pdflatex is installed
    pdflatex_path = shutil.which("pdflatex")
    if not pdflatex_path:
        print("pdflatex not found. Cannot compile TikZ code.")
        print("Please install a LaTeX distribution (like TeX Live or MiKTeX).")
        return

    # Create a temporary directory to show the image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tex_file = temp_path / "figure.tex"
        pdf_file = temp_path / "figure.pdf"
        png_file = temp_path / "figure.png"

        # Step 1: Get the TikZ code
        tex_code = get_tikz_code(figure, standalone=True)

        # Step 2: Write it to a file
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(tex_code)

        # Step 3: Compile with pdflatex
        try:
            cmd = [
                pdflatex_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(temp_path),
                str(tex_file),
            ]
            
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            # Step 4: Convert PDF to PNG (using helper)
            if _convert_to_png(pdf_file, png_file):
                # Display the image
                if png_file.exists():
                    img = mpimg.imread(str(png_file))
                    
                    # Create a new figure for the rendered output
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title("Rendered TikZ Output")
                    
                    plt.show()
                else:
                    print(f"Error: PNG file not found at {png_file}")
            else:
                print("Error: Conversion to PNG failed.")
                print("Please install pdftocairo (part of Poppler).")

        except subprocess.CalledProcessError as e:
            print("Error during processing:")
            if e.stderr:
                print(e.stderr.decode(errors="replace"))
            else:
                print("Process failed with no stderr output.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def _convert_to_png(pdf_path, png_path):
    """
    Convert a PDF file to a PNG file using available system tools.
    Tries pdftocairo, then ghostscript.
    """
    
    if not pdf_path:
        print("Error: PDF path not specified.")

    if not png_path:
        print("Error: PNG path not specified.")

    # Use pdftocairo to render the pdf as an image (with cairo renderer)
    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo:
        try:
            subprocess.run(
                [pdftocairo, "-png", "-singlefile", str(pdf_path), str(png_path.with_suffix(""))],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            print("Error: Render failed. This might be an issue with the new child process.")
    else:
        print("Error: Cairo renderer failed. It may not be installed.")

    

