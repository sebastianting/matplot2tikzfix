import shutil
import subprocess
import tempfile
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt

from ._save import get_tikz_code


def compare(figure):
    """
    Compare the matplotlib figure with the rendered TikZ code.

    This function generates the TikZ code for the given matplotlib figure,
    compiles it using pdflatex, and creates a side-by-side PDF comparison.

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
        
        # TikZ file paths
        tikz_tex_file = temp_path / "tikz_figure.tex"
        tikz_pdf_file = temp_path / "tikz_figure.pdf"

        # Matplotlib file paths
        mpl_pdf_file = temp_path / "mpl_figure.pdf"

        # Source PDF Generation
        mpl_success = False
        tikz_success = False

        # Step 1: Save original Matplotlib figure to PDF
        try:
            figure.savefig(mpl_pdf_file, format="pdf")
            plt.close(figure)
            mpl_success = True
        except Exception as e:
            print(f"Error saving matplotlib figure to PDF: {e}")

        # Step 2: Generate TikZ Code and Compile
        try:
            tex_code = get_tikz_code(figure, standalone=True)
            with open(tikz_tex_file, "w", encoding="utf-8") as f:
                f.write(tex_code)

            cmd = [
                pdflatex_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(temp_path),
                str(tikz_tex_file),
            ]
            
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            tikz_success = True

        except subprocess.CalledProcessError as e:
            print("Error during TikZ compilation:")
            if e.stderr:
                print(e.stderr.decode(errors="replace"))
            else:
                print("Process failed with no stderr output.")
        except Exception as e:
            print(f"An unexpected error occurred during TikZ generation: {e}")

        # Step 3: Create Comparison PDF
        if mpl_success and tikz_success:
            comparison_tex = temp_path / "comparison.tex"
            with open(comparison_tex, "w", encoding="utf-8") as f:
                f.write(_get_comparison_tex_template(mpl_pdf_file, tikz_pdf_file))
            
            try:
                subprocess.run(
                   [
                        pdflatex_path,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory",
                        str(temp_path),
                        str(comparison_tex),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                
                # Move result to current directory and open
                result_pdf = temp_path / "comparison.pdf"
                if result_pdf.exists():
                    dest_pdf = Path.cwd() / "matplot2tikz_comparison.pdf"
                    shutil.copy(result_pdf, dest_pdf)
                    print(f"Comparison PDF generated: {dest_pdf}")
                    webbrowser.open(str(dest_pdf))
                else:
                    print("Error: Comparison PDF was not generated.")

            except subprocess.CalledProcessError as e:
                print("Error during comparison PDF compilation:")
                if e.stderr:
                    print(e.stderr.decode(errors="replace"))
        else:
            print("Skipping comparison PDF generation due to missing source files.")


def _get_comparison_tex_template(mpl_path, tikz_path):
    # Escape Windows paths for LaTeX
    mpl_path_str = str(mpl_path).replace("\\", "/")
    tikz_path_str = str(tikz_path).replace("\\", "/")
    
    return f"""\\documentclass[landscape]{{article}}
\\usepackage{{graphicx}}
\\usepackage{{geometry}}
\\geometry{{margin=1cm}}
\\usepackage{{caption}}

\\begin{{document}}
    \\begin{{figure}}[h]
        \\centering
        \\begin{{minipage}}{{0.48\\textwidth}}
            \\centering
            \\includegraphics[width=\\linewidth]{{{mpl_path_str}}}
            \\caption*{{Matplotlib Output}}
        \\end{{minipage}}
        \\hfill
        \\begin{{minipage}}{{0.48\\textwidth}}
            \\centering
            \\includegraphics[width=\\linewidth]{{{tikz_path_str}}}
            \\caption*{{TikZ Output}}
        \\end{{minipage}}
    \\end{{figure}}
\\end{{document}}
"""


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

    

