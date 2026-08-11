"""PDF compilation helpers for ReportGenerator."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional


def detect_pdf_engine() -> Optional[str]:
    """Detect available PDF compilation engine (pandoc)."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "pandoc"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def compile_markdown_to_pdf(
    md_path: str,
    output_dir: str,
    pdf_engine: Optional[str] = None,
) -> None:
    """Compile markdown file to PDF using pandoc."""
    pdf_file_name = os.path.splitext(os.path.basename(md_path))[0] + ".pdf"
    pdf_path = os.path.join(output_dir, pdf_file_name)

    if pdf_engine is None:
        pdf_engine = detect_pdf_engine()

    if pdf_engine is None:
        print("Warning: No PDF engine found. Install pandoc to enable PDF compilation.")
        print("  Install pandoc: https://pandoc.org/installing.html")
        return

    try:
        if pdf_engine == "pandoc":
            md_path_abs = os.path.abspath(md_path)
            pdf_path_abs = os.path.abspath(pdf_path)
            output_dir_abs = os.path.abspath(output_dir)

            header_file = tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False, encoding="utf-8")
            header_file.write("\\usepackage[utf8]{inputenc}\n")
            header_file.write("\\usepackage[russian]{babel}\n")
            header_file.write("\\renewcommand{\\contentsname}{Содержание}\n")
            header_file.write("\\usepackage{float}\n")
            header_file.write("\\usepackage{placeins}\n")
            header_file.write("\\floatplacement{figure}{H}\n")
            header_file.write("\\floatplacement{table}{H}\n")
            header_file.write("\\let\\oldincludegraphics\\includegraphics\n")
            header_file.write(
                "\\renewcommand{\\includegraphics}[2][]{\\FloatBarrier\\oldincludegraphics[#1]{#2}\\FloatBarrier}\n"
            )
            header_file.close()

            cmd = [
                "pandoc",
                md_path_abs,
                "-o",
                pdf_path_abs,
                "--pdf-engine=xelatex",
                "--standalone",
                "--toc",
                "--wrap=none",
                "--include-in-header",
                header_file.name,
                "--variable",
                "lang=ru-RU",
                "--variable",
                "geometry:margin=1in",
                "--variable",
                "mainfont:DejaVu Serif",
                "--variable",
                "sansfont:DejaVu Sans",
                "--variable",
                "monofont:DejaVu Sans Mono",
            ]

            result = subprocess.run(cmd, cwd=output_dir_abs, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print("Warning: xelatex failed, trying pdflatex...")
                cmd[4] = "--pdf-engine=pdflatex"
                result = subprocess.run(cmd, cwd=output_dir_abs, capture_output=True, text=True, check=False)

            try:
                if os.path.exists(header_file.name):
                    os.unlink(header_file.name)
            except Exception:
                pass

            if result.returncode == 0:
                print(f"Compiled PDF report to: {pdf_path}")
            else:
                print(f"Error compiling PDF: {result.stderr}")
                if result.stdout:
                    print(f"Pandoc stdout: {result.stdout}")
        else:
            print(f"PDF engine '{pdf_engine}' is not yet supported. Using pandoc.")
            compile_markdown_to_pdf(md_path, output_dir, "pandoc")

    except FileNotFoundError:
        print(f"Error: {pdf_engine} not found. Please install it to enable PDF compilation.")
    except Exception as e:
        print(f"Error compiling PDF: {e}")
