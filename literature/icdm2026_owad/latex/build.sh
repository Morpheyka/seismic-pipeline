#!/usr/bin/env bash
# Rebuild ICDM OWAD PDFs (IEEEtran conference).
# Default: anonymous EN submission (paper_en.pdf) + RU draft.
# Optional: PAPER_CAMERA=1 also builds named paper_en_camera.pdf.
set -euo pipefail
cd "$(dirname "$0")"

build_one() {
  local stem="$1"
  echo "==> building ${stem}.pdf"
  xelatex -interaction=nonstopmode "${stem}.tex" >/dev/null
  bibtex "${stem}" >/dev/null || true
  xelatex -interaction=nonstopmode "${stem}.tex" >/dev/null
  xelatex -interaction=nonstopmode "${stem}.tex" >/dev/null
}

build_one paper_en
if [[ "${PAPER_CAMERA:-0}" == "1" ]]; then
  build_one paper_en_camera
fi
build_one paper_ru

ls -la paper_en.pdf paper_ru.pdf
if [[ -f paper_en_camera.pdf ]]; then
  ls -la paper_en_camera.pdf
fi
