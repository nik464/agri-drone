# compile.ps1 — Build the Springer LNCS PDF
# Requirements: MiKTeX or TeX Live installed
#   Install MiKTeX: https://miktex.org/download
#   Or install TeX Live: https://tug.org/texlive/
#
# Usage (from d:\Projects\agri-drone\paper\):
#   .\compile.ps1

Set-Location $PSScriptRoot

Write-Host "Pass 1: pdflatex..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode main.tex

Write-Host "Pass 2: bibtex..." -ForegroundColor Cyan
bibtex main

Write-Host "Pass 3: pdflatex..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode main.tex

Write-Host "Pass 4: pdflatex (final)..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode main.tex

if (Test-Path "main.pdf") {
    Write-Host ""
    Write-Host "✅  PDF built: $PSScriptRoot\main.pdf" -ForegroundColor Green
    # Open the PDF automatically
    Start-Process "main.pdf"
} else {
    Write-Host "❌  Build failed — check main.log for errors" -ForegroundColor Red
}
