# Build the LaTeX report.
#
# Requires a TeX distribution with pdflatex. On macOS:
#   brew install --cask basictex     # ~100 MB
#   eval "$(/usr/libexec/path_helper)"
#   sudo tlmgr update --self
#   sudo tlmgr install latexmk fancyvrb fvextra biblatex booktabs microtype
#
# Or use the full MacTeX / MiKTeX install.
#
# Targets:
#   make pdf      # build report.pdf
#   make clean    # remove build artifacts
#   make watch    # rebuild on save (requires latexmk)

TEX     := pdflatex
LATEXMK := latexmk

SRC  := report.tex
PDF  := report.pdf

.PHONY: pdf clean watch

pdf: $(PDF)

$(PDF): $(SRC)
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC) > /dev/null
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC) > /dev/null

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz $(PDF)
