LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out
SUFF        = pdf

NAME        = ms

RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})

CHECK_RERUN = grep Rerun $*.log

all: ${NAME}.${SUFF}

%.pdf: %.tex *.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( ${CHECK_RERUN} && ${LATEX} ${NAME} ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} ${NAME} ) || echo "Done."

clean:
	${RM_TMP}

