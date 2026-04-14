PDF=project_proposal.pdf
TEX=project_proposal.tex

.PHONY: all clean train-baseline train-simvp

all:
	tectonic $(TEX)

train-baseline:
	python3 scripts/train_baseline.py

train-simvp:
	python3 scripts/train_baseline.py --model simvp --output-dir outputs/simvp_baseline

clean:
	rm -f $(PDF) project_proposal.aux project_proposal.bbl project_proposal.blg project_proposal.log
