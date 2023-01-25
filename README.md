## Welcome to Explainable Concept Drift in Process Mining

Please use the library [ocpa](https://github.com/ocpm/ocpa) for up-to-date OCPM methods. 

### This repository provides the code to reproduce the experiments, results and figures provided in the corresponding paper.

First, unzip example_logs/mdl/BPI2017.zip into the same directory, i.e., example_logs/mdl/BPI2017.csv

A conda environment can be used to quickly install the dependencies:
run

``conda env create --file environment.yml``

in the anaconda prompt.

Then, run

``conda activate explainable_concept_drift_experiments``

to activate the environment.

Switch to the folder of this repository and run

```python experiments.py```

to reproduce experiments and generate the figures.

