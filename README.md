# Belyev UIRS

This project analyzes vehicle motion measurements across terrain types and estimates the normalized kinetic-energy coefficient (`Ke`). The pipeline filters measurements by experiment, removes outliers, aggregates values on a symmetric angular-velocity grid, fits discrete cosine transform (DCT) models, and evaluates terrain classification with leave-one-experiment-out validation.

## Run

Install the project dependencies and run the main pipeline:

```powershell
python -m pip install -e .
python Osnova.py
```

The input dataset is expected at `data/borealtc/merged_interpolation.csv`. Reports, model artifacts, motion splits, and plots are written to `data/borealtc/`.

## Included results

The repository includes representative median `Ke` plots for ASPHALT, FLOORING, ICE, and SNOW, as well as aggregate binned-energy visualizations.