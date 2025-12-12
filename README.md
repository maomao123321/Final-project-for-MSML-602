# Data602 Medical Data Science Tutorial

This repository delivers a complete case study on forecasting ICU daily mortality, covering data preparation, feature engineering, machine learning, visualization, and the final narrative tutorial (`docs/index.md`).

## Dataset
- Raw observations live in `data/features_selected.csv`, one row per admission.
- Fields originate from the public MIMIC-IV database and span demographics, admission sources, discharge destinations, and length of stay.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate metrics and charts:
   ```bash
   python scripts/generate_analysis.py
   ```
   - Tabular artifacts and evaluation metrics will appear in `artifacts/`.
   - The visualization is written to `docs/assets/mortality_forecast.png`.
3. Review the finished tutorial by opening `docs/index.md` or previewing it in your IDE.

## Repository Layout
```
data/            # Source CSV data
docs/            # GitHub Pages publication root
  └─ assets/     # Images embedded in the tutorial
scripts/         # Analysis scripts
artifacts/       # Generated tables and metrics
README.md        # Project overview
requirements.txt # Python dependencies
```

## Publish to GitHub Pages
1. Push the repository to GitHub.
2. Under “Settings → Pages,” set the source to the `docs/` folder on the `main` branch.
3. After the build completes, access the site via `https://<username>.github.io/<repo>/`.

## License
Provided for course-work demonstration only. Do not redistribute sensitive or regulated data.

