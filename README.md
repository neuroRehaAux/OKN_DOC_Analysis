# OKN in Disorders of Consciousness

Statistical analysis for the study *Optokinetic Nystagmus as an Easy Bedside Test in Disorders of Consciousness – A Retrospective Study*.
It investigates optokinetic nystagmus (OKN) as a bedside marker of preserved consciousness in patients with disorders of consciousness (DoC).

Two clinical questions drive the analysis:

1. **Diagnosis**: can OKN status (+OKN / −OKN) distinguish coma/UWS from MCS?
2. **Prognosis**: does OKN status at the start of early neurological rehabilitation (ENR) predict outcome at discharge?

The sample is 78 subacute DoC patients assessed with a CRS-R and a routine OKN examination on the same day, with GOS-E and CRS-R collected again at discharge.

This repository contains exactly the analyses reported in the manuscript and its supplementary material.

## Running the analysis

No prior Python installation or programming experience is needed.
The analysis is run with [uv](https://docs.astral.sh/uv/), a small tool that installs the exact Python and package versions the published results were produced with (pinned in `pyproject.toml` and `uv.lock`).

1. **Get the code.**
   On <https://github.com/neuroRehaAux/OKN_DOC_Analysis>, click *Code → Download ZIP* and unzip the file.
   (Alternatively, with git installed: `git clone https://github.com/vifle/OKN.git`.)

2. **Install uv** (only once per computer).
   Open a terminal (Windows: *PowerShell* from the Start menu; macOS: *Terminal* from Applications → Utilities; Linux: your terminal application) and paste the line for your system:

   - Windows:

     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

   - macOS and Linux:

     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   Then close the terminal and open a new one, so that the `uv` command is found.
   Other ways to install uv are listed in its [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Add the data.**
   Inside the unzipped folder, next to `calculateStats.py`, create a folder named `data` and put the three data files into it (see [Data](#data)).

4. **Open a terminal in the unzipped folder.**

   - Windows 11: open the folder in File Explorer, right-click an empty area of the folder and choose *Open in Terminal*.
     (Windows 10: click into the address bar of File Explorer, type `powershell` and press Enter.)
   - macOS: open *Terminal*, type `cd` followed by a space, drag the folder from Finder into the Terminal window and press Enter.
   - Linux: right-click an empty area of the folder in the file manager and choose *Open in Terminal*, or `cd` into the folder.

5. **Run the analysis** by typing the following command into that terminal and pressing Enter:

   ```bash
   uv run calculateStats.py
   ```

   The first run needs an internet connection: uv downloads Python 3.12 and the pinned packages into a folder `.venv` inside the unzipped folder, without changing any other Python installation on the computer.
   The analysis itself takes about half a minute.
   A few lines `RuntimeWarning: overflow encountered in exp` / `divide by zero encountered in log` may appear in the terminal.
   They come from bootstrap resamples whose model fit does not converge; these resamples are discarded (see [Comparison OKN vs CRS-R for prediction](#comparison-okn-vs-crs-r-for-prediction-table-4)) and the warnings do not affect the results.

6. **Find the results** in the new folders `results/` and `figures/` (see [Output](#output)).
   `results/results.txt` contains everything the script printed.

If the code was downloaded as ZIP, the log records the git commit as *unknown (code is not in a git repository)*; this is expected.

## Data

The patient data are **not included** in this repository.
The anonymized data set the analysis reads, together with its codebook, will be made available on request from any qualified investigator (see the Data Availability statement of the manuscript).
It consists of three files:

| File | Content |
|---|---|
| `okn_doc_data.csv` | the data, one row per patient (78 patients, 21 variables) |
| `okn_doc_variables.csv` | codebook, one row per variable: label, description, type, level, unit, time point, valid range, number of missing values and what they mean, derivation, use in the analysis |
| `okn_doc_values.csv` | codebook, one row per variable-value pair |

Conventions of the data set:

- Variable names are English snake_case.
- Binary variables are coded 0/1, with 1 = the state named by the variable (e.g. `okn_positive` 1 = +OKN, `mcs_admission` 1 = MCS at baseline).
- Nominal and ordinal categories are stored as short English text labels (`coma`, `UWS`, `MCS-`, ...); in the value codebook the categories of ordinal variables are listed from lowest to highest.
- Missing values are empty fields. What an empty field means (not examined, not applicable, not documented) is given per variable in `missing_meaning`.
- Variables derived from other variables state their rule in `derivation`.
- Exact dates are not included; they are replaced by the intervals from the injury to the baseline assessment and to discharge, in days.

## Repository layout

| File | Purpose |
|---|---|
| `calculateStats.py` | the analysis, one section per part of the manuscript |
| `config.py` | input file, output paths, bootstrap settings, figure style |
| `reporting.py` | console logging (`Tee`), provenance (`gitVersion`), CSV and text export (`saveTable`, `saveText`) |
| `statsTools.py` | diagnostic metrics, odds ratios, logistic regression printouts, likelihood ratio test, AUC confidence interval, bootstrap optimism correction, exact Mann-Whitney U and Fisher-Freeman-Halton tests, descriptive table |
| `visualization.py` | ROC curve, forest plot and figure export |
| `pyproject.toml`, `uv.lock` | the pinned package versions, read by uv |
| `.python-version` | the Python version, read by uv |
| `LICENSE.md` | license of the code |
| `data/` | data set and codebook (not included, available on request) |
| `results/`, `figures/` | output of `calculateStats.py` (not included) |

## Analysis sections

Each section prints a banner into the log, so the log can be read alongside this list.
The sections and their output follow the manuscript (Table/Fig.) and the supplementary material (Table S).

### Descriptive statistics (Table 1, Table S1)

The demographics table: age, sex, etiology and days after injury on test day, as median (Tukey hinges) or n (%).

- **Table 1** compares coma/UWS against MCS with the exact Mann-Whitney U test for the continuous variables and Fisher's exact test for sex.
  Etiology has four categories, so its 4 × 2 table is tested with the Fisher-Freeman-Halton test.
  Both tests are enumerated exactly: the Mann-Whitney U test conditions on the observed ties (`exactRankSum` in `statsTools.py`), and the Fisher-Freeman-Halton test walks through every table with the observed margins (`fisherFreemanHalton`).
- **Table S1** splits the sample into coma, UWS, MCS− and MCS+, without group comparison.

### Contingency table and primary analysis: diagnosis (Table 2, Fig. 3, Table S8)

- The 2×2 table of OKN against the CRS-R diagnosis (coma/UWS vs MCS), with sensitivity, specificity, PPV, NPV and accuracy, MCS being the positive class.
- The odds ratio with its Wald 95% CI and Fisher's exact p.
- The binary logistic regression of the diagnosis on OKN as the only predictor (Table S8).
  With a single binary predictor its odds ratio and Wald interval are those of the 2×2 table.
- The ROC curve of OKN (Fig. 3) and its AUC with the DeLong 95% CI.

### Sensitivity analysis (Tables S4, S5)

The diagnostic analysis is repeated on two subsamples:

| Sample | n | Rationale |
|---|---|---|
| UWS vs MCS (coma excluded) | 73 | coma is the easiest group to separate |
| UWS vs MCS− only | 66 | the most conservative contrast |

Each sample gets its 2×2 table, the diagnostic metrics and the odds ratio with Wald CI and Fisher's exact p.

### VEP analysis (Tables S6, S7)

The diagnostic analysis restricted to the patients with visual evoked potentials present (n = 21).
All four MCS patients are +OKN, so the −OKN/MCS cell is zero and OKN separates the two classes quasi-completely; no logistic regression can be estimated.
The section reports what the 2×2 table defines on its own:

- the 2×2 table and the metrics, including the AUC of OKN.
- the odds ratio.
  The sample OR is infinite because of the zero cell, so `Table2x2` adds 0.5 to every cell (Haldane-Anscombe correction); the OR table flags whether the correction was applied.
  Fisher's exact p needs no correction and is the p value to report.

### Directional analysis (Tables S2, S3)

The diagnostic analysis with four definitions of +OKN, taken from the direction(s) in which OKN was elicited (`okn_direction`), all on the full sample (n = 78):

| Definition | +OKN if | −OKN if |
|---|---|---|
| any direction (primary) | horizontal or vertical OKN | neither |
| horizontal | horizontal OKN | no horizontal OKN (vertical ignored) |
| vertical | vertical OKN | no vertical OKN (horizontal ignored) |
| both directions | horizontal and vertical OKN | absent in at least one direction |

The first definition is `okn_positive` itself; the script checks that it equals `okn_direction != 'none'` before running.
Each definition gets its 2×2 table, the diagnostic metrics with the AUC of OKN, and the odds ratio with Wald CI and Fisher's exact p.

### Secondary analysis: prognosis (Table 3, Fig. 4, Tables S9–S12)

Binary logistic regression of four outcomes at discharge on OKN, adjusted for age, days after injury on test day and etiology (traumatic vs non-traumatic):

- favourable GOS-E (≥ 3)
- survival to discharge (`survived_discharge = 1 − deceased_discharge`, derived in `calculateStats.py`)
- improvement of the CRS-R diagnosis (any step up the ranking deceased < coma < UWS < MCS− < MCS+ < eMCS)
- reaching eMCS

All outcomes are coded so that an odds ratio above 1 means a better outcome for +OKN.
Per outcome the section writes the full regression output (Tables S9–S12) and the 2×2 table of OKN against the outcome with its metrics, where OKN is read as a test for the favourable outcome (Table 3).
It also collects the adjusted odds ratios with Wald CI and Wald p (Table 3) and draws them as a forest plot (Fig. 4).
The forest plot axis is capped at 6; intervals reaching past it end in an arrow head.

### Comparison OKN vs CRS-R for prediction (Table 4)

For each of the four outcomes, three logistic regression models are fitted on the same patients, each with the covariates of the prognosis section:

| Model | Predictors |
|---|---|
| M1 | covariates + OKN |
| M2 | covariates + CRS-R diagnosis at admission (MCS vs coma/UWS) |
| M3 | covariates + OKN + CRS-R diagnosis |

The incremental value of each marker is tested with likelihood ratio tests: M3 against M2 (OKN added to the CRS-R diagnosis) and M3 against M1 (CRS-R diagnosis added to OKN).
Discrimination is the AUC, corrected for optimism with Harrell's bootstrap (`bootstrapSamples` = 500 resamples, seeded by `bootstrapSeed` in `config.py`).
The ΔAUC (OKN − CRS-R) is the difference of the corrected AUCs, and its 95% CI is the percentile interval of the per-resample corrected differences.
That interval reflects the resampling variability of the correction and holds the apparent difference fixed, so it is not a complete accounting of uncertainty.

A resample is discarded when the refit of one of the models does not converge, and it is then discarded for all three models, so the corrected AUCs stay computed on the same resamples and remain comparable.
This affects 1–3 of the 500 resamples for most outcomes, but 52 for survival, so the optimism estimate for survival should be treated as approximate.
The number of discarded resamples is printed in the log and stored in `comparison_summary.csv`.

## Output

`results/results.txt` is overwritten on every run and holds the complete console output, prefixed with a provenance header (timestamp, data file, git commit, and whether the working tree had uncommitted changes).

Tables are also written as CSV, so they can be opened in Excel rather than retyped from the log:

```
results/
├── results.txt
├── descriptive/
│   ├── 2classes/   descriptive_demographics.csv/.md                       (Table 1)
│   └── 4classes/   descriptive_demographics.csv/.md                       (Table S1)
├── diagnostic/     contingency_OKN_by_CRSgroup.csv, metrics_diagnostic.csv,
│                   or_diagnostic.csv, auc_diagnostic.csv                  (Table 2)
│                   logit_OKN_MCS.txt                                      (Table S8)
├── sensitivity/    metrics_diagnostic_per_subsample.csv,
│   │               or_diagnostic_per_subsample.csv                        (Tables S4, S5)
│   └── contingency/ contingency_OKN_coma_excluded.csv, contingency_OKN_UWS_vs_MCSminus.csv
├── VEPAnalysis/    contingency_OKN_VEP.csv, metrics_diagnostic_VEP.csv,
│                   or_diagnostic_VEP.csv                                  (Tables S6, S7)
├── directionalAnalysis/ metrics_diagnostic_per_definition.csv,
│   │               or_diagnostic_per_definition.csv                       (Tables S2, S3)
│   └── contingency/ contingency_OKN_any.csv, contingency_OKN_horizontal.csv,
│                   contingency_OKN_vertical.csv, contingency_OKN_both.csv
├── prognosis/      metrics_prognostic_per_outcome.csv,
│   │               or_prognostic_per_outcome.csv                          (Table 3)
│   │               logit_OKN_gose.txt, logit_OKN_survival.txt,
│   │               logit_OKN_crsr_improved.txt, logit_OKN_emcs.txt        (Tables S9–S12)
│   └── contingency/ contingency_OKN_gose.csv, contingency_OKN_survival.csv,
│                   contingency_OKN_crsr_improved.csv, contingency_OKN_emcs.csv
└── comparison/     comparison_summary.csv                                 (Table 4)
```

Figures are written as PNG (to look at), PDF (vector, for print), SVG (vector, for Word) and TIFF (300 dpi, RGB, uncompressed, for journal submission):

```
figures/
├── diagnostic/  roc_OKN_MCS             (Fig. 3)
└── prognosis/   forest_OKN_by_outcome   (Fig. 4)
```

Their font and line colours are set in `config.py`: `figureFont`/`figureFontSize`, and `colourScheme`, which switches the figures between green (`'default'`) and black and grey (`'simple'`, used in the manuscript).
If the font (Times New Roman) is not installed, as is usual on Linux, the figures fall back to another serif font; the numbers are not affected.
Every figure is exactly as wide as the text block of the manuscript (`textWidth`, from the page width and margins in `config.py`), with the font and size of the body text, so inserted at 100 % the text in the figures has the same size as the text around them.

CSVs are comma-separated and written as UTF-8 with BOM, so special characters survive the trip into Excel.
If Excel shows everything in one column (common with a German or other locale that uses the comma as decimal separator), import the file via *Data → From Text/CSV* and choose comma as the delimiter.

## License

The code is released under the MIT License, see `LICENSE.md`.
