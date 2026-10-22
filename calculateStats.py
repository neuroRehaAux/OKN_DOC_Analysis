import atexit
import os
import sys
import textwrap
from datetime import datetime

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# settings (input file, output paths, display options) live in config.py,
# the logging/export infrastructure in reporting.py and the statistics in statsTools.py
from config import bootstrapSamples, bootstrapSeed, dataFile, logFile
from reporting import Tee, gitVersion, saveTable, saveText
from statsTools import (
    aucConfidenceInterval,
    bootstrapModelComparison,
    calcMetrics,
    descriptiveTable,
    fitLogit,
    likelihoodRatioTest,
    logitReport,
    metricsTable,
    oddsRatioFrame,
    oddsRatioTable,
)
from visualization import plotOddsRatioForest, plotRocCurve, saveFigure

"""
Important Notes:
"""
# When fitting a logistic regression model, statsmodels can drop rows with nan values.
# However, this is only checked in the exog array (i.e. independent variables).
# Thus, rows with nan values in the dependent variable are not dropped.
# Instead, the nan value is then treated as 0.
# In this script, arrays of dependent variables are separately checked for nans and found rows are dropped for the analysis.


"""
Helpers
"""
def descriptiveSection(df, variant, mapping, groupOrder, tableLabel, captionGroups, tests,
                       variables):
    # the demographics table for one definition of the diagnosis groups, written into
    # results/descriptive/<variant>/ as descriptive_demographics.csv/.md. The variants differ only
    # in how the admission diagnosis is pooled (mapping, a {value in crsr_diagnosis_admission:
    # group label} dict). captionGroups is the phrase naming the groups in the caption, e.g.
    # "two (coma/UWS, MCS)". tests is None for a table without group comparison, otherwise the
    # names of the continuous and the categorical comparison for the caption, because with two
    # groups the tests carry different names (see below).
    print('\n' * 2)
    print('#' * 150)
    print(f'DESCRIPTIVE STATISTICS - {variant}')
    print('#' * 150)
    print('\n')
    subFolder = os.path.join('descriptive', variant)
    # named 'diagnosis' rather than crsr_diagnosis_admission, because the values are the pooled
    # group labels of this variant and not the raw column
    groups = df['crsr_diagnosis_admission'].map(mapping).rename('diagnosis')

    # Etiology is taken from 'etiology' (traumatic, vascular, hypoxic, other) and not from
    # etiology_nontraumatic, which only separates traumatic from all other causes.
    # The test day is counted from the date of injury and is strongly skewed, which is why the
    # table reports medians.
    # With two groups the group comparisons of descriptiveTable become the tests that are
    # normally named for that case, which is why the caption names them that way: Kruskal-Wallis
    # on two samples is the Mann-Whitney U test (and is then reported as the exact one, see
    # descriptiveTable), and Fisher-Freeman-Halton on a 2 x 2 table is Fisher's exact test -
    # which covers sex, but not the 4 x 2 etiology table.
    caption = (f"{tableLabel}. Demographic characteristics of the {captionGroups} disorders of "
               f"consciousness groups (n = {len(df)}).")
    if tests is not None:
        continuousTest, categoricalTest = tests
        caption += (f" Continuous variables (age, days after injury on test day) were analyzed "
                    f"using {continuousTest} and categorical variables (sex, etiology) were "
                    f"assessed using {categoricalTest}.")
    caption += (" The Tukey method was used to represent interquartile range. Abbreviations: "
                "UWS = unresponsive wakefulness syndrome, MCS = minimally conscious state, "
                "IQR = interquartile range")
    caption += ", *is marked as significant (p < 0.05)." if tests is not None else "."

    print(f'Demographic characteristics by group (n = {len(df)})')
    table = descriptiveTable(df, groups, groupOrder, variables, compareGroups=tests is not None)
    print(table.to_string())
    print()
    print(textwrap.fill(caption, 150))
    saveTable(table, 'descriptive_demographics.csv', subFolder, caption=caption)


def contingencyTable(okn, y, negativeLabel, positiveLabel):
    # 2x2 table of OKN against the 0/1 class with row and column totals, as reported.
    # The class labels are passed in because they differ between the samples (e.g. coma/UWS vs
    # MCS in the primary sample, UWS vs MCS- in the hardest subsample).
    table = pd.crosstab(okn, y, margins=True, margins_name='total')
    table.index = ['-OKN', '+OKN', 'total']
    table.columns = [negativeLabel, positiveLabel, 'total']
    return table


def oddsRatioRow(unadjusted, y, eventLabel='MCS cases'):
    # odds ratio of OKN from the 2x2 table as one row of an odds ratio table. unadjusted is the
    # output of oddsRatioTable, eventLabel the name of the column counting the y = 1 cases.
    # The Wald interval is that of the OKN-only logistic regression (a single binary predictor),
    # the p value Fisher's exact one, as reported in the tables.
    return {'n': len(y),
            eventLabel: int(y.sum()),
            'OR': unadjusted['OR'],
            'CI low': unadjusted['low'],
            'CI high': unadjusted['high'],
            'p (Fisher)': unadjusted['pFisher']}


def diagnosticSubsample(label, dfSub, positiveLabels, fileStem, subFolder, oknCol='okn_positive',
                        withAuc=False):
    # the diagnostic analysis of OKN for one sample and one definition of +OKN, as run per
    # subsample in the sensitivity analysis and per OKN definition in the directional analysis.
    # The positive class is an admission diagnosis in positiveLabels, the test is the 0/1 column
    # oknCol. Writes results/<subFolder>/contingency/contingency_<fileStem>.csv, the 2x2 table
    # with the admission diagnoses of either side as column labels.
    # withAuc adds the AUC of OKN on its own to the metrics; for the binary test it is
    # (TPR + TNR)/2.
    # Returns (metrics of the 2x2 table, row of the odds ratio table).
    print('\n' * 2)
    print('#' * 75)
    print(f'Analysis: {label}')
    print('#' * 75)
    print('\n')
    y = dfSub['crsr_diagnosis_admission'].isin(positiveLabels).astype(int)

    # the class labels name the admission diagnoses that are actually in the sample on either side
    negativeLabels = [dx for dx in dxOrder
                      if dx not in positiveLabels
                      and dx in set(dfSub['crsr_diagnosis_admission'])]
    contTable = contingencyTable(dfSub[oknCol], y, '/'.join(negativeLabels),
                                 '/'.join(positiveLabels))
    print(contTable)
    saveTable(contTable, f'contingency_{fileStem}.csv', os.path.join(subFolder, 'contingency'))

    metrics = calcMetrics(dfSub, positiveLabels, oknCol=oknCol)
    if withAuc:
        metrics['AUC'] = roc_auc_score(y, dfSub[oknCol])
    unadjusted = oddsRatioTable(metrics)
    return metrics, oddsRatioRow(unadjusted, y)


def oddsRatioSummary(rows):
    # rows of odds ratios as a table. The odds ratios are rounded for reading, the p values are
    # not: the smallest of them is of the order 1e-07 and would round to 0.000, which reads as
    # an exact zero and is never true.
    table = pd.DataFrame(rows)
    columns = [column for column in table.columns if column.endswith(('OR', 'CI low', 'CI high'))]
    table[columns] = table[columns].round(3)
    return table


"""
Logging
"""
# everything printed from here on is written to the log file as well.
# atexit closes the file even if the script raises, so a failed run is still logged.
sys.stdout = Tee(logFile)
atexit.register(sys.stdout.close)

print('#' * 200)
print('OKN ANALYSIS - RESULTS')
print('#' * 200)
print(f"generated:  {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"data file:  {dataFile}")
print(f"git commit: {gitVersion()}")
print(f"log file:   {logFile}")

"""
Import data
"""
# the data set, available on request. The meaning and coding of every variable is in the
# codebook next to it (okn_doc_variables.csv, okn_doc_values.csv).
# All binary variables are 0/1 with 1 = the state named by the variable, e.g.
#   okn_positive       1 = +OKN
#   mcs_admission      1 = MCS at baseline, 0 = coma/UWS
#   gose_favourable    1 = GOS-E 3 or better
#   crsr_improved      1 = improvement of the CRS-R diagnosis (any step up the ranking, so also
#                      coma to UWS and MCS- to MCS+)
#   deceased_discharge 1 = died before discharge
if not os.path.isfile(dataFile):
    raise FileNotFoundError(f"{dataFile} not found - the data set is available on request, "
                            f"see the README")
df = pd.read_csv(dataFile)
# the prognosis section models survival rather than death, so that an odds ratio above 1 means a
# better outcome for +OKN, as it does for the other outcomes. The data set codes the death, so
# survival is derived here; deceased_discharge has no missing values.
df['survived_discharge'] = 1 - df['deceased_discharge']

"""
Descriptive statistics
"""
# Descriptive statistics
print('\n' * 3)
print('#' * 150)
print('DESCRIPTIVE STATISTICS')
print('#' * 150)
print('\n')

# the admission diagnoses as they are coded in the data, from least to most responsive. Used
# below to pool them into the groups of the descriptive variants and, further down, to label
# the sides of the 2x2 tables of the sensitivity subsamples.
dxOrder = ['coma', 'UWS', 'MCS-', 'MCS+']

# the demographics table is written once per definition of the diagnosis groups:
#   2classes  coma and UWS pooled against MCS, the contrast the diagnostic analysis is about,
#             with group comparison (Table 1)
#   4classes  the raw admission diagnoses, MCS split into MCS- and MCS+, without group
#             comparison (Table S1)
# Per variant: the pooling of crsr_diagnosis_admission, the column order, the table label, the
# phrase naming the groups in the caption, and the names of the two group comparisons in the
# caption (None = no comparison). The categorical comparison names both tests, because only sex
# is a 2 x 2 table: etiology has four categories, so its exact test is the Fisher-Freeman-Halton
# one even with two groups.
descriptiveVariants = {
    '2classes': ({'coma': 'Coma/UWS', 'UWS': 'Coma/UWS', 'MCS-': 'MCS', 'MCS+': 'MCS'},
                 ['Coma/UWS', 'MCS'],
                 'Table 1',
                 'two (coma/UWS, MCS)',
                 ('the exact Mann-Whitney U test',
                  "Fisher's exact test (Fisher-Freeman-Halton for the 4 × 2 etiology table)")),
    '4classes': ({'coma': 'Coma', 'UWS': 'UWS', 'MCS-': 'MCS-', 'MCS+': 'MCS+'},
                 ['Coma', 'UWS', 'MCS-', 'MCS+'],
                 'Table S1',
                 'four (coma, UWS, MCS-, MCS+)',
                 None),
}
# (column in the data, row label, categories); categories = None marks a continuous variable
variables_Descriptive = [
    ('age_years', 'Age', None),
    ('female', 'Sex', {1: 'Female', 0: 'Male'}),
    ('etiology', 'Etiology', {'traumatic': 'Traumatic', 'vascular': 'Vascular',
                              'hypoxic': 'Hypoxic', 'other': 'Other'}),
    ('days_injury_to_assessment', 'Days after injury on test day', None)]

for variant, (mapping, groupOrder, tableLabel, captionGroups,
              tests) in descriptiveVariants.items():
    descriptiveSection(df, variant, mapping, groupOrder, tableLabel, captionGroups, tests,
                       variables_Descriptive)


"""
Contingency table + metrics
"""
# Contingency table
print('\n' * 3)
print('#' * 150)
print('CONTINGENCY TABLE')
print('#' * 150)
print('\n')
# contingency table (Table 2)
# mcs_admission: 0 = coma/UWS, 1 = MCS, so the positive class is mcs_admission == 1
contTable = contingencyTable(df['okn_positive'], df['mcs_admission'], 'coma/UWS', 'MCS')
print(contTable)
saveTable(contTable, 'contingency_OKN_by_CRSgroup.csv', 'diagnostic')
print('\n')

# sensitivity (TPR), specificity (TNR), accuracy and the remaining rates
metrics_Diagnosis = calcMetrics(df, [1], dxCol='mcs_admission')
print('Diagnostic performance of OKN for MCS')
table_DiagnosisMetrics = metricsTable({'coma+UWS vs MCS': metrics_Diagnosis})
print(table_DiagnosisMetrics)
saveTable(table_DiagnosisMetrics, 'metrics_diagnostic.csv', 'diagnostic')
unadjusted_Diagnosis = oddsRatioTable(metrics_Diagnosis)

"""
Primary Analysis - Diagnosis
"""
# Primary Analysis - Diagnosis
print('\n' * 3)
print('#' * 150)
print('PRIMARY ANALYSIS - DIAGNOSIS')
print('#' * 150)
print('\n')
# binary logistic regression of the diagnosis (coma/UWS vs MCS) on OKN alone (Table S8). OKN
# is the only predictor, so its odds ratio and Wald interval are those of the 2x2 table above.
print('#' * 75)
print('Analysis: Binary Logistic Regression coma/UWS vs MCS')
print('#' * 75)
print('\n')
y_Diagnosis = df['mcs_admission'] # no missing values
X_Diagnosis = sm.add_constant(df[['okn_positive']])
logReg_Diagnosis, report_Diagnosis = logitReport(y_Diagnosis, X_Diagnosis)
print(report_Diagnosis, end='')
saveText(report_Diagnosis, 'logit_OKN_MCS.txt', 'diagnostic')

# ROC analysis of OKN (Fig. 3). The predicted probability of the OKN-only model is a monotone
# function of OKN, so curve and AUC are taken from OKN itself. OKN is a binary test, so the curve
# has a single operating point and the AUC reduces to (TPR + TNR)/2.
print()
auc_Diagnosis, aucLow_Diagnosis, aucHigh_Diagnosis = aucConfidenceInterval(y_Diagnosis,
                                                                           df['okn_positive'])
print(f'    AUC = {auc_Diagnosis:.3f} [95% CI = {aucLow_Diagnosis:.3f} - {aucHigh_Diagnosis:.3f}]')
figure_Diagnosis, _ = plotRocCurve(y_Diagnosis, df['okn_positive'])
saveFigure(figure_Diagnosis, 'roc_OKN_MCS', 'diagnostic')

print()
print('AUC of OKN for MCS')
table_DiagnosisAuc = pd.DataFrame([{'n': len(y_Diagnosis), 'MCS cases': int(y_Diagnosis.sum()),
                                    'AUC': auc_Diagnosis,
                                    'CI low': aucLow_Diagnosis,
                                    'CI high': aucHigh_Diagnosis}]).round(3)
print(table_DiagnosisAuc.to_string(index=False))
saveTable(table_DiagnosisAuc, 'auc_diagnostic.csv', 'diagnostic', index=False)

print()
print('Odds ratio of OKN for MCS')
table_DiagnosisOr = oddsRatioSummary([oddsRatioRow(unadjusted_Diagnosis, y_Diagnosis)])
print(table_DiagnosisOr.to_string(index=False))
saveTable(table_DiagnosisOr, 'or_diagnostic.csv', 'diagnostic', index=False)

"""
Sensitivity Analysis
"""
# Sensitivity Analysis
print('\n' * 3)
print('#' * 150)
print('SENSITIVITY ANALYSIS')
print('#' * 150)
print('\n')
# coma vs MCS is the easy contrast, UWS vs MCS- is the hard one (Tables S4 and S5).
# Two subsamples in addition to the primary analysis:
#   (a) coma patients excluded (UWS vs MCS)
#   (b) UWS vs MCS- only, i.e. the genuinely difficult contrast
# label: (subsample, admission diagnoses counted as positive, file name stem of the outputs).
# the stem is written out rather than derived from the label, because the labels contain
# spaces, brackets and a '-' and would not survive being turned into a path.
samples_Sensitivity = {
    'coma excluded (UWS vs MCS)': (df[df['crsr_diagnosis_admission'] != 'coma'], ['MCS-', 'MCS+'],
                                   'OKN_coma_excluded'),
    'UWS vs MCS- only': (df[df['crsr_diagnosis_admission'].isin(['UWS', 'MCS-'])], ['MCS-'],
                         'OKN_UWS_vs_MCSminus'),
}

metrics_Sensitivity = {} # 2x2 table measures per subsample, collected for the table after the loop
rows_SensitivityOr = [] # the same for the odds ratios
for label_Sensitivity, (df_Sensitivity, positiveLabels_Sensitivity,
                        fileStem_Sensitivity) in samples_Sensitivity.items():
    metrics_Sensitivity[label_Sensitivity], orRow_Sensitivity = diagnosticSubsample(
        label_Sensitivity, df_Sensitivity, positiveLabels_Sensitivity, fileStem_Sensitivity,
        'sensitivity')
    rows_SensitivityOr.append({'sample': label_Sensitivity, **orRow_Sensitivity})

print('\n')
print('Diagnostic performance of OKN per subsample')
table_Sensitivity = metricsTable(metrics_Sensitivity)
print(table_Sensitivity)
saveTable(table_Sensitivity, 'metrics_diagnostic_per_subsample.csv', 'sensitivity')

print('\n')
print('Odds ratio of OKN per subsample')
table_SensitivityOr = oddsRatioSummary(rows_SensitivityOr)
print(table_SensitivityOr.to_string(index=False))
saveTable(table_SensitivityOr, 'or_diagnostic_per_subsample.csv', 'sensitivity', index=False)

"""
Supplementary analysis: VEP
"""
# diagnosis
print('\n' * 3)
print('#' * 150)
print('VEP ANALYSIS')
print('#' * 150)
print('\n')
# The diagnostic analysis of the contingency and primary sections, restricted to the patients with
# visual evoked potentials present (vep_present == 1; Tables S6 and S7). VEP was examined in a
# subset of patients only; patients with VEP absent or not examined are excluded. The contrast is
# the primary one, coma/UWS vs MCS.
# In this subsample every MCS patient is +OKN, so the -OKN/MCS cell of the 2x2 table is zero and
# OKN separates the classes quasi-completely. The maximum likelihood estimate of the OKN
# coefficient is then infinite and no regression model can be fitted. What remains is everything
# the 2x2 table defines on its own.
df_VEP = df[df['vep_present'] == 1]
y_VEP = df_VEP['mcs_admission'] # 0 = coma/UWS, 1 = MCS

contTable_VEP = contingencyTable(df_VEP['okn_positive'], y_VEP, 'coma/UWS', 'MCS')
print(contTable_VEP)
saveTable(contTable_VEP, 'contingency_OKN_VEP.csv', 'VEPAnalysis')
print('\n')

# metrics and the AUC of OKN on its own. OKN is a fixed binary test and not fitted to the data,
# so the AUC has no optimism to correct.
metrics_VEP = calcMetrics(df_VEP, [1], dxCol='mcs_admission')
metrics_VEP['AUC'] = roc_auc_score(y_VEP, df_VEP['okn_positive'])
print('Diagnostic performance of OKN for MCS, VEP present')
table_VEPMetrics = metricsTable({'VEP present (coma+UWS vs MCS)': metrics_VEP})
print(table_VEPMetrics)
saveTable(table_VEPMetrics, 'metrics_diagnostic_VEP.csv', 'VEPAnalysis')
print()

# odds ratio from the 2x2 table only. With a zero cell the sample odds ratio is infinite;
# Table2x2 then adds 0.5 to every cell (Haldane-Anscombe correction), which gives a finite OR and
# Wald interval. Fisher's exact p needs no correction and is the p value to report.
zeroCell_VEP = min(metrics_VEP[cell] for cell in ('TP', 'FP', 'FN', 'TN')) == 0
if zeroCell_VEP:
    print('2x2 table has a zero cell: OR and Wald CI below are Haldane-Anscombe corrected '
          '(0.5 added to every cell)')
unadjusted_VEP = oddsRatioTable(metrics_VEP)

print()
print('Odds ratio of OKN for MCS, VEP present')
table_VEPOr = oddsRatioSummary([{**oddsRatioRow(unadjusted_VEP, y_VEP),
                                 'Haldane-Anscombe corrected': zeroCell_VEP}])
print(table_VEPOr.to_string(index=False))
saveTable(table_VEPOr, 'or_diagnostic_VEP.csv', 'VEPAnalysis', index=False)

"""
Supplementary analysis: horizontal/vertical OKN
"""
# diagnosis
print('\n' * 3)
print('#' * 150)
print('DIRECTIONAL ANALYSIS')
print('#' * 150)
print('\n')
# How does the diagnostic value of OKN depend on the direction in which it was elicited?
# (Tables S2 and S3)
# okn_direction records whether OKN was present horizontally, vertically, in both or in neither
# direction. +OKN is defined in four ways, each analysed on the primary sample (coma+UWS vs
# MCS) with the analysis of the sensitivity section:
#   any direction  horizontal or vertical OKN; -OKN only if absent in both directions. This is
#                  okn_positive and therefore reproduces the primary diagnosis section
#   horizontal     horizontal OKN, whatever the vertical result
#   vertical       vertical OKN, whatever the horizontal result
#   both           horizontal and vertical OKN; -OKN if absent in at least one direction, so
#                  stricter than the primary definition
# A patient with OKN only in the direction a definition ignores is -OKN under that definition.
df_Directional = df.assign(
    okn_horizontal=df['okn_direction'].isin(['both', 'horizontal_only']).astype(int),
    okn_vertical=df['okn_direction'].isin(['both', 'vertical_only']).astype(int),
    okn_both=df['okn_direction'].eq('both').astype(int))
# the first definition is only the primary analysis if okn_positive is exactly 'OKN in any
# direction', which is checked here because the whole comparison rests on it.
if not df['okn_positive'].eq(df['okn_direction'].ne('none').astype(int)).all():
    raise ValueError("okn_positive does not equal okn_direction != 'none' - the 'any direction' "
                     "definition would not reproduce the primary analysis")

# label: (0/1 column of the definition, file name stem of the outputs)
samples_Directional = {
    'any direction (horizontal or vertical)': ('okn_positive', 'OKN_any'),
    'horizontal': ('okn_horizontal', 'OKN_horizontal'),
    'vertical': ('okn_vertical', 'OKN_vertical'),
    'both directions (horizontal and vertical)': ('okn_both', 'OKN_both'),
}
positiveLabels_Directional = ['MCS-', 'MCS+'] # MCS vs coma/UWS, as in the primary sample

metrics_Directional = {} # 2x2 table measures and AUC per definition, collected for the table
rows_DirectionalOr = [] # the same for the odds ratios
for label_Directional, (oknCol_Directional, fileStem_Directional) in samples_Directional.items():
    metrics_Directional[label_Directional], orRow_Directional = diagnosticSubsample(
        f'OKN {label_Directional}', df_Directional, positiveLabels_Directional,
        fileStem_Directional, 'directionalAnalysis', oknCol=oknCol_Directional, withAuc=True)
    rows_DirectionalOr.append({'OKN definition': label_Directional, **orRow_Directional})

print('\n')
print('Diagnostic performance of OKN per OKN definition')
table_Directional = metricsTable(metrics_Directional)
print(table_Directional)
saveTable(table_Directional, 'metrics_diagnostic_per_definition.csv', 'directionalAnalysis')

print('\n')
print('Odds ratio of OKN per OKN definition')
table_DirectionalOr = oddsRatioSummary(rows_DirectionalOr)
print(table_DirectionalOr.to_string(index=False))
saveTable(table_DirectionalOr, 'or_diagnostic_per_definition.csv', 'directionalAnalysis',
          index=False)

"""
Secondary Analysis - Prognosis
"""
# Secondary Analysis - Prognosis
print('\n' * 3)
print('#' * 150)
print('SECONDARY ANALYSIS - PROGNOSIS')
print('#' * 150)
print('\n')

# covariates of all prognostic models. All of them are known at the baseline assessment.
covariates_Prognosis = ['age_years', 'days_injury_to_assessment', 'etiology_nontraumatic']
Xtrain_Outcome = sm.add_constant(df[['okn_positive'] + covariates_Prognosis])

# The four prognosis analyses differ only in the dependent variable, so they run in one loop,
# in the order of Table 3 and Fig. 4.
# outcome column -> (label of the forest plot, label of the summary tables, file name stem of the
#                    outputs, class label of outcome = 0, class label of outcome = 1), the class
#                    labels as in the codebook
outcomes_Prognosis = {
    'gose_favourable': ('GOS-E', 'GOS-E (favourable >= 3)', 'OKN_gose', 'GOS-E 1-2',
                        'GOS-E 3 or better'),
    'survived_discharge': ('Survival', 'survival to discharge', 'OKN_survival',
                           'died before discharge', 'alive at discharge'),
    'crsr_improved': ('CRS-R', 'CRS-R improvement', 'OKN_crsr_improved', 'no improvement',
                      'improvement'),
    'emcs_discharge': ('Improvement to eMCS', 'reaching eMCS', 'OKN_emcs', 'no eMCS', 'eMCS'),
}
logReg_Prognosis = {} # fitted models stay available for inspection after the run
metrics_Prognosis = {} # 2x2 table measures per outcome, collected for the table after the loop
rows_PrognosisOr = [] # the same for the adjusted odds ratios
forest_Prognosis = {} # the OKN rows of the models, keyed by the label of the forest plot

for outcomeCol_Prognosis, (forestLabel_Prognosis, summaryLabel_Prognosis, fileStem_Prognosis,
                           negativeLabel_Prognosis,
                           positiveLabel_Prognosis) in outcomes_Prognosis.items():
    print('\n' * 3)
    print('#' * 75)
    print(f'Analysis: Binary Logistic Regression Outcome {summaryLabel_Prognosis}')
    print('#' * 75)
    print('\n')
    ytrain_Prognosis = df[outcomeCol_Prognosis]
    inds = ytrain_Prognosis.isna().values # find nans
    ytrain_Prognosis = ytrain_Prognosis[~inds] # delete nan rows from dependent variable
    Xtrain_Prognosis = Xtrain_Outcome[~inds] # delete nan rows from independent variables
    # adjusted logistic regression (Tables S9-S12)
    model_Prognosis, report_Prognosis = logitReport(ytrain_Prognosis, Xtrain_Prognosis)
    print(report_Prognosis, end='')
    saveText(report_Prognosis, f'logit_{fileStem_Prognosis}.txt', 'prognosis')
    logReg_Prognosis[outcomeCol_Prognosis] = model_Prognosis

    # 2x2 table of OKN against the outcome and its measures (Table 3), on the same complete cases
    # as the model. They read OKN as a test for the outcome as it is coded (1 = event), i.e. for
    # survived_discharge they describe +OKN as a predictor of survival.
    contTable_Prognosis = contingencyTable(Xtrain_Prognosis['okn_positive'], ytrain_Prognosis,
                                           negativeLabel_Prognosis, positiveLabel_Prognosis)
    print()
    print(contTable_Prognosis)
    saveTable(contTable_Prognosis, f'contingency_{fileStem_Prognosis}.csv',
              os.path.join('prognosis', 'contingency'))
    # calcMetrics reads the class and the test off one frame, so outcome and OKN are handed to it
    # together
    metrics_Prognosis[summaryLabel_Prognosis] = calcMetrics(
        pd.DataFrame({'outcome': ytrain_Prognosis,
                      'okn_positive': Xtrain_Prognosis['okn_positive']}),
        [1], dxCol='outcome')

    # adjusted odds ratio of OKN: the OKN row of the model
    adjusted_Prognosis = oddsRatioFrame(model_Prognosis).loc['okn_positive']
    forest_Prognosis[forestLabel_Prognosis] = adjusted_Prognosis
    rows_PrognosisOr.append({'outcome': summaryLabel_Prognosis,
                             'n': int(model_Prognosis.nobs),
                             'events': int(ytrain_Prognosis.sum()),
                             'adjusted OR': float(adjusted_Prognosis['OR']),
                             'adjusted CI low': float(adjusted_Prognosis['low']),
                             'adjusted CI high': float(adjusted_Prognosis['high']),
                             'adjusted p (Wald)': float(adjusted_Prognosis['p'])})

print('\n')
# prevalence is the event rate of the outcome, not the share of MCS patients
print('Prognostic performance of OKN per outcome')
table_PrognosisMetrics = metricsTable(metrics_Prognosis)
print(table_PrognosisMetrics)
saveTable(table_PrognosisMetrics, 'metrics_prognostic_per_outcome.csv', 'prognosis')

print('\n')
print('Adjusted odds ratio of OKN per outcome')
table_PrognosisOr = oddsRatioSummary(rows_PrognosisOr)
print(table_PrognosisOr.to_string(index=False))
saveTable(table_PrognosisOr, 'or_prognostic_per_outcome.csv', 'prognosis', index=False)

# Forest plot of the adjusted OKN odds ratio across the outcomes (Fig. 4), one row per outcome in
# the order of the loop above. The axis is capped at 6; the GOS-E and the survival interval
# reach past it and are drawn with an arrow head instead of stretching the axis for all other
# rows.
table_PrognosisForest = pd.DataFrame(forest_Prognosis).T
print()
print('Adjusted OKN odds ratio per outcome (forest plot):')
print(table_PrognosisForest.round(3))
figure_Prognosis, _ = plotOddsRatioForest(table_PrognosisForest, xLimits=(0, 6),
                                          xTicks=(0, 2, 4, 6))
saveFigure(figure_Prognosis, 'forest_OKN_by_outcome', 'prognosis')

"""
Comparison OKN vs CRS-R for Prediction
"""
# Comparison OKN vs CRS-R for Prediction
print('\n' * 3)
print('#' * 150)
print('COMPARISON OKN VS CRS-R FOR PREDICTION')
print('#' * 150)
print('\n')
# How does OKN alone, and OKN in addition to the CRS-R diagnosis, compare to the CRS-R diagnosis
# alone in predicting outcome? (Table 4) Three nested models per outcome, all with the
# covariates of the prognosis section:
#   M1 = covariates + OKN, M2 = covariates + CRS-R diagnosis, M3 = covariates + OKN + CRS-R
# The incremental value of each marker is tested with likelihood ratio tests of M3 against M2
# (OKN added) and against M1 (CRS-R added); discrimination is the optimism corrected AUC.

# mcs_admission = CRS-R diagnostic category (0 = coma/UWS, 1 = MCS)
crsrCol = 'mcs_admission'
outcomes_Comparison = {'gose_favourable': 'GOS-E (favourable >= 3)',
                       'survived_discharge': 'survival to discharge',
                       'crsr_improved': 'CRS-R improvement',
                       'emcs_discharge': 'reaching eMCS'}

rows_Comparison = [] # collected for the summary table written at the end of this section

for outcomeCol, outcomeLabel in outcomes_Comparison.items():
    print('\n' * 2)
    print('#' * 75)
    print(f'Analysis: OKN vs CRS-R diagnosis - Outcome {outcomeLabel}')
    print('#' * 75)
    print('\n')

    # complete cases across outcome, both predictors and all covariates, so that the
    # three models are fitted on the exact same patients and the LR tests stay valid
    df_Comparison = df.dropna(subset=[outcomeCol, crsrCol, 'okn_positive'] + covariates_Prognosis)
    ytrain_Comparison = df_Comparison[outcomeCol] # all outcomes are coded 0/1
    Xtrain_M1 = sm.add_constant(df_Comparison[['okn_positive'] + covariates_Prognosis])
    Xtrain_M2 = sm.add_constant(df_Comparison[[crsrCol] + covariates_Prognosis])
    Xtrain_M3 = sm.add_constant(df_Comparison[['okn_positive', crsrCol] + covariates_Prognosis])
    logReg_M1 = fitLogit(ytrain_Comparison, Xtrain_M1, 'M1: covariates + OKN')
    logReg_M2 = fitLogit(ytrain_Comparison, Xtrain_M2, 'M2: covariates + CRS-R diagnosis')
    logReg_M3 = fitLogit(ytrain_Comparison, Xtrain_M3, 'M3: covariates + OKN + CRS-R diagnosis')

    print('\nNested model comparisons:')
    pOKNonCRSR = likelihoodRatioTest(logReg_M2, logReg_M3, 'OKN added to CRS-R diagnosis')
    pCRSRonOKN = likelihoodRatioTest(logReg_M1, logReg_M3, 'CRS-R diagnosis added to OKN')

    # optimism corrected AUCs (Harrell's bootstrap). The apparent AUCs favour the larger model,
    # so M3 could look best through overfitting.
    bootstrap_Comparison = bootstrapModelComparison(
        ytrain_Comparison,
        {'M1 (OKN)': Xtrain_M1, 'M2 (CRS-R)': Xtrain_M2, 'M3 (both)': Xtrain_M3},
        bootstrapSamples, bootstrapSeed, comparePair=('M1 (OKN)', 'M2 (CRS-R)'))
    optimism_Comparison = bootstrap_Comparison['models']
    correctedDiff_Comparison = bootstrap_Comparison['comparison']
    print(f'    optimism corrected AUCs ({bootstrapSamples} bootstrap resamples, '
          f'used {bootstrap_Comparison["nUsed"]}, discarded {bootstrap_Comparison["nFailed"]}):')
    for modelLabel, opt in optimism_Comparison.items():
        print(f'      {modelLabel:<12} apparent={opt["apparent"]:.3f}  optimism={opt["optimism"]:+.3f}  '
              f'corrected={opt["corrected"]:.3f}')
    print(f'      corrected difference (M1 - M2) = {correctedDiff_Comparison["correctedDiff"]:+.3f} '
          f'[{correctedDiff_Comparison["low"]:+.3f}, {correctedDiff_Comparison["high"]:+.3f}]')

    rows_Comparison.append({
        'Outcome': outcomeLabel,
        'Events/n': f"{int(ytrain_Comparison.sum())}/{int(logReg_M1.nobs)}",
        'AUC OKN': round(optimism_Comparison['M1 (OKN)']['corrected'], 3),
        'AUC CRS-R': round(optimism_Comparison['M2 (CRS-R)']['corrected'], 3),
        'AUC both': round(optimism_Comparison['M3 (both)']['corrected'], 3),
        'dAUC (OKN-CRS-R) [95% CI]': (f"{correctedDiff_Comparison['correctedDiff']:+.3f} "
                                      f"[{correctedDiff_Comparison['low']:+.3f}, "
                                      f"{correctedDiff_Comparison['high']:+.3f}]"),
        'p (OKN added)': round(pOKNonCRSR, 4),
        'p (CRS-R added)': round(pCRSRonOKN, 4),
        'bootstrap resamples discarded': bootstrap_Comparison['nFailed'],
    })

# Summary table (Table 4): discrimination on the optimism corrected scale, including the AUC
# difference and its percentile interval, next to the p values of the likelihood ratio tests
print('\n')
print('Summary: OKN alone vs CRS-R diagnosis alone vs both (optimism corrected AUCs)')
table_Comparison = pd.DataFrame(rows_Comparison)
print(table_Comparison.to_string(index=False))
saveTable(table_Comparison, 'comparison_summary.csv', 'comparison', index=False)
