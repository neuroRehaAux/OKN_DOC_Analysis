"""
Statistical helpers: diagnostic performance measures, logistic regression printouts and
model comparison tests. The functions print their results and return the underlying object,
so they can be used both for the log and for further calculations.
"""
import contextlib
import io
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import gammaln
from sklearn.metrics import confusion_matrix, roc_auc_score
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def calcMetrics(dfSub, positiveLabels, dxCol='crsr_diagnosis_admission', oknCol='okn_positive'):
    # diagnostic performance of OKN for an arbitrary subsample / class definition.
    # positive class = patient labelled with one of positiveLabels in dxCol, positive test = +OKN.
    # TPR = sensitivity = recall, TNR = specificity, PPV = precision.
    yTrue = dfSub[dxCol].isin(positiveLabels).astype(int)
    yPred = dfSub[oknCol].astype(int)
    TN, FP, FN, TP = (int(cell) for cell in confusion_matrix(yTrue, yPred, labels=[0,1]).ravel())
    n = TN + FP + FN + TP
    TPR = TP/(TP+FN) if TP+FN else np.nan
    TNR = TN/(TN+FP) if TN+FP else np.nan
    return {'n': n, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
            'prevalence': (TP+FN)/n,
            'TPR': TPR, 'TNR': TNR, 'FPR': 1-TNR, 'FNR': 1-TPR,
            'PPV': TP/(TP+FP) if TP+FP else np.nan,
            'NPV': TN/(TN+FN) if TN+FN else np.nan,
            'F1': 2*TP/(2*TP+FP+FN) if TP else np.nan,
            'ACC': (TP+TN)/n}


def metricsTable(metricsByLabel):
    # dict of {label: calcMetrics output} as one readable table
    table = pd.DataFrame(metricsByLabel).T
    countCols = ['n', 'TN', 'FP', 'FN', 'TP']
    table[countCols] = table[countCols].astype(int)
    return table.round(3)


def oddsRatioTable(metrics):
    # unadjusted association between OKN and the positive class.
    # Table2x2 rows = test result (+OKN / -OKN), columns = true class (positive / negative).
    contTable = [[metrics['TP'], metrics['FP']], [metrics['FN'], metrics['TN']]]
    table = Table2x2(contTable)
    lower, upper = table.oddsratio_confint()
    # Fisher's exact p is reported next to the Wald p because of the small cell counts
    _, pFisher = stats.fisher_exact(contTable, alternative='two-sided')
    print(f"unadjusted OR = {table.oddsratio:.3f} [{lower:.3f} - {upper:.3f}], "
          f"Wald p = {table.oddsratio_pvalue():.3g}, Fisher exact p = {pFisher:.3g}")
    # returned as a dict rather than as the Table2x2, because Fisher's p is computed here and
    # is not a property of that object - a caller collecting these into a table needs both
    return {'OR': table.oddsratio, 'low': lower, 'high': upper,
            'pWald': table.oddsratio_pvalue(), 'pFisher': pFisher}


def fitLogit(y, X, label):
    # binary logistic regression with printout of odds ratios and confidence intervals
    model = sm.Logit(y, X, missing='drop').fit(disp=0)
    print(f"--- {label} (n={int(model.nobs)}, LLF={model.llf:.3f}, AIC={model.aic:.2f})")
    confInt = model.conf_int()
    for term in model.params.index:
        if term == 'const':
            continue
        print(f"    {term:<22} OR={np.exp(model.params[term]):8.3f} "
              f"[{np.exp(confInt.loc[term,0]):7.3f} - {np.exp(confInt.loc[term,1]):8.3f}]  p={model.pvalues[term]:.4f}")
    return model


def logitReport(y, X):
    # binary logistic regression with its full output (optimizer message, summary, odds ratios)
    # captured into a string. The primary and prognosis sections print this string to the log
    # and write it to text files, so that log and files cannot drift apart in format. redirect_stdout puts the previous sys.stdout (the Tee of the
    # log) back afterwards.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = sm.Logit(y, X, missing='drop').fit()
        print(model.summary())
        print('Odds Ratio:')
        oddsRatio = model.conf_int()
        # conf_int() defaults to alpha=0.05, so its two columns are the 2.5% and 97.5% bounds
        # of a 95% interval, not the 5% and 95% ones the widespread snippet labels them with.
        oddsRatio.columns = ['2.5%', '97.5%']
        oddsRatio['Odds Ratio'] = model.params.values
        print(np.exp(oddsRatio))
    return model, buffer.getvalue()


def likelihoodRatioTest(modelSmall, modelBig, label):
    # nested model comparison: does the extra term improve model fit?
    chi2 = 2*(modelBig.llf - modelSmall.llf)
    dof = int(modelBig.df_model - modelSmall.df_model)
    p = stats.chi2.sf(chi2, dof)
    print(f"    LR test {label:<32} chi2({dof})={chi2:6.3f}, p={p:.4f}, dAIC={modelBig.aic-modelSmall.aic:+.2f}")
    return p


def aucConfidenceInterval(y, scores, alpha=0.05):
    # DeLong confidence interval for a single AUC, from the structural components of the
    # DeLong method.
    # The interval is a normal approximation and therefore symmetric around the AUC, so it can
    # reach past 0 or 1 at small samples or extreme AUCs; it is clipped to the valid range.
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    scoresPos, scoresNeg = scores[y==1], scores[y==0]
    numPos, numNeg = len(scoresPos), len(scoresNeg)
    auc = roc_auc_score(y, scores)
    # the variance needs at least two cases per class, otherwise ddof=1 divides by zero
    if numPos < 2 or numNeg < 2:
        return auc, np.nan, np.nan
    v10 = np.array([np.mean((scoresNeg < s) + 0.5*(scoresNeg == s)) for s in scoresPos])
    v01 = np.array([np.mean((scoresPos > s) + 0.5*(scoresPos == s)) for s in scoresNeg])
    halfWidth = stats.norm.ppf(1 - alpha/2)*np.sqrt(v10.var(ddof=1)/numPos + v01.var(ddof=1)/numNeg)
    return auc, max(0.0, auc - halfWidth), min(1.0, auc + halfWidth)


def oddsRatioFrame(model, dropConstant=True):
    # odds ratios with confidence interval and p value of a fitted logistic regression, in the
    # shape the forest plot expects (one row per predictor, exponentiated coefficients).
    # The constant is dropped by default: its odds ratio is the baseline odds, not an effect,
    # and its confidence interval would stretch the axis of the plot for nothing.
    bounds = model.conf_int()
    frame = pd.DataFrame({'OR': np.exp(model.params),
                          'low': np.exp(bounds[0]),
                          'high': np.exp(bounds[1]),
                          'p': model.pvalues})
    return frame.drop(index='const', errors='ignore') if dropConstant else frame


def bootstrapModelComparison(y, designs, nBoot=500, seed=0, comparePair=None):
    # Harrell's bootstrap optimism correction of the AUC, for several models at once.
    # The in-sample (apparent) AUC is optimistically biased and the bias grows with the
    # number of predictors, which systematically favours the larger model in a comparison -
    # the combined model can look best purely through overfitting.
    # Procedure: refit on each bootstrap resample, take the drop in AUC between the resample
    # and the original data as the optimism of that resample, and subtract the average.
    #
    # designs: dict of {label: design matrix}. All models share the same resamples, and a
    # resample is discarded for all of them as soon as one fails, so the corrected AUCs stay
    # mutually comparable and their difference equals the mean per-resample difference.
    #
    # comparePair: (labelA, labelB). Adds the corrected AUC difference together with a
    # percentile interval taken from the per-resample corrected differences. That keeps the
    # difference on the same (corrected) scale as the AUCs, rather than mixing a corrected
    # AUC with an apparent-scale DeLong interval. Note the interval reflects the resampling
    # variability of the correction; it is not a full accounting of the uncertainty in the
    # apparent difference.
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    designs = {label: np.asarray(X, dtype=float) for label, X in designs.items()}
    apparent = {label: roc_auc_score(y, sm.Logit(y, X).fit(disp=0).predict(X))
                for label, X in designs.items()}
    gaps = {label: [] for label in designs}
    nFailed = 0
    # ConvergenceWarning is expected here: some resamples run into near-separation and are
    # discarded below. It is silenced for the resampling loop only, so that a convergence
    # problem in one of the main model fits elsewhere in the analysis stays visible.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for _ in range(nBoot):
            inds = rng.integers(0, len(y), len(y))
            yBoot = y[inds]
            if len(np.unique(yBoot)) < 2: # resample without both classes carries no information
                nFailed += 1
                continue
            gapsBoot = {}
            for label, X in designs.items():
                try:
                    modelBoot = sm.Logit(yBoot, X[inds]).fit(disp=0)
                    # non converged fits (near separation) would distort the optimism estimate
                    if not modelBoot.mle_retvals['converged']:
                        break
                    gapsBoot[label] = (roc_auc_score(yBoot, modelBoot.predict(X[inds]))
                                       - roc_auc_score(y, modelBoot.predict(X)))
                except Exception:
                    break
            if len(gapsBoot) < len(designs): # keep the resample only if every model fitted
                nFailed += 1
                continue
            for label, gap in gapsBoot.items():
                gaps[label].append(gap)

    models = {}
    for label in designs:
        optimism = float(np.mean(gaps[label])) if gaps[label] else np.nan
        models[label] = {'apparent': apparent[label], 'optimism': optimism,
                         'corrected': apparent[label] - optimism}
    result = {'models': models, 'nUsed': len(next(iter(gaps.values()))), 'nFailed': nFailed}

    if comparePair is not None:
        labelA, labelB = comparePair
        apparentDiff = apparent[labelA] - apparent[labelB]
        # corrected difference per resample; its mean is corrected[A] - corrected[B]
        correctedDiffs = apparentDiff - (np.array(gaps[labelA]) - np.array(gaps[labelB]))
        result['comparison'] = {
            'correctedDiff': models[labelA]['corrected'] - models[labelB]['corrected'],
            'low': float(np.percentile(correctedDiffs, 2.5)),
            'high': float(np.percentile(correctedDiffs, 97.5))}
    return result


def tukeyHinges(values):
    # quartiles the way Tukey defines them (SPSS: "Tukey's hinges"): the lower hinge is the
    # median of the lower half of the sorted values, the upper hinge the median of the upper
    # half, and with an odd number of values the median belongs to both halves.
    # This is not what numpy and pandas compute by default (linear interpolation between the
    # two neighbouring order statistics); the two disagree by a few units on samples of this
    # size, so the method is stated in the table caption and fixed here.
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return np.nan, np.nan
    half = (len(values) + 1)//2 # the median belongs to both halves when the count is odd
    return float(np.median(values[:half])), float(np.median(values[len(values)-half:]))


def countTablesWithMargins(rowSums, colSums):
    # number of tables with the given margins, used to decide whether the exact enumeration in
    # fisherFreemanHalton is affordable. Counting stays cheap even where enumerating would not
    # be, because all partial tables that leave the same column margins have the same number of
    # completions and the recursion is memoised on exactly that state.
    @lru_cache(maxsize=None)
    def walk(rowIndex, remaining):
        if rowIndex == len(rowSums) - 1:
            return 1 # the last row is fixed by what the columns have left
        total = 0
        def cells(colIndex, left, filled):
            nonlocal total
            if colIndex == len(colSums) - 1:
                if left <= filled[colIndex]:
                    total += walk(rowIndex + 1, filled[:colIndex] + (filled[colIndex] - left,))
                return
            for cell in range(min(left, filled[colIndex]) + 1):
                cells(colIndex + 1, left - cell,
                      filled[:colIndex] + (filled[colIndex] - cell,) + filled[colIndex+1:])
        cells(0, int(rowSums[rowIndex]), remaining)
        return total

    return walk(0, tuple(int(count) for count in colSums))


def fisherFreemanHalton(counts, maxTables=2_000_000, monteCarloSamples=100_000, seed=0):
    # Fisher-Freeman-Halton test: Fisher's exact test generalised to an r x c table.
    # Both margins are held fixed, the probability of a table under independence is the
    # multiple hypergeometric one, and the two sided p value is the total probability of all
    # tables that are at least as unlikely as the observed one. The exact test is used instead
    # of chi square because several cells of the demographics table have expected counts below
    # five, where the chi square approximation is not reliable.
    # scipy's fisher_exact does accept r x c tables from version 1.15 on, but answers them by
    # Monte Carlo sampling and therefore returns a slightly different p value on every call.
    # The enumeration below is deterministic, which matters for a number that goes into a table.
    counts = np.asarray(counts, dtype=int)
    rowSums, colSums = counts.sum(axis=1), counts.sum(axis=0)
    n = int(counts.sum())
    logFactorial = gammaln(np.arange(n + 1) + 1)
    # the margin terms are the same for every table with these margins, so only the cell terms
    # have to be recomputed while walking through the tables
    constant = logFactorial[rowSums].sum() + logFactorial[colSums].sum() - logFactorial[n]
    observed = constant - logFactorial[counts].sum()
    # a table whose probability ties with the observed one counts as at least as extreme; the
    # tolerance is on the log scale, i.e. a relative tolerance on the probability itself
    tolerance = 1e-7

    if countTablesWithMargins(rowSums, colSums) > maxTables:
        # far more tables than can be walked through in reasonable time. This does not happen
        # for the tables of this analysis, but the fallback keeps the function usable for
        # larger ones: the p value is then estimated from random tables with the observed
        # margins. The generator is seeded, so repeated runs still give the same number.
        samples = stats.random_table(rowSums, colSums).rvs(
            monteCarloSamples, random_state=np.random.default_rng(seed))
        logProbabilities = constant - logFactorial[samples].sum(axis=(1, 2))
        # (hits + 1)/(samples + 1) rather than hits/samples: the observed table is one of the
        # possible draws, and the plain proportion can return an impossible p value of zero
        pValue = (1 + int((logProbabilities <= observed + tolerance).sum()))/(monteCarloSamples + 1)
        print(f"note: Fisher-Freeman-Halton estimated from {monteCarloSamples} random tables "
              f"(exact enumeration would exceed {maxTables} tables)")
        return pValue

    def walkRows(rowIndex, remaining, logCells):
        # one row at a time; the last row is whatever the column margins still demand
        if rowIndex == len(rowSums) - 1:
            logProbability = constant - logCells - logFactorial[list(remaining)].sum()
            return float(np.exp(logProbability)) if logProbability <= observed + tolerance else 0.0
        return walkCells(rowIndex, 0, int(rowSums[rowIndex]), remaining, logCells)

    def walkCells(rowIndex, colIndex, left, remaining, logCells):
        # distribute the row total over the columns, never taking more than a column has left
        if colIndex == len(colSums) - 1:
            if left > remaining[colIndex]:
                return 0.0
            return walkRows(rowIndex + 1, remaining[:colIndex] + (remaining[colIndex] - left,),
                            logCells + logFactorial[left])
        probability = 0.0
        for cell in range(min(left, remaining[colIndex]) + 1):
            probability += walkCells(
                rowIndex, colIndex + 1, left - cell,
                remaining[:colIndex] + (remaining[colIndex] - cell,) + remaining[colIndex+1:],
                logCells + logFactorial[cell])
        return probability

    return min(walkRows(0, tuple(int(count) for count in colSums), 0.0), 1.0)


def formatPValue(pValue, digits=3, alpha=0.05):
    # p values as they appear in the reported table: a fixed number of decimals, marked with
    # an asterisk when significant and reported as "< 0.001" instead of rounding down to 0.000
    if not np.isfinite(pValue):
        return 'n/a'
    marker = '*' if pValue < alpha else ''
    smallest = 10**(-digits)
    if pValue < smallest:
        return f"< {smallest:.{digits}f}{marker}"
    return f"{pValue:.{digits}f}{marker}"


def exactRankSum(a, b):
    # exact two sided Mann-Whitney U test (Wilcoxon rank sum test) for two samples.
    #
    # The p value is the conditional one: the observed values are taken as given and the null
    # distribution is the rank sum over all C(n+m, n) ways of splitting them into the two groups.
    # That is what makes the test valid with ties, which matters here because the variables of
    # the demographics table are whole years and whole days and tie heavily (30 of 78 age values
    # are shared). scipy's method='exact' assumes untied data - it enumerates the tie free null
    # distribution and its documentation says so - while method='asymptotic' corrects for the
    # ties but is a normal approximation. This function conditions on the observed ties and
    # enumerates, so it is exact for the data as they are.
    #
    # The distribution is built by the shift algorithm (Streitberg and Roehmel 1986), the same
    # approach R's coin package uses for its exact rank tests: the values are added one at a
    # time, and counts[k, s] (the number of subsets of size k with doubled rank sum s) is
    # updated by shifting the subsets that include the new value. Ranks are midranks and
    # therefore multiples of 0.5, so they are doubled to keep the sum on an integer grid.
    # C(78, 29) is about 1.6e22 subsets, far too many to list, but the counts are only ~6000
    # numbers per subset size and the recursion fills them in a fraction of a second.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    values = np.concatenate([a, b])
    n, total = len(a), len(values)
    doubledRanks = np.rint(2*stats.rankdata(values)).astype(int)
    observed = int(doubledRanks[:n].sum())

    # counts are held as floats: the exact integers exceed what a float64 represents exactly,
    # but the p value is a ratio of these numbers and needs relative, not absolute, precision
    counts = np.zeros((n + 1, int(doubledRanks.sum()) + 1))
    counts[0, 0] = 1.0
    for index, rank in enumerate(doubledRanks):
        # k downwards, so that each value enters a subset at most once. A subset of size k can
        # only be formed once k values have been seen, and only sizes up to n are of interest.
        for k in range(min(n, index + 1), 0, -1):
            counts[k, rank:] += counts[k - 1, :counts.shape[1] - rank]
    distribution = counts[n]/counts[n].sum()

    # the rank multiset is symmetric about (total + 1)/2 and so is the null distribution of the
    # rank sum, which is why the two sided p value is the smaller tail doubled (the convention
    # of R's wilcox.test). Both tails include the observed sum itself.
    lowerTail = float(distribution[:observed + 1].sum())
    upperTail = float(distribution[observed:].sum())
    return min(1.0, 2*min(lowerTail, upperTail))


def descriptiveTable(df, groups, groupOrder, variables, totalLabel='Total sample',
                     valueDigits=2, pDigits=3, compareGroups=True):
    # demographics table: one column per group plus the total sample, one row
    # per variable (categorical variables followed by one row per category) and, with
    # compareGroups, the group comparison in the last column.
    #
    # groups     : series aligned to df holding the group each patient belongs to
    # groupOrder : group labels in the order in which they become columns
    # variables  : list of (column, row label, categories) in the order of the table rows.
    #              categories = None marks a continuous variable, otherwise it is a
    #              {value in the data: label in the table} mapping that also fixes the order of
    #              the category rows.
    #
    # Continuous variables are summarised as median with the interquartile range and compared
    # with the Kruskal-Wallis test (small groups and strongly skewed day counts, so no normal
    # distribution is assumed); categorical variables are compared with the Fisher-Freeman-
    # Halton exact test because of the small expected cell counts.
    #
    # Percentages: the row of a categorical variable shows how the sample is split across the
    # groups (percentage of the total sample), the category rows below it show how that
    # category is split across the groups (percentage of the category).
    groups = pd.Series(groups, index=df.index)
    columns = [totalLabel] + list(groupOrder)
    masks = {totalLabel: pd.Series(True, index=df.index)}
    masks.update({groupLabel: (groups == groupLabel) for groupLabel in groupOrder})

    def countCell(count, reference):
        return f"{count} ({100*count/reference:.1f})" if reference else f"{count}"

    rows = {}
    for column, label, categories in variables:
        values = df[column]
        missing = int(values.isna().sum())
        if missing:
            print(f"note: {label} ({column}) has {missing} missing value(s); they are excluded "
                  f"from the summary and from the group comparison")

        if categories is None:
            cells = {}
            for columnLabel in columns:
                sample = values[masks[columnLabel]].dropna()
                if len(sample) == 0:
                    cells[columnLabel] = 'n/a'
                    continue
                lowerHinge, upperHinge = tukeyHinges(sample)
                cells[columnLabel] = (f"{np.median(sample):.{valueDigits}f} "
                                      f"({lowerHinge:.{valueDigits}f}-{upperHinge:.{valueDigits}f})")
            if not compareGroups:
                rows[label] = cells
                continue
            samples = [values[masks[groupLabel]].dropna() for groupLabel in groupOrder]
            samples = [sample for sample in samples if len(sample) > 0]
            # kruskal raises when all values are identical: that is not a broken data set but a
            # comparison without any variance to test, so it is reported as not available
            try:
                if len(samples) == 2:
                    # two groups: the Kruskal-Wallis test is the Mann-Whitney U test, and with
                    # only two groups the exact version of it is affordable, so the table reports
                    # the exact p value rather than the chi square approximation of H
                    pValue = exactRankSum(*samples)
                else:
                    pValue = stats.kruskal(*samples).pvalue
                cells['p-value'] = formatPValue(pValue, pDigits)
            except ValueError:
                cells['p-value'] = 'n/a'
            rows[label] = cells
            continue

        # categorical variable: categories against groups, in the order given by the caller
        table = pd.crosstab(values, groups).reindex(index=list(categories),
                                                    columns=list(groupOrder))
        table = table.fillna(0).astype(int)
        total = int(table.values.sum())
        header = {totalLabel: f"{total}"}
        header.update({groupLabel: countCell(int(table[groupLabel].sum()), total)
                       for groupLabel in groupOrder})
        if compareGroups:
            header['p-value'] = formatPValue(fisherFreemanHalton(table.values), pDigits)
        rows[label] = header
        for value, categoryLabel in categories.items():
            categoryTotal = int(table.loc[value].sum())
            cells = {totalLabel: f"{categoryTotal}"}
            cells.update({groupLabel: countCell(int(table.loc[value, groupLabel]), categoryTotal)
                          for groupLabel in groupOrder})
            if compareGroups:
                cells['p-value'] = '' # the test belongs to the variable, not to its single categories
            # indented, so that the category is readable as belonging to the row above it
            rows[f"  {categoryLabel}"] = cells

    table = pd.DataFrame(rows).T.reindex(columns=columns + (['p-value'] if compareGroups else []))
    table.index.name = 'Value n (%) or median (IQR 25-75)'
    return table
