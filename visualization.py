"""
Figures for the OKN analysis: house style, ROC curves and figure export.
The plotting functions return (figure, axes) so that a figure can still be adjusted before it
is written to disk with saveFigure. Nothing here computes statistics - the AUC and its
confidence interval come from statsTools, so that log and figure cannot disagree.
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from config import figuresPath, figureFont, figureFontSize, figureColours, textWidth
from statsTools import aucConfidenceInterval

# line colours of all figures. Which set is used is decided in config.colourScheme; the names
# below say what each colour is for, the values live in config.py so that the whole figure set
# can be switched from green to greyscale in one place.
rocLine = figureColours['rocLine'] # ROC curve and the confidence interval bars of the forest plot
rocPoint = figureColours['rocPoint'] # ROC points and odds ratio markers
referenceColour = figureColours['referenceColour'] # diagonal of the random classifier
gridColour = figureColours['gridColour']

# a ROC curve of a binary test has three points and they carry the information (sensitivity and
# specificity of the test), so they are marked. A model prediction has as many points as there
# are patients, where markers would only clutter the line - hence up to this count only.
maxMarkedPoints = 12


def applyPlotStyle():
    # serif type and restrained grid lines, to match the typesetting of the write-up.
    # Called by the plotting functions, so a figure looks the same no matter which script
    # produced it. Font and base size come from config.py.
    plt.rcParams.update({
        'font.family': 'serif',
        # the configured font first, then fallbacks for machines that do not have it
        # (matplotlib silently walks down the list)
        'font.serif': [figureFont, 'DejaVu Serif', 'Liberation Serif'],
        'mathtext.fontset': 'stix',
        # the other sizes follow the configured base size, so that one number in config.py
        # scales the whole figure instead of leaving the labels behind. The figures are inserted
        # unscaled, so the base size is the body text size: labels and titles use it as is,
        # tick labels and legend are one point smaller to rank below them.
        'font.size': figureFontSize,
        'axes.labelsize': figureFontSize,
        'axes.titlesize': figureFontSize,
        'figure.titlesize': figureFontSize,
        'legend.fontsize': figureFontSize - 1,
        'xtick.labelsize': figureFontSize - 1,
        'ytick.labelsize': figureFontSize - 1,
        'axes.linewidth': 0.8,
        # margin around the constrained layout. The default of 3 pt lets the descenders of the
        # x axis label touch the edge of the saved figure.
        'figure.constrained_layout.h_pad': 6/72,
        'figure.constrained_layout.w_pad': 6/72,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def saveFigure(figure, fileName, subFolder, formats=('png', 'pdf', 'svg', 'tiff'), dpi=300,
               close=False):
    # counterpart of saveTable in reporting.py, but writing into figures/ instead of results/,
    # with the same subfolder logic and the same [saved] line in the log.
    # fileName is given without extension, because the figure is written once per format:
    # png to look at and to paste into a slide, pdf as vector graphic for print, svg as vector
    # graphic for Word (which cannot insert pdf), tiff for journal submission.
    # The figure is saved at exactly its figure size - no bbox_inches='tight', which would trim
    # or pad the edges and change the width. That is what lets Word insert it at 100 % (the png
    # carries its dpi, the svg its size in points), so the text keeps the configured size.
    # close=True closes the figure once it is written, for figures that are not inspected
    # after the run.
    folder = os.path.join(figuresPath, subFolder)
    os.makedirs(folder, exist_ok=True)
    for fileFormat in formats:
        path = os.path.join(folder, f"{fileName}.{fileFormat}")
        if fileFormat == 'tiff':
            # matplotlib writes tiff as RGBA, and some journal portals reject the alpha channel.
            # The figure is therefore rendered at the same dpi, flattened to RGB (the background
            # is opaque white, so nothing visible changes) and written with its dpi.
            # The tiff is left uncompressed: compressed tiff is written through libtiff, which
            # crashes the interpreter in some conda environments (Pillow/libtiff mismatch).
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=dpi)
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.convert('RGB').save(path, format='TIFF', dpi=(dpi, dpi))
        else:
            figure.savefig(path, format=fileFormat, dpi=dpi)
        print(f"[saved] {path}")
    if close:
        plt.close(figure)
    return figure


def decorateRocAxes(ax):
    # the parts that every ROC figure shares: diagonal, limits, labels and grid.
    # the diagonal is dashed and black rather than coloured, so that it reads as a reference
    # line and not as a third result.
    ax.plot([0, 1], [0, 1], linestyle='--', color=referenceColour, linewidth=1.0,
            zorder=2, label='Random classifier')
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True, linestyle='--', linewidth=0.6, color=gridColour)
    ax.set_axisbelow(True) # grid behind the curve, otherwise it cuts through the line


def annotateAuc(ax, text):
    # AUC box in the lower right: the only free corner of a ROC plot whose curve is above the
    # diagonal, and far enough from the legend in the upper left
    ax.text(0.975, 0.20, text, transform=ax.transAxes, ha='right', va='center',
            bbox=dict(boxstyle='square,pad=0.55', facecolor='white', edgecolor='0.35',
                      linewidth=0.8))


def rocLegend(ax, loc):
    # the diagonal is drawn before the curve (so that it lies underneath it), which would put
    # the reference line at the top of the legend. The results are listed first instead and the
    # reference line last, matching the reading order of the figure.
    handles, labels = ax.get_legend_handles_labels()
    order = [i for i, text in enumerate(labels) if text != 'Random classifier']
    order += [i for i, text in enumerate(labels) if text == 'Random classifier']
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc=loc, frameon=True,
              facecolor='white', edgecolor='0.75', framealpha=1.0, handlelength=2.4,
              borderpad=0.6)


def plotRocCurve(yTrue, scores, label='ROC curve', title=None, showCi=True, alpha=0.05,
                 markPoints='auto', figSize=(textWidth, 0.8*textWidth)):
    # ROC curve of one predictor with its AUC and confidence interval.
    # yTrue: 0/1 class labels, scores: test result or predicted probability. For the binary OKN
    # the two are the same thing, which is why the curve then has a single corner point.
    # markPoints: 'auto' marks the ROC points only for a test with few thresholds (see
    # maxMarkedPoints), True/False force it.
    applyPlotStyle()
    yTrue = np.asarray(yTrue)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(yTrue, scores)

    # constrained layout instead of tight_layout: it fits the labels into the given figure size,
    # so the saved figure is exactly figSize wide (see saveFigure)
    figure, ax = plt.subplots(figsize=figSize, layout='constrained')
    decorateRocAxes(ax)
    # round caps and joins keep the corner of a binary ROC curve from looking clipped
    ax.plot(fpr, tpr, color=rocLine, linewidth=2.2, solid_capstyle='round',
            solid_joinstyle='round', zorder=3, label=label)
    if markPoints is True or (markPoints == 'auto' and len(fpr) <= maxMarkedPoints):
        ax.plot(fpr, tpr, linestyle='none', marker='o', markersize=6.5, color=rocPoint, zorder=4)

    auc, lower, upper = aucConfidenceInterval(yTrue, scores, alpha=alpha)
    if showCi and np.isfinite(lower):
        annotateAuc(ax, f"AUC = {auc:.3f}; {100*(1-alpha):.0f}% CI = [{lower:.3f}, {upper:.3f}]")
    else:
        annotateAuc(ax, f"AUC = {auc:.3f}")

    # one curve plus the diagonal: the legend is still needed, because the dashed line has to be
    # identifiable as the reference rather than as a second result
    rocLegend(ax, 'upper left')
    if title:
        ax.set_title(title, pad=12)
    return figure, ax


def formatPValue(pValue, digits=3):
    # p values are printed as in the log: rounded, with the smallest ones not rounded to 0.000,
    # which would read as "p = 0" and is never true.
    if not np.isfinite(pValue):
        return 'n/a'
    smallest = 10**(-digits)
    return f"<{smallest:.{digits}f}" if pValue < smallest else f"{pValue:.{digits}f}"


def plotOddsRatioForest(estimates, orColumn='OR', lowColumn='low', highColumn='high',
                        pColumn='p', title=None, xLabel='Odds Ratio', xLimits=None,
                        xScale='linear', xTicks=None, significanceLevel=0.05, figSize=None,
                        rowHeight=0.55):
    # forest plot of odds ratios: one row per entry of estimates, the dot at the odds ratio and
    # the bar over its confidence interval, with the numbers repeated as text on the right so
    # that the figure can be read without measuring against the axis.
    # estimates: DataFrame indexed by the row label, see statsTools.oddsRatioFrame. What a row
    # is, is up to the caller - one predictor of one model, or as in the prognosis section the
    # same predictor across several outcomes.
    # Marker and bar use the two colours of the ROC curves, so both figure types read as one
    # family: the lighter tone for the interval, the darker one for the point estimate.
    applyPlotStyle()
    labels = list(estimates.index)
    rows = len(labels)
    if figSize is None:
        # text width, so it is inserted unscaled like the ROC curves; the height grows with the
        # number of rows, so their spacing stays the same
        figSize = (textWidth, max(2.0, rowHeight*rows + 1.1))

    # left axis carries the plot, right axis only the number columns
    figure, (ax, axText) = plt.subplots(1, 2, figsize=figSize, width_ratios=[2.2, 1.5],
                                        layout='constrained')
    ax.set_xscale(xScale)
    if xLimits is None:
        highest = float(np.nanmax(estimates[highColumn]))
        lowest = float(np.nanmin(estimates[lowColumn]))
        xLimits = (0.0, highest*1.05) if xScale == 'linear' else (lowest*0.8, highest*1.25)
    ax.set_xlim(*xLimits)
    xMin, xMax = ax.get_xlim()

    for position, label in enumerate(labels):
        row = estimates.loc[label]
        low, high = float(row[lowColumn]), float(row[highColumn])
        # the interval is clipped to the axis and gets an arrow head where it runs off it.
        # Without that, a single wide interval would set the axis range and push all remaining
        # estimates onto the line of no effect, where they could no longer be told apart.
        ax.plot([max(low, xMin), min(high, xMax)], [position, position], color=rocLine,
                linewidth=2.0, solid_capstyle='round', zorder=3)
        for bound, marker in ((high, '>'), (low, '<')):
            if (bound > xMax) if marker == '>' else (bound < xMin):
                ax.plot([xMax if marker == '>' else xMin], [position], marker=marker,
                        markersize=7, color=rocLine, linestyle='none', zorder=3,
                        clip_on=False) # the head sits exactly on the axis and would be halved
        ax.plot([float(row[orColumn])], [position], marker='o', markersize=6, color=rocPoint,
                linestyle='none', zorder=4)

    # line of no effect. Dashed and black like the diagonal of the ROC curves, for the same
    # reason: it is a reference, not a result.
    ax.axvline(1.0, linestyle='--', color=referenceColour, linewidth=1.0, zorder=2)
    ax.set_yticks(range(rows))
    # the row labels are at body size rather than tick size, because they sit on one line with
    # the numbers of the text columns
    ax.set_yticklabels(labels, fontsize=figureFontSize)
    ax.set_ylim(rows - 0.5, -1.2) # inverted: first row on top, with space for the header line
    if xScale == 'log':
        # plain numbers instead of matplotlib's 10^0 notation: an odds ratio is read as "2.0"
        # or "0.5", and powers of ten would have to be decoded first. On a linear axis the
        # default ticks are already plain numbers and are left alone.
        ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                 if xMin <= t <= xMax]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    if xTicks is not None:
        ax.set_xticks(list(xTicks))
        ax.set_xticklabels([f"{t:g}" for t in xTicks])
    ax.set_xlabel(xLabel)
    # only vertical grid lines: one horizontal line per row would box in the estimates
    ax.grid(True, axis='x', linestyle='--', linewidth=0.6, color=gridColour)
    ax.set_axisbelow(True)

    # number columns. They live in a second axes so that their position does not depend on the
    # data range, which text placed inside the plot would.
    axText.set_ylim(ax.get_ylim())
    axText.set_xlim(0, 1)
    axText.axis('off')
    headerRow = -1.2
    for position, label in enumerate(labels):
        row = estimates.loc[label]
        # the star marks the conventional significance level, as in the earlier figure
        star = '*' if float(row[pColumn]) < significanceLevel else ''
        axText.text(0.18, position, f"{formatPValue(row[pColumn])}{star}",
                    va='center', ha='center')
        axText.text(0.5, position, f"{float(row[orColumn]):.3f}\n"
                                   f"[{float(row[lowColumn]):.3f}, {float(row[highColumn]):.3f}]",
                    va='center', ha='left')
    axText.text(0.18, headerRow, 'p-value', va='center', ha='center', fontweight='bold')
    axText.text(0.5, headerRow, 'Odds Ratio\n[95% CI]', va='center', ha='left', fontweight='bold')

    if title:
        figure.suptitle(title)
    return figure, ax
