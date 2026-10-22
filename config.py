"""
Settings of the OKN analysis: input file, output paths, bootstrap, figure style and display options.
"""
import os

import pandas as pd

# paths
# the data set, available on request, with its codebook (okn_doc_variables.csv,
# okn_doc_values.csv) in the same folder
dataFile = os.path.join("data", "okn_doc_data.csv")
resultsPath = os.path.join("results")
os.makedirs(resultsPath, exist_ok=True)
logFile = os.path.join(resultsPath, "results.txt")
# figures are kept apart from the tables: results/ holds text and csv that are read, figures/
# holds the graphics that go into the write-up
figuresPath = os.path.join("figures")
os.makedirs(figuresPath, exist_ok=True)

# font of all figures. Changing these two values restyles every figure at once, so that the
# figures stay typographically consistent with each other. The remaining sizes (ticks, legend)
# are derived from figureFontSize in visualization.applyPlotStyle.
# The font has to be installed; if it is not, matplotlib falls back to a generic serif.
# Font and size are those of the body text of the manuscript. They only come out at that size
# if the figure is inserted at 100 %, i.e. without scaling - see textWidth below.
figureFont = 'Times New Roman'
figureFontSize = 10

# width of the text block of the manuscript in inches (matplotlib measures figures in inches):
# A4 is 21 cm wide, minus 2.5 cm margin on either side. Figures are drawn exactly this wide, so
# Word inserts them at 100 % and a 10 pt label in the figure is 10 pt on the page. Scaling a
# figure in Word afterwards scales its text as well.
pageWidthCm = 21.0
marginLeftCm = 2.5
marginRightCm = 2.5
textWidth = (pageWidthCm - marginLeftCm - marginRightCm) / 2.54

# colours of the figures. colourScheme picks one of the schemes below and thereby restyles
# every line in the ROC curves and the forest plot at once:
#   'default' - the green house colours, for slides and on-screen use
#   'simple'  - black and grey only, for journals that print in greyscale or charge for colour
# Each scheme sets:
#   rocLine           line of the ROC curve and the confidence interval bars of the forest plot
#   rocPoint          ROC points and the odds ratio markers - the darker partner of rocLine
#   referenceColour   diagonal of the random classifier and the line of no effect
#   gridColour        grid lines; grey values are given as matplotlib strings ('0' to '1')
colourSchemes = {
    # the ROC curve is drawn in a light green with the ROC points in a darker green of the same
    # hue, so that line and points read as one series instead of two.
    'default': {
        'rocLine': '#93c1a3',
        'rocPoint': '#2e7d32',
        'referenceColour': 'black',
        'gridColour': '0.85',
    },
    # greyscale counterpart: the same light/dark pairing, only in grey values.
    'simple': {
        'rocLine': '0.62',
        'rocPoint': '0.15',
        'referenceColour': 'black',
        'gridColour': '0.85',
    },
}
colourScheme = 'simple'
if colourScheme not in colourSchemes:
    # fail here rather than in the middle of a plot, where the message would be a KeyError
    raise ValueError(f"unknown colourScheme {colourScheme!r}, "
                     f"expected one of {sorted(colourSchemes)}")
figureColours = colourSchemes[colourScheme]

# bootstrap settings for the optimism correction of the AUCs in the comparison section.
# the seed is fixed so that reported AUCs are reproducible.
bootstrapSamples = 500
bootstrapSeed = 42

# print wide tables (e.g. the 14 column metrics table) without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
