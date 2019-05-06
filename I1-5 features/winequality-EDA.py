import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats

raw_data = pd.read_csv('winequality-red.csv', sep=";")

# view correlation efficieny result where |r|=1 has the strongest relation and |r|=0 the weakest
df = pd.DataFrame(data=raw_data)
print(df.corr())


#view data if it is normally distributed
plt.hist(raw_data["quality"], range=(1, 10),edgecolor='black', linewidth=1)
plt.xlabel('quality')
plt.ylabel('amount of samples')
plt.title("distribution of red wine quality")

# feature selection
import scipy.stats as stats
from scipy.stats import chi2_contingency


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)
        print('self:%s'%(self), self.chi2, self.p)

# Initialize ChiSquare Class
cT = ChiSquare(raw_data)

# Feature Selection
testColumns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
for var in testColumns:
    cT.TestIndependence(colX=var, colY="quality")
