

import numpy as np
import pandas as pd


import itertools


import statsmodels.stats.multitest
from scipy.stats import fisher_exact



def stats_results_categorical_fisher(df):
    
    '''
    Takes input of dataframe summarizing yes/no as 0/1 for a categorical variable such as a tracheostomy 
    and iterates over the four patient categories 
    '''
    
    stats_results = []

    for d1, d2 in itertools.combinations(df.Patient_category.unique(), 2):
            a1 = df[1][df.Patient_category==d1].values[0]
            a2 = df[1][df.Patient_category==d2].values[0]
            b1 = df[0][df.Patient_category==d1].values[0]
            b2 = df[0][df.Patient_category==d2].values[0]

            odds, pval = fisher_exact([ [a1, b1],
                                            [a2, b2]])
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    return pairs, stat_results_sign
