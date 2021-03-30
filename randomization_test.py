"""
    Author: Giorgia Cantisani
    Code for running an adaptation of the computationally-intensive randomization test [1], 
    a non-parametric hypothesis test. This code was used in the paper:

    Cantisani et al. "EEG-based decoding of auditory attention to a target instrument in 
    polyphonic music." 2019 IEEE Workshop on Applications of Signal Processing to Audio 
    and Acoustics (WASPAA).

    Considering a random classifier, the function computes its performances n_iter times, 
    leading to an empirical distribution of the performances. This empirical distribution 
    is then approximated with a theoretical distribution which could be a normal or a 
    t-distribution (the one that fits better). At this point, the function evaluates how 
    likely the input performances (given by y_pred and y_true) were to be produced by 
    this artificial distribution of performances obtaining the P-value.

    [1] E. W. Noreen, "Computer-intensive methods for testing hypotheses". Wiley New York, 1989.
"""

import random
import pandas as pd
import numpy as np
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def get_statistical_significance(p):
    """

    Parameters
    ----------
    p: [float > 0] P value

    Returns
    -------
    significance: [string] expressing the statistical significance associated with the given P value.
                  "****" denotes very high (p < 0.0001), "***" high (p < 0.001), "**" good (p < 0.01), 
                  "*" marginal (0.01 < p < 0.05) and "n.s." no (p > 0.05) statistical significance of the results.

    """

    if 5.00e-02 <= p:
        significance = 'n.s.'
    elif 1.00e-02 < p <= 5.00e-02:
        significance = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        significance = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        significance = '***'
    elif p <= 1.00e-04:
        significance = '****'

    return significance


def best_fit_distribution(data, distributions=[st.norm, st.t], bins=200, ax=None):
    """
    Credits to https://stackoverflow.com/a/37616966 

    Parameters
    ----------
    data: [array_like] input data
    distributions: [list of scipy's distributions] to search which is the one that best fits
    bins: [int > 0] number of bins for the histogram

    Returns
    -------
    the name and the parameters of the distribution that best fits the data

    """

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in distributions:

        # fit dist to data
        params = distribution.fit(data)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        # if axis pass in add to plot
        try:
            if ax:
                pd.Series(pdf, x).plot(ax=ax)
        except Exception:
            pass

        # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """
    Credits to https://stackoverflow.com/a/37616966 

    Parameters
    ----------
    dist: data distribution
    params: parameters of the
    size: [int > 0] size of the pdf and cdf

    Returns
    -------
    Generate distributions's probability density functions (y_pdf) and cumulative density functions (y_cdf) and
    the corresponding pandas Series (pdf and cdf).
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = 0  # dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = 1  # dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y_pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y_pdf, x)

    y_cdf = dist.cdf(x, loc=loc, scale=scale, *arg)
    cdf = pd.Series(y_cdf, index=x)

    return x, y_pdf, pdf, y_cdf, cdf


def randomization_hp_test(y_true, y_pred, tot_classes=[], classes_per_sample=[], n_iter=10000):
    """

    Parameters
    ----------
    y_true: [list] of the ground truth labels.
    y_pred: [list] of the predicted labels.
    tot_classes: [list] containing the set of labels of the entire dataset
    classes_per_sample: [list of lists] containing the sets of labels of each test sample 
    n_iter: [int > 0] number of iterations over wich computing the distribution of random performances

    Returns
    -------
    P value and corresponding statistical significance for an adaptation of the computationally-intensive 
    randomization test [1].
    """

    # This is f1 score but can be whatever metric you prefer (accuracy, precision, recall, ...)
    score = f1_score(y_true, y_pred)

    # Compute the score for a random classifier 
    score_random = np.empty(n_iter)
    for i in range(n_iter):
        y_pred_rand = []

        # in my case I can have a different set of classes for each test mixture (e.g. one with the bass and the drums and one contaning voice, bass and guitar).
        # Thus, for each song I cannot randomly choose among all the classes in the dataset (voice, bass, drums and guitar) because it would not be fair.
        # For each mixture in the test set, I randomly choose one of the instrument within that mixture. If you are not in this situation, and the possible 
        # classes for each test sample are always the same, you can give them as input as tot_classes and forget the parameter classes_per_sample.
        for j in range(len(y_pred)):       
            if len(tot_classes) == 0:
                y_pred_rand.append(np.random.choice(classes_per_sample[j]))
            else:
                y_pred_rand.append(np.random.choice(tot_classes))

        score_random[i] = f1_score(y_true, y_pred_rand)

    # Find best fit distribution
    data = pd.Series(score_random)
    best_fit_name, best_fit_params = best_fit_distribution(data, distributions=[st.norm, st.t], bins=200)
    best_dist = getattr(st, best_fit_name)

    # Make PDF with best params= 
    x, y_pdf, pdf, y_cdf, cdf = make_pdf(best_dist, best_fit_params)

    # Compute P value
    if score == 1:
        P = 1 - y_cdf[int(np.round(score, decimals=4)*10000-1)]
    else:
        P = 1 - y_cdf[int(np.round(score, decimals=4)*10000)]

    # Display
    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    cdf.plot(lw=2, label='CDF', legend=True)
    plt.axvline(x=score, color='red')

    # title
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    ax.set_title(' Scores with best fit distribution \n' + dist_str + ' P value ' + str(P) + ' ' + get_statistical_significance(P))
    ax.set_ylabel('Frequency')

    return P, get_statistical_significance(P)


