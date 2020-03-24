import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares

CONFIRMED_CASES = 'CSSEGI-stats/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
POL_IMG_PATH = './docs/poland.png'
POLAND = 'Poland'
COUNTRY = 'Country/Region'
INIT_DATE = '3/8/20'
SPLIT_DATE = '3/17/20'
FINAL_DATE = '3/23/20'
CORR_DATE_POL_1 = '3/23/20'
CORR_DATE_POL_2 = '3/18/20'
PREDICTION_TERM = 2

def convertToDateFormat(s):
    return datetime.strptime(s, '%m/%d/%y').date()

def importDataOfCaseType(cases):
    return pd.read_csv(cases)

def selectCountryDataForCaseType(country, caseType):
    return caseType.loc[caseType[COUNTRY] == country]
    
def getDatesAndValuesForCountryDataForCaseType(countryDataForCaseType):
    countryDataForCaseType = countryDataForCaseType.T

    #######  Hack to fix faulty source data for Poland only
    for i in range(0, len(countryDataForCaseType.index)):
        if countryDataForCaseType.index[i] == COUNTRY and countryDataForCaseType.values[i] == POLAND:
            for i in range(0, len(countryDataForCaseType.index)):
                if countryDataForCaseType.index[i] == CORR_DATE_POL_1:
                    countryDataForCaseType.values[i] = [740]
                if countryDataForCaseType.index[i] == CORR_DATE_POL_2:
                    countryDataForCaseType.values[i] = [281]
    #######################################################

    countryDataForCaseType = countryDataForCaseType.drop(
    countryDataForCaseType.index[0:47])
    dataPlotIndices = range(0, len(countryDataForCaseType))

    return list(countryDataForCaseType.index.values), list(countryDataForCaseType.values), dataPlotIndices

def showDataForCountryDataset(countryDataForCaseType, xTestE, xTestQ, yTestExp, yTestQuad):
    (time, cases, dataPlotIndices) = getDatesAndValuesForCountryDataForCaseType(
        countryDataForCaseType)
    for i in range(len(cases)):
        cases[i] = int(cases[i])
    f, ax = plt.subplots(figsize=(14, 6))
    plt.scatter(dataPlotIndices, cases)
    plt.plot(xTestE, yTestExp, color='r')
    plt.plot(xTestQ, yTestQuad, color='g')
    plt.axvline(timeToPlotIndex(SPLIT_DATE), color='gray', linewidth=1.5, linestyle="--")
    plt.axvline(timeToPlotIndex(FINAL_DATE), color='gray',
                linewidth=1.5, linestyle="--")
    plt.xticks(np.arange(len(time)), time)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.title.set_text(POLAND)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(color='gray', linestyle='-', linewidth=0.3)
    ax.figure.savefig(POL_IMG_PATH)

def fitModels(countryDataForCaseType, initIndex, finalIndex, splitIndex):
    data = getDatesAndValuesForCountryDataForCaseType(
        countryDataForCaseType)
    dataPoints = list(map(lambda x: x[0], data[1]))
    yTrainExp = np.asarray(dataPoints[initIndex:splitIndex + 1])
    xTrainExp = np.asarray(range(initIndex, splitIndex + 1))
    xTestExp = np.asarray(range(initIndex, finalIndex + 1))

    yTrainQuad = np.asarray(dataPoints[splitIndex:timeToPlotIndex(FINAL_DATE) + 1])
    xTrainQuad = np.asarray(range(splitIndex, timeToPlotIndex(FINAL_DATE) + 1))
    xTestQuad = np.asarray(range(splitIndex, finalIndex + 1))


    def expFunMin(params, X, Y):
        return params[0] + params[1] * np.exp(params[2] * X) - Y

    def quadFunMin(params, X, Y):
        return params[0] + params[1] * X + params[2] * X**2 - Y

    def expFun(params, X):
        return params[0] + params[1] * np.exp(params[2] * X)

    def quadFun(params, X):
        return params[0] + params[1] * X + params[2] * X**2

    expOpt = least_squares(expFunMin, np.zeros(3), args=(xTrainExp, yTrainExp))
    quadOpt = least_squares(quadFunMin, np.zeros(3), args=(xTrainQuad, yTrainQuad))

    yTestExp = expFun(expOpt.x, xTestExp)
    yTestQuad = quadFun(quadOpt.x, xTestQuad)

    return xTestExp, xTestQuad, yTestExp, yTestQuad

confirmedPolandDataForConfirmed = selectCountryDataForCaseType(
    POLAND, importDataOfCaseType(CONFIRMED_CASES))

(time, cases, dataPlotIndices) = getDatesAndValuesForCountryDataForCaseType(
    confirmedPolandDataForConfirmed)

def timeToPlotIndex(timeStamp):
    return sum([x[1] if(x[0] == timeStamp) else 0 for x in zip(time, dataPlotIndices, cases)])

showDataForCountryDataset(confirmedPolandDataForConfirmed,
                          *fitModels(confirmedPolandDataForConfirmed, timeToPlotIndex(INIT_DATE),
                           len(cases) + PREDICTION_TERM,
                            timeToPlotIndex(SPLIT_DATE)))
