import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares

CONFIRMED_CASES = 'CSSEGI-stats/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
POLAND = 'Poland'
COUNTRY = 'Country/Region'
INIT_DATE = '3/8/20'
SPLIT_DATE = '3/17/20'
FINAL_DATE = '3/22/20'

def convertToDateFormat(s):
    return datetime.strptime(s, '%m/%d/%y').date()

def importDataOfCaseType(cases):
    return pd.read_csv(cases)

def selectCountryDataForCaseType(country, caseType):
    return caseType.loc[caseType[COUNTRY] == country]
    
def getDatesAndValuesForCountryDataForCaseType(countryDataForCaseType):

    countryDataForCaseType = countryDataForCaseType.T
    countryDataForCaseType = countryDataForCaseType.drop(
    countryDataForCaseType.index[0:47])
    dataPlotIndices = range(0, len(countryDataForCaseType))
    
    return list(countryDataForCaseType.index.values), list(countryDataForCaseType.values), dataPlotIndices

def showDataForCountryDataset(countryDataForCaseType, xTest, yTestExp, yTestQuad):
    (time, cases, dataPlotIndices) = getDatesAndValuesForCountryDataForCaseType(
        countryDataForCaseType)
    for i in range(len(cases)):
        cases[i] = int(cases[i])
    
    f, ax = plt.subplots(figsize=(12, 6))
    plt.plot(xTest, yTestExp, color='r')
    plt.plot(xTest, yTestQuad, color='g')
    plt.scatter(dataPlotIndices, cases)
    plt.axvline(timeToPlotIndex(SPLIT_DATE), color='gray', linewidth=1.5, linestyle="--")
    plt.xticks(np.arange(len(time)), time)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.title.set_text(POLAND)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(color='gray', linestyle='-', linewidth=0.3)
    plt.show()
    ax.figure.savefig('test.pdf')

def fitModels(countryDataForCaseType, initIndex, finalIndex, splitIndex):
    data = getDatesAndValuesForCountryDataForCaseType(
        countryDataForCaseType)
    
    dataPoints = list(map(lambda x: x[0], data[1]))

    yTrain = np.asarray(dataPoints[initIndex:splitIndex + 1])

    xTrain = np.asarray(range(initIndex, splitIndex + 1))
    xTest = np.asarray(range(initIndex, finalIndex + 1))

    def expFunMin(params, X, Y):
        return params[0] + params[1] * np.exp(params[2] * X) - Y

    def quadFunMin(params, X, Y):
        return params[0] + params[1] * X + params[2] * X**2 - Y

    def expFun(params, X):
        return params[0] + params[1] * np.exp(params[2] * X)

    def quadFun(params, X):
        return params[0] + params[1] * X + params[2] * X**2

    expOpt = least_squares(expFunMin, np.zeros(3), args=(xTrain, yTrain))
    quadOpt = least_squares(quadFunMin, np.zeros(3), args=(xTrain, yTrain))

    yTestExp = expFun(expOpt.x, xTest)
    yTestQuad = quadFun(quadOpt.x, xTest)

    return xTest, yTestExp, yTestQuad

confirmedPolandDataForConfirmed = selectCountryDataForCaseType(
    POLAND, importDataOfCaseType(CONFIRMED_CASES))

(time, cases, dataPlotIndices) = getDatesAndValuesForCountryDataForCaseType(
    confirmedPolandDataForConfirmed)

allSet = zip(time, dataPlotIndices, cases)

print(confirmedPolandDataForConfirmed)

def timeToPlotIndex(timeStamp):
    return sum([x[1] if(x[0] == timeStamp) else 0 for x in allSet])

showDataForCountryDataset(confirmedPolandDataForConfirmed,
                          *fitModels(confirmedPolandDataForConfirmed, timeToPlotIndex(INIT_DATE),
                           timeToPlotIndex(FINAL_DATE),
                            timeToPlotIndex(SPLIT_DATE)))