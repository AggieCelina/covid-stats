import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np

CONFIRMED_CASES = 'CSSEGI-stats/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
POLAND = 'Poland'
COUNTRY = 'Country/Region'

def convertToDateFormat(s):
    return datetime.strptime(s, '%m/%d/%y').date()

def importDataOfCaseType(cases):
    return pd.read_csv(cases)

def selectCountryDataForCaseType(country, caseType):
    return caseType.loc[caseType[COUNTRY] == country]

def getDatesAndValuesForCountryDataForCaseType(countryDataForCaseType):
    countryDataForCaseType = countryDataForCaseType.T
    countryDataForCaseType = countryDataForCaseType.drop(
    countryDataForCaseType.index[0:40])
    return list(countryDataForCaseType.index.values), list(countryDataForCaseType.values)

def showDataForCountryDataset(countryDataForCaseType):
    (time, cases) = getDatesAndValuesForCountryDataForCaseType(
        countryDataForCaseType)
    for i in range(len(cases)):
        time[i] = convertToDateFormat(time[i])
        cases[i] = int(cases[i])
    out_time = np.array(time)
    out_cases = np.array(cases)


    #plt.scatter(time, cases,c = "g", marker='o')
    sns.set()
    sns.regplot(out_time, out_cases)
    plt.show()

confirmedPolandDataForConfirmed = selectCountryDataForCaseType(
    POLAND, importDataOfCaseType(CONFIRMED_CASES))
showDataForCountryDataset(confirmedPolandDataForConfirmed)
print(getDatesAndValuesForCountryDataForCaseType(confirmedPolandDataForConfirmed))


#TODO: plot since the certain date
