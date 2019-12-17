"""
Created on Wed April 17 2019
@author: oliver.mirat
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expFunc(x, a, b, c):
    return a * np.exp( -(1/b) * x) + c
    
def getExponentialPart(x, y, nbPointsForFit):
  minIndex = y.argmin(axis=0)
  if nbPointsForFit == 0:
    nbPointsForFit = len(y)
  else:
    nbPointsForFit = minIndex + nbPointsForFit
  xExpPart = x[minIndex:nbPointsForFit]  
  yExpPart = y[minIndex:nbPointsForFit]
  return [xExpPart, yExpPart]

def exponentialFitGetTau(x, y, showPlot=0, nbPointsForFit=0):
  [xExpPart, yExpPart] = getExponentialPart(x, y, nbPointsForFit)
  popt, pcov = curve_fit(expFunc, xExpPart, yExpPart, p0=[np.amin(yExpPart), 200, 0])
  if showPlot:
    print('Monoexponential fit is superimposed (red) on raw data (blue)')
    plt.plot(xExpPart, yExpPart)
    plt.plot(xExpPart, expFunc(xExpPart, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()
  return popt[1]
