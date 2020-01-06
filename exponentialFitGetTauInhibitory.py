
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expFunc(x, a, b, c):
    return a * np.exp( -(1/b) * x) + c
    
def getExponentialPart(x, y, nbPointsForFit):
  maxIndex = y.argmax(axis=0)
  if nbPointsForFit == 0:
    nbPointsForFit = len(y)
  else:
    nbPointsForFit = maxIndex + nbPointsForFit
  xExpPart = x[maxIndex:nbPointsForFit]  
  yExpPart = y[maxIndex:nbPointsForFit]
  return [xExpPart, yExpPart]

def exponentialFitGetTau(x, y, showPlot=0, nbPointsForFit=0):
  """
  This function outputs a monoexponential fit to points denoted by x and y. 
  It works for upward slopes as it finds the max of y as starting point.
  3rd term = show plot of monoexponential fit (1) or not (0)
  4th term = number of points for which to do the fit
  """
  
  [xExpPart, yExpPart] = getExponentialPart(x, y, nbPointsForFit)
  popt, pcov = curve_fit(expFunc, xExpPart, yExpPart, p0=[np.amax(yExpPart), 200, 0])
  if showPlot:
    print(popt)
    plt.plot(xExpPart, yExpPart)
    plt.plot(xExpPart, expFunc(xExpPart, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()
  return popt[1]
