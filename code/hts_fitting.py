'''
    A library to load, plot and fit Critical Current (Ic) and Critical Temperature (Tc) of REBCO coated conductors, 
    as required by the Jupyter Notebook that generate the figures of the RSI Paper titled: A facility for cryogenic ion 
    irradiation and in operando characterization of Rare-Earth Barium Copper Oxide superconducting tapes by A.R Devitre, D.X. 
    Fischer, K.B. Woller, B.C. Clark, M.P. Short, D.G. Whyte, and Z.S. Hartwig.
    
    @author Alexis Devitre (devitre@mit.edu)
    @update 2024/04/15
'''
import numpy as np
import matplotlib.pyplot as plt
import platform
import seaborn as sns
import pandas as pd
from scipy import integrate, constants
from scipy.optimize import curve_fit
import ipywidgets as widgets

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.markersize'] = 8
plt.rcParams['figure.figsize'] = 8, 8


########################################################################################
########################################################################################
######################## LOADING DATA FROM IV & TV MEASUREMENTS ########################
########################################################################################
########################################################################################


def readIV(fpath, fformat='mit', logIV=False, vc=2e-7, maxV=20e-6, iMin=0, vb=False):
    try:
        if vb: print('\n\n'+fpath+'\n')

        if fformat == 'mit':
            current, voltage, tHTS, tTAR = np.genfromtxt(fpath, usecols=[2, 3, 4, 5], unpack=True)
        elif fformat == 'tuv':
            current, voltage = np.genfromtxt(fpath, usecols=[0, 1], skip_footer=2, delimiter=' ', unpack=True)
            if platform.system() == "Windows":
                sep = '\\'
            else:
                sep = '/'
            temperature = float(fpath.split(sep)[-1].split('_')[1])

        # Invert the voltage if current was run in reverse direction
        if voltage[np.argmax(np.abs(voltage))] < 0:
            voltage *= -1
            if vb: print('Voltage was negative, array multiplied by -1')

        # filter points above the voltage threshold, where the power law is a good fit
        stable = (voltage < maxV)

        if logIV:
            # This assumes that there are at least 3 points before the exponential rise.
            # it was added to deal with a case where there is a large negative offset that 
            # eliminates most points when filtering the logvoltage
            voltage -= np.nanmean(voltage[:3])
            current, voltage = np.log(current), np.log(voltage)
            inRange = np.ones(len(current), dtype=bool)
        else:
            inRange = (current >= iMin)

        # filter points where current | voltage is NaN
        finite = (np.isfinite(current) & np.isfinite(voltage))

        cut = finite & stable & inRange
        current, voltage = current[cut], voltage[cut]

        if fformat == 'mit':
            temperature = [tHTS[cut], tTAR[cut]]
            
    except Exception as e:
        print('readIV raised: ', e)
        
    return current, voltage, temperature

def aggregateIVs(fpaths):
    current, voltage = np.array([]), np.array([])
    for fpath in fpaths:
        ic, n, i, v, chisq, pcov = fitIcMeasurement(fpath, function='powerLaw', vMax=20e-6, iMin=0, vb=False)
        current, voltage = np.append(current, i), np.append(voltage, v)
    return current, voltage

    
def readTV(fname, fformat='mit', vb=False):
    time, voltage, sampleT, targetT = np.genfromtxt(fname, usecols=[1, 3, 4, 5], unpack=True)
    # Invert the voltage if current was run in reverse direction
    if voltage[np.argmax(np.abs(voltage))] < 0:
        voltage *= -1
        if vb: print('Voltage was negative, array multiplied by -1')

    return time, voltage, sampleT, targetT


########################################################################################
########################################################################################
######################## PLOTTING DATA FROM IV & TV MEASUREMENTS #######################
########################################################################################
########################################################################################

def proportional(x, a):
    return a*x

def linear(i, a, b):
    return a*i+b

def powerLaw(i, ic, n):
    return 2e-7*(i/ic)**n

def plotFit(x, function, popt, fig=None, alpha=.5, linewidth=5, **kwargs):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = fig.axes[0]
    xsmooth = np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), 100000)
    ax.plot(xsmooth, 1e6*function(xsmooth, *popt), alpha=alpha, linewidth=linewidth, **kwargs)
    
    if (ax.get_legend() is not None) and any(l is not None for l in ax.get_legend().get_texts()):
        ax.legend(loc='upper left', fontsize=8)
    
    
def plotIV(fpath, fformat='mit', fig=None, linestyle='None', marker='+', **kwargs):
    current, voltage, _ = readIV(fpath, fformat=fformat, logIV=False, maxV=20e-6)
    voltage *= 1e6
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = fig.axes[0]
    
    ax.plot(current, voltage, linestyle=linestyle, marker=marker, **kwargs)

    ax.set_xticks(np.arange(np.floor(np.min(current)), np.ceil(np.max(current))+1, 1))
    ax.set_yticks(np.arange(np.floor(np.min(voltage)), np.ceil(np.max(voltage))+1, 1))
    ax.set_xlim(np.floor(np.min(current)), np.ceil(np.max(current)))
    ax.set_ylim(np.floor(np.min(voltage)), np.ceil(np.max(voltage)))
    ax.set_xlabel('Current [A]', fontsize=16)
    ax.set_ylabel('Voltage [uV]', fontsize=16)
    if (ax.get_legend() is not None) and any(l is not None for l in ax.get_legend().get_texts()):
        ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()

    
def plotIVs(fnames, fformat='mit', fig=None, outpath=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    for (fname, c) in zip(fnames, sns.color_palette('viridis', n_colors=len(fnames))):
        plotIV(fname, color=c, fig=fig, label=fname.split('/')[-1])
    
    if outpath is not None:
        plt.savefig(outpath, format='png', transparent=False)


########################################################################################
########################################################################################
######################## LOADING DATA FROM IV & TV MEASUREMENTS ########################
########################################################################################
########################################################################################


def fitIV(current, voltage, vc=.2e-6, function='powerLaw', p0=[], noiseLevel=1e-7, vb=False):
    popt, pcov, chisq = np.nan, np.nan, np.nan
    
    if isinstance(noiseLevel, np.ndarray): # user provided errorbars for each point
        yerr = noiseLevel  
    else: # we need to estimate the errorbars
        # If there are more than 50 points in the IV, take the first 10% to estimate the noise level.
        noiseN0, noiseN1 = int(np.ceil(len(current)*.1)), int(np.floor(len(current)*.8))
        if noiseN0-noiseN1 > 3:
            noiseLevel = np.std(voltage[noiseN0:noiseN1])
        yerr = np.ones_like(current)*noiseLevel
    
    if vb: print('current:\n', current,'\n\nvoltage:\n', voltage,'\n\nnoise:\n', yerr)
    
    if function == 'powerLaw':
        if p0 == []: p0 = [0.95*np.nanmax(current), 25.]
        popt, pcov = curve_fit(powerLaw, current, voltage, p0=p0, sigma=yerr, absolute_sigma=True)
        residuals = voltage - powerLaw(current, *popt)
    else:
        valid = (np.log(vc) <= voltage)
        current, voltage, yerr = current[valid], voltage[valid], yerr[valid]
        popt, pcov = curve_fit(linear, current, voltage, sigma=yerr, absolute_sigma=True)
        residuals = voltage - linear(current, *popt)
        noiseLevel = np.log(noiseLevel)
        
    dof = len(voltage)-2 #datapoints - #parameters
    chisq = np.sum((residuals/noiseLevel)**2)/dof
    
    if vb: 
        print('\ndof {:4.0f}, chisq = {:4.2f}'.format(dof, chisq))
        print('Current:\n', current, '\n')
        print('Voltage:\n', voltage, '\n')
        
    return popt, pcov, chisq


def fitIcMeasurement(fpath, fformat='mit', vc=.2e-6, function='powerLaw', vThreshold=1e-7, iMin=0, vMax=20e-6, vb=False):
    """
        fitIcMeasurement fits the IV data to a 2-parameter power law

        INPUTS
        ----------------------------------------------------------
        fpath (str)           - Path to file containing IV curve
        vc (float)            - Critical voltage as defined from Ec = 1 uV/cm. Typically 0.2 uV.
        function (str)        - Fitting function (powerLaw | linear on loglog scale)
        vThreshold (float)    - threshold use for background removal. Use voltage noise level ~ +/-.1 uV.

        RETURNS
        ----------------------------------------------------------
        ic - Fitted value of the critical current
        n  - Fitted value of the exponent
    """
    if vb: print('\n\n'+fpath+'\n')
        
    ic, n, chisq, current, voltage, pcov = np.nan, np.nan, np.nan, [], [], [[np.nan, np.nan], [np.nan, np.nan]]
    
    try:
        current, voltage, temperature = readIV(fpath, fformat=fformat, logIV=False, maxV=vMax, iMin=iMin)
        logcurrent, logvoltage, _ = readIV(fpath, fformat=fformat, logIV=True, maxV=vMax, iMin=iMin)
         
        # Find an initial estimate for Ic an n using a linear fit in logspace.
        if vb: print('File read:\n\n', logvoltage, '\n\n', logcurrent)
        popt, pcov, _ = fitIV(logcurrent, logvoltage, vc=vc, function='linear', vb=vb)
        
        ic, n = vc**(1./popt[0])/np.exp(popt[1]/popt[0]), popt[0]
        if vb: print(popt, pcov, '\n\nAfter a first round, ic and n are :', ic, n)
          
        # Remove the background using a powerLaw fit in linear space and the initial estimates.
        voltage = correctBackground(current, voltage, ic=ic, n=n, vThreshold=vThreshold)
        
        # Fit the data with the requested function
        if function == 'powerLaw':
            popt, pcov, chisq = fitIV(current, voltage, vc=vc, function='powerLaw', p0=[ic, n], vb=vb)
            ic, n = popt[0], popt[1]
        elif function == 'linear':
            popt, pcov, chisq = fitIV(logcurrent, logvoltage, vc=vc, function='linear', vb=vb)
            n, ic = popt[0], vc**(1./popt[0]) / np.exp(popt[1]/popt[0])
            
        if ((ic < 0) | (n < 1)):
            ic = n = np.nan
            
    except Exception as e: # Usually TypeError, IndexError
        if vb: print('fittingFuntions:fitIV returned: ', e)
    
    return ic, n, current, voltage, chisq, pcov
    
def correctBackground(current, voltage, vc=.2e-6, ic=30, n=30, vThreshold=1e-7, vb=False):
    iThreshold, newThreshold, tolerance, counter = ic*(vThreshold/vc)**(1/n), 1e6, 0.01, 1
    while((counter < 5) and (np.abs(iThreshold-newThreshold) > tolerance)):
        iThreshold = newThreshold
    
        cut = (current < ic*(vThreshold/vc)**(1./n))
        ccut, vcut = current[cut], voltage[cut]
        if (len(vcut) > 10):
            popt, pcov = curve_fit(linear, ccut, vcut)
        else:
            popt, pcov = curve_fit(linear, current, voltage)
        background = linear(current, *popt)
        voltage -= background

        popt, pcov, _ = fitIV(current, voltage, vc=vc, function='powerLaw', p0=[ic, n], vb=vb)
        ic, n = popt[0], popt[1]
        newThreshold = ic*(vThreshold/vc)**(1/n)
        counter += 1
        if vb: print('Remove background cycle {}/5\n\n'.format(counter))
    return voltage
