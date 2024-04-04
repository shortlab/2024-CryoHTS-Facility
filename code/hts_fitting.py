'''
    A library to load, plot and fit Critical Current (Ic) and Critical Temperature (Tc) of REBCO coated conductors.
    @author Alexis Devitre (devitre@mit.edu)
'''
import hts_misc
import numpy as np
import matplotlib.pyplot as plt
import platform
import seaborn as sns
import pandas as pd
from scipy import integrate, constants
from scipy.optimize import curve_fit
import ipywidgets as widgets

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

def showcaseIVs(fpaths):
    fig, ax = plt.subplots(figsize=(9, 4))
    
    def on_spinbox_value_change(change, ax):
        try:
            ax.clear()
            f = fpaths[spinbox.value]
            ax.set_title(f.split('/')[-1])
            i, v, temperature = readIV(f)
            ic, n, current, voltage, chisq, pcov = fitIcMeasurement(f, function='linear')
            ax.semilogy(i, 1e6*v, color='lightgray', marker='+', label='raw data')
            ax.semilogy(current, 1e6*voltage, color='k', marker='+', label='corrected voltage')
            xsmooth = np.linspace(np.min(current), np.max(current), 10000)
            cut = voltage > .2e-6
            ax.semilogy(current[cut], 1e6*voltage[cut], color='b', marker='+')
            ax.semilogy(xsmooth, 1e6*powerLaw(xsmooth, ic, n), linewidth=3, alpha=.2, color='b', label='powerLaw fit')
            ax.axhline(0.2)
            ax.legend()
            ax.set_ylim(1e-2, 1e2)
        except Exception as e:
            print(e)
    spinbox = widgets.IntText(description="IV#:", min=0, max=len(fpaths), value=1)
    spinbox.observe(lambda change: on_spinbox_value_change(change, ax), names='value')
    display(spinbox)
    spinbox.value = 0
    
    
def readTV(fname, fformat='mit', vb=False):
    time, voltage, sampleT, targetT = np.genfromtxt(fname, usecols=[1, 3, 4, 5], unpack=True)
    # Invert the voltage if current was run in reverse direction
    if voltage[np.argmax(np.abs(voltage))] < 0:
        voltage *= -1
        if vb: print('Voltage was negative, array multiplied by -1')

    return time, voltage, sampleT, targetT


def showcaseTVs(fpaths):
    fig, ax = plt.subplots(figsize=(9, 4))
    def on_spinbox_value_change(change, ax):
        try:
            ax.clear()
            f = fpaths[spinbox.value]
            ax.set_title(f.split('/')[-1])
            time, voltage, sampleT, targetT = readTV(f)
            ax.plot(sampleT, 1e6*voltage, color='k', marker='+', label='raw data')
            ax.axhline(0.2)
            ax.legend()
            ax.set_xlim(60, 90)
            ax.set_ylim(-1, 5)
        except Exception as e:
            print(e)
    spinbox = widgets.IntText(description="TV#:", min=0, max=len(fpaths), value=1)
    spinbox.observe(lambda change: on_spinbox_value_change(change, ax), names='value')
    display(spinbox)
    spinbox.value = 0

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

def inverseExponential(temperature, a, b, c, t50):
    #return a*temperature*(1-1/(np.exp(b*(temperature-t50))+1))
    return a*temperature-c/(np.exp(b*(temperature-t50))+1)

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

'''
def removeBackground(current, voltage, ic, n, noiseThreshold, vc):
    #print(ic, noiseThreshold, vc, n)
    iThreshold = ic*(noiseThreshold/vc)**(1./n)
    
    cut = (current < iThreshold)
    ccut, vcut = current[cut], voltage[cut]
    if (len(vcut) > 10):
        popt, pcov = curve_fit(linear, ccut, vcut)
    else:
        popt, pcov = curve_fit(linear, current, voltage)
    
    background = linear(current, *popt)
    return voltage - background
'''

def fitTc(temperature, voltage, time, bounds=(80, 90), ax=None, label='', filter_strength=(1.2*1e6, 11)):
    # slice the temperature range
    cut = (bounds[0] <= temperature) & (temperature <= bounds[1])
    temp = temperature
    temperature, voltage, time = np.sort(temperature)[cut], 1e6*np.array([x for _, x in sorted(zip(temp, voltage))])[cut], np.array([x for _, x in sorted(zip(temp, time))])[cut]
    
    # weed-out the large fluctuations
    dV = np.abs(np.append(0, np.diff(voltage)))
    cut = dV < filter_strength[0]
    temperature, voltage, time = temperature[cut], savgol_filter(voltage[cut], filter_strength[1], 2), time[cut]
    dV = np.abs(np.append(0, np.diff(voltage)))
    
    if ax is not None:
        rampRate = 60*np.append(0, np.diff(temperature)/np.diff(time))
        ax[0].plot(temperature, voltage, linestyle='-', marker='+', color='k')
        ax[1].plot(temperature, dV/np.max(dV), linestyle='-', marker='+', color='purple')
        ax[2].plot(temperature, rampRate, color='k', label='{:4.2f} K/min'.format(np.mean(rampRate)))
        ax[0].set_ylim(-1, np.max(voltage))
        ax[1].set_ylim(0, 1)
        
    # find t50
    cut = 0.5 <= dV/np.max(dV)
    temperature, voltage, dV = temperature[cut], voltage[cut], dV[cut]
    t50 = temperature[np.argmax(dV)]
    
    xsmooth = np.linspace(bounds[0], bounds[1], 100000)
    popt_rise, _ = curve_fit(linear, temperature, voltage)
    ysmooth_rise = linear(xsmooth, *popt_rise)

    tore = -popt_rise[1]/popt_rise[0]
    
    if ax is not None:
        ax[0].plot(temperature, voltage, linestyle='-', marker='+', color='orange', label=label)
        ax[1].plot(temperature, dV/np.max(dV), linestyle='-', marker='+', color='orange')
        ax[1].axhline(.5, color='orange', linestyle='--')
        ax[0].axvline(tore, label='Tore = {:4.2f} K'.format(tore), color='green')
        
        ax[0].plot(xsmooth, ysmooth_rise, linewidth=3, alpha=.4, color='orange')
        ax[1].set_xlim(bounds[0], bounds[1])
        ax[0].legend(loc='upper left')
        ax[2].legend(loc='upper right')
        
    return t50, tore


def fitTV(temperature, voltage):
    '''
        fitTV fits the TV data to a 3-parameter inverse exponential
        
        INPUTS
        ----------------------------------------------------------
        temperature (float, list) - measured temperatures
        voltage (float, list) - Measured voltages (must be same length as temperature)
        
        RETURNS
        ----------------------------------------------------------
        tc_lossless (float) - value of the temperature below which cooper pairs form
        tc_superconducting (float) - value of the temperature below which the current is carried without resistance
    '''
    try:
        # initial estimates
        a0 = voltage[-1]/temperature[-1]                                # Ohm's law crosses zero!
        b0 = 2.                                                         # Inverse of the transition width in the Fermi function
        T0 = temperature[np.argmin(np.abs(voltage-np.max(voltage))/2)]  # T50 is more or less at half the maximum voltage
        
        # use inverse exponential fit to determine T50
        popt, pcov = curve_fit(inverseExponential, temperature, voltage, p0=[a0, b0, T0])
        xsmooth = np.linspace(np.min(temperature), np.max(temperature))
        ysmooth = inverseExponential(xsmooth, *popt)
        
        # use linear fits to determine Tosc and Tore
        derivative = np.diff(ysmooth)/np.diff(xsmooth)
        
        slope_bottom = 0.
        inter_bottom = 0.
        
        slope_rise = np.max(derivative)
        inter_rise = ysmooth[np.argmax(derivative)]-slope_rise*xsmooth[np.argmax(derivative)]
        
        slope_top = popt[0] # Slope of the upper shelf in the Fermi function
        inter_top = 0.      # Ohm's law crosses zero
        
        Tore = -inter_rise/slope_rise
        T50  = popt[2]
        Tosc = inter_rise/(slope_top-slope_rise)
    
    except Exception as e:
        print('fittingFunctions::fitTV returned: ', e)
        Tore, T50, Tosc, popt = np.nan, np.nan, np.nan, [np.nan, np.nan, np.nan]
        
    return Tore, T50, Tosc, popt


########################################################################################
########################################################################################
######################## PLOTTING SPECIFIC DATASETS ####################################
########################################################################################
########################################################################################

def plotIcT(fpaths):
    fig, ax = plt.subplots()
    
    ics, temperatures = [], []
    for fpath in fpaths:
        current, voltage, temperature = readIV(fpath, fformat='mit', logIV=False, vc=2e-7, maxV=20e-6, iMin=0, vb=False)
        popt, pcov, chisq = fitIV(current, voltage)
        ics.append(popt[0])
        temperatures.append(np.mean(temperature[:-10]))
        
    ax.plot(temperatures, ics)