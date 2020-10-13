# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:39:24 2020

author: Trey Harrison (username: hmharri3)

Common plot functions compiled for repeated use.

List of functions:
    instrTypeToStr
    setCycler
"""

#%% import modules

from cycler import cycler
import matplotlib

#%% instrTypeToStr

def instrTypeToStr(instrumentationType):
    """ 
    Subcategory indicating instrumentation type used converted to string
    for plot labels.
    
    <333 T. Harrison 10/2020
    
    Input:
        1) instrumentationType - string to convert to words for people
    
    Output:
        1) iStr - the string
    
    boomboomboom
    """
    ##########################################################################
    if instrumentationType == 'PS':
        iStr = 'Static Pressure'
    elif instrumentationType == 'PT':
        iStr = 'Total Pressure'
    elif instrumentationType == 'TS':
        iStr = 'Static Temperature'
    elif instrumentationType == 'TT':
        iStr = 'Total Temperature'
        
    return iStr

#%% setFigSize

def setFigFormat(figFormat):
    """ 
    Set figure size for either journal/conference paper or presentation.
    
    <333 T. Harrison 10/2020
    
    Input:
        1) figFormat - string - either 'paper' or 'presentation'
    
    Output:
        1) changed matplotlib rcParams
    """
    ##########################################################################
    if figFormat == 'paper' or figFormat == 'journal':
        matplotlib.rcParams['figure.figsize'] = (3.5, 2.625)
        matplotlib.rcParams['font.size'] = 8
    elif figFormat == 'presentation' or figFormat == 'pres':
        matplotlib.rcParams['figure.figsize'] = (6.4, 4.8)
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['axes.xmargin'] = 0.5
    # return iStr

#%% define customer cycler for specified axes

def setCycler(ax=None):
    """ 
    Cycler set up to make plots in a way that Trey likes.
    Define cycler to apply to axes if provided. Also return the cycler.
    Ignore application to axes if axes not provided.
    
    <333 T. Harrison 9/2020
    
    Input:
        1) ax - optional - axes to apply cycler to
    
    Output:
        1) cc - cycler that Trey likes.
    """
    ##########################################################################
    # set up cycler
    
    cc = (cycler(color = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#e377c2', '#7f7f7f', '#9467bd']) +
      cycler(ls     = ['-',  '--', ':',  '-.', '-',  '--', ':',  '-.', '-',  '--']) + 
      cycler(lw     = [1.75, 1.75, 2.00, 1.75, 1.75, 1.75, 2.00, 1.75, 1.75, 1.75]) + 
      cycler(marker = ['o',  's',  '^',  'D',  '>',  'p',  'o',  's',  '^',  'D'])  +
      cycler(ms     = [5.00, 4.50, 5.50, 4.50, 5.25, 5.50, 4.50, 4.50, 5.50, 4.50]) + 
      cycler(mew    = [1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50]) + 
      cycler(mfc    = ['w',  'w',  'w',  'w',  'w',  'w',  'w',  'w',  'w',  'w',]))
    if ax is None:
        pass
    else:
        ax.set_prop_cycle(cc)    
    return cc

