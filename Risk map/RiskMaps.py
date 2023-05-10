# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:44:41 2023

@author: Wesse
"""

import matplotlib.pyplot as plt
import pandas as pd

def risk_matrix(data):
    if not isinstance(data,pd.DataFrame): raise ValueError("Wrong input, please enter pandas.DataFrame")
        
    """
        Function that gives a risk map:
        input param::
            - Data: pandas.Dataframe
        pandas.Dataframe has to following shape
        columns are the risk name that will be plotted
        First index of data is likelihood
        Second index of data is consequences
                
            
    """
    #setting up the figure
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.ylabel('Consequence')
    plt.xlabel('Likelihood')
    
    #This example is for a 5 * 5 matrix
    nrows=5
    ncols=5
    axes = [fig.add_subplot(nrows, ncols, r * ncols + c + 1) for r in range(0, nrows) for c in range(0, ncols) ]
    
    # remove the x and y ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,5)
        ax.set_ylim(0,5)

    
    green = [ 10, 15, 16, 20,21 , 22, ] #Green boxes
    yellow = [ 5,11, 17,23, 24] #yellow boxes
    orange = [0, 1 ,6,  7, 12, 13, 18, 19] # orange boxes
    red = [2, 3, 4, 8, 9, 14] #red boxes
    
    for _ in green:
        axes[_].set_facecolor('green')
    
    for _ in yellow:
        axes[_].set_facecolor('yellow')
    
    for _ in orange:
        axes[_].set_facecolor('orange')
    
    for _ in red:
        axes[_].set_facecolor('red')
    
    
    #reordering data for plotting
    boxes = [ [] ,[], [], [], [], [] ,[], [], [], [],[] ,[], [], [], [], [] ,[], [], [], [],[],[],[],[],[]]
    for risk in data:
        likelihood, consequence = data[str(risk)]
        box = (5-consequence) * 5 + likelihood -1
        boxes[box].append(risk)
    
    
    
    #Plot some data
    for i,risks in enumerate(boxes):
        if len(risks) ==0: continue 
        number_risks = len(risks)
        for j, risk in enumerate(risks):
            y_location_text = 2.5 + (j- number_risks/2) * 4.5/number_risks
            axes[i].text( 1.5, y_location_text , str(risk),)
        
            
    
    
    
  

props =        [ 2 , 1 , 1 , 1 , 1 , 2 , 3 , 3 , 1 , 2  , 1  , 1  , 4  ,  1 ,  2 , 2  , 3   , 2  , 1]
consequences = [ 3 , 3 , 4 , 4 , 3 , 3 , 2 , 2 , 4 , 3  , 2  , 3  , 1  , 3  ,  2 , 3  , 2  ,  3 , 3]
risks = []

for i in range(1,len(props)+1):
    risks.append( "TR-" + str(i)) 

data = pd.DataFrame([props,consequences],columns = risks, index = ["Propability", "Consequences"])

figure = risk_matrix(data)