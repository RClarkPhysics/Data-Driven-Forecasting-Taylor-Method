#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:48:17 2021

@author: RandallClark
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


class MultiTaylorRC:
    '''
    Initialize the Class
    Parametrs:
        name - The name of the Data set being trained on
        D - The number of dimensions of the data set being trained on and predicted
        T - The highest order of the taylor expansion to train and predict with
        dt - the time step
    '''
    def __init__(self,name,D,T,dt):
        self.name = name
        self.D = D
        self.T = T
        self.dt = dt
        self.order = 0
        for i in range(self.T+1):
            self.order = self.order + self.D**(i)
    
    '''
    Train on the Data set using the Taylor Series Expansion method
    This training method utilizes Ridge Regression along with a convenient choice of X and P
    Parameters:
        data - The data set that will be trained on (must be longer than trainlength)
        trainlength - the number of time steps to be trained on
        beta - the size of the regulator term
    '''
    def train(self,data,trainlength,beta):
        #We to generate our Y Target from the Data(1XT), and we need to put together our X (D^3 X T)
        funYTarget = self.generateY(data,trainlength,self.D,self.dt)
        YTarget = funYTarget()
        funX = self.generateX(data,trainlength,self.order,self.T,self.D)
        X = funX()

        P = np.zeros((self.D,int(self.order)))
        XX = np.linalg.inv(np.matmul(X,np.transpose(X))+beta*np.identity(self.order))
        for i in range(self.D):
            YX = np.matmul(YTarget[i],np.transpose(X))
            P[i] = np.matmul(YX,XX)
        self.P = P
        self.beta = beta
        self.endstate = np.zeros(self.D)
        for i in range(self.D):
            self.endstate[i] = data[i][trainlength-1]
        return P
    
    '''
    Predict forward using the Fast Explainable Forcasting Method
    Parameters:
        predictionlength - The amount of time to predict forward
    '''
    def predict(self,predictionlength):
        @njit
        def makeprediction(order,D,endstate,T,dt,P):
            Prediction = np.zeros((predictionlength,D))
            #To perform the prediction we first perform the base step followed by the recursive step
            #The lines of code below perform the same process done in the generateX function (organizing the data in the format")
            Xpre = np.zeros(order)
            for j in range(T+1):
                displacement = 0
                for l in range(j):
                    displacement = displacement+D**(l)
                for o in range(int(D**j)):
                    placeholder = o
                    Floor = np.zeros(j)
                    for s in range(j):
                        Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           
                        placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)    
                    
                    Xpre[o+displacement] = 1
                    for r in range(j):
                        Xpre[int(o+displacement)] = Xpre[int(o+displacement)]*endstate[int(Floor[r])] 
            for d in range(D):
                Prediction[0][d] = endstate[d] + dt*np.dot(P[d],Xpre)
            
            
            for i in range(1,predictionlength):
                #Construct X vector here
                Xpre = np.zeros(order)
                for j in range(T+1):
                    displacement = 0
                    for l in range(j):
                        displacement = displacement+D**(l)
                    for o in range(int(D**j)):
                        placeholder = o
                        Floor = np.zeros(j)
                        for s in range(j):
                            Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           
                            placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)         
                        
                        Xpre[o+displacement] = 1
                        for r in range(j):
                            Xpre[int(o+displacement)] = Xpre[int(o+displacement)]*Prediction[i-1][int(Floor[r])]
                
                for d in range(D):
                    #Make predictions
                    Prediction[i][d] = Prediction[i-1][d] + dt*np.dot(P[d],Xpre)
            return Prediction
        
        Prediction = makeprediction(self.order,self.D,self.endstate,self.T,self.dt,self.P)
        self.Prediction = Prediction
        return Prediction
    
    
    
    #SECONDARY FUNCTIONS-----------------------------------------------------------------------------------
    '''
    This is a tool used to put together the YTarget matrix during training
    '''
    def generateY(self,data,length,D,dt):
        @njit
        def genY():
            #This data is assumed to be D dimensional
            YTarget = np.zeros((D,length))
            for i in range(D):
                for j in range(length):
                    YTarget[i][j] = (data[i][j+1]-data[i][j])/dt
            return YTarget
        return genY
    '''
    This is a tool used to construct the X matrix during training
    '''
    def generateX(self,data,length,order,T,D):
        @njit
        def giveX():
            
            X = np.zeros((order,length))
            
            #We will cycle through the different phases of X (i.e. 1,x,x^2,x^3, and so on)
            for j in range(T+1):
                
                #This displacement counter will count to find the starting point for each cycle
                displacement = 0
                for l in range(j):
                    displacement = displacement+D**(l)
                
                #Cycle though all timepoints
                for i in range(length):
                    
                    #Cycle through all D**j inputs for a cycle
                    for l in range(int(D**j)):
                        #The Floor matrix will count in base D, and its values will span all combinations of X
                        placeholder = l
                        Floor = np.zeros(j)
                        for s in (range(j)):
                            Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           #Find out how many D**j can fit in l
                            placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)         #Now remove that value*D**j to calculate down one lower order in magnitude
                        
                        #Set the value to 1 (shift the inputs by the amount "displacement")
                        X[l+displacement][i] = 1
                        #Now multiply 1 by all combinations of X, where there are D**j combinations of X for j X's
                        for r in range(j):
                            X[int(l+displacement)][i] = X[int(l+displacement)][i]*data[int(Floor[r])][i]
            return X
        return giveX
    
    '''
    This is a graphing tool used once the prediction phase is over
    Parameter:
        Truedata - The Trued data from the data set that will be plotted side by side with the predicted data
    '''
    def SavePrettyGraph(self,TrueData):
        fig, axs = plt.subplots(self.D)
        fig.suptitle('Beta = '+str(self.beta))
        for i in range(self.D):
            axs[i].plot(np.transpose(self.Prediction)[i],label = 'Predicted Value')
            axs[i].plot(TrueData[i], label = 'True Value', color = 'r')
        plt.legend()
        plt.savefig('Plot.jpeg')
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    