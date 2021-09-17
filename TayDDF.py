#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:48:17 2021

@author: RandallClark
"""
import numpy as np
import matplotlib.pyplot as plt


class TaylorDDF:
    '''
    Initialize the Class
    Parametrs:
        name - The name of the Data set being trained on
        D - The number of dimensions of the data set being trained on and predicted
    '''
    def __init__(self,D):
        self.D = D
    
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
        YTarget = self.generateY(data,trainlength)
        X = self.generateX(data,trainlength)
        
        P = np.zeros((self.D,int(1+self.D+self.D**2)))
        
        
        XX = np.linalg.inv(np.matmul(X,np.transpose(X))+beta*np.identity(1+self.D+self.D**2))
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
        Prediction = np.zeros((predictionlength,self.D))
        
        #Build K
        K = np.zeros(self.D)
        for i in range(self.D):
            K[i] = self.P[i][0]
        
        #Build M
        M = np.zeros((self.D,self.D))
        for i in range(self.D):
            for j in range(self.D):
                M[i][j] = self.P[i][j+1]
        
        #Build J
        J = np.zeros((self.D,self.D,self.D))
        for i in range(self.D):
            for j in range(self.D):
                for l in range(self.D):
                    J[i][j][l] = self.P[i][int(j*self.D+l+1+self.D)]
        
        
        #Initiate Predictions
        for i in range(self.D):
            Prediction[0][i] = self.endstate[i]+(
                                K[i] + 
                                np.matmul(self.endstate,M[i]) + 
                                np.matmul(np.matmul(self.endstate,J[i]),self.endstate)
                                )
        #Predict forward from 1 to the full prediction length
        for i in range(1,predictionlength):
            for j in range(self.D):
                Prediction[i][j] = Prediction[i-1][j]+(
                                K[j] + 
                                np.matmul(Prediction[i-1],M[j]) + 
                                np.matmul(np.matmul(Prediction[i-1],J[j]),Prediction[i-1])
                                )
        
        self.Prediction = Prediction
        return Prediction
    

    
    #SECONDARY FUNCTIONS-----------------------------------------------------------------------------------
    '''
    This is a tool used to put together the YTarget matrix during training
    '''
    def generateY(self,data,length):
        #This data is assumed to be D dimensional
        YTarget = np.zeros((self.D,length))
        for i in range(self.D):
            for j in range(length):
                YTarget[i][j] = data[i][j+1]-data[i][j]
        return YTarget
    '''
    This is a tool used to construct the X matrix during training
    '''
    def generateX(self,data,length):
        X = np.zeros((int(self.D**2+self.D+1),length))
        for j in range(length):
            X[0][j] = 1
        for i in range(self.D):
            for j in range(length):
                X[i+1][j] = data[i][j]
        
        for i in range(self.D):
            for l in range(self.D):
                for j in range(length):
                    X[int(i*self.D+l+1+self.D)][j] = data[i][j]*data[l][j]
        
        self.X = X
        return X
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    