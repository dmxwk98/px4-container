import numpy as np
import sklearn.gaussian_process as gp
from math import ceil

# private libs.
from .BaseModules._gpr_modf import GaussianProcessRegressor
from .BaseModules.ParamsOffBoardCtrl import DataMPPI

class PF_GPR():
    def __init__(self) -> None:
        
        self.MPPIParams     =   DataMPPI()

        ## GPR params
        self.N              =   self.MPPIParams.N
        self.TrainMax       =   2 * self.N
        self.yPred          =   np.zeros((self.N, 3))
        self.var            =   np.zeros((self.N, 3))
        self.dt_Est         =   self.MPPIParams.dt_MPPI

        ## GPR vars
        self.GPRfullFLAG = 0
        self.GPRupdtFLAG = 0
        self.xTrain     =   []
        self.yTrainX    =   []
        self.yTrainY    =   []
        self.yTrainZ    =   []
        self.yTrain     =   []
        
        self.yPred      =   np.zeros((self.N, 3))
        self.var        =   np.zeros((self.N, 3))

    #.. GPR learning
        kernel      = gp.kernels.RBF(5e-1, (1e-4, 1e2)) * gp.kernels.ConstantKernel(5e-1, (1e-3, 1e2)) + gp.kernels.DotProduct(1,(1e-3, 1e2))
        self.gpr    =   GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0.1, normalize_y=True) #0.001
        pass

    def PF_GPR_Module(self, timestemp, outNDO):
        x_new   =   timestemp * 10 **(-6)
        Y_new   =   outNDO
        self.GPR_dataset(x_new,Y_new)
        self.GPR_update()

        self.GPR_estimate(x_new,testSize=self.N,dt=self.dt_Est)

        GPR_out     =   self.yPred        
        return GPR_out.tolist()

    def GPR_dataset(self, x_new, Y_new):
        self.GPRfullFLAG =   0

        self.xTrain     =   np.append(self.xTrain,  x_new)

        self.yTrainX    =   np.append(self.yTrainX, Y_new[0])
        self.yTrainY    =   np.append(self.yTrainY, Y_new[1])
        self.yTrainZ    =   np.append(self.yTrainZ, Y_new[2])

        # Maintaining buffer size
        if (len(self.yTrainX) > self.TrainMax):
            self.xTrain     =   np.delete(self.xTrain, 0, 0)

            self.yTrainX    =   np.delete(self.yTrainX, 0, 0)
            self.yTrainY    =   np.delete(self.yTrainY, 0, 0)
            self.yTrainZ    =   np.delete(self.yTrainZ, 0, 0)

            self.GPRfullFLAG = 1

        pass


    def GPR_update(self):

        if (self.GPRfullFLAG == 1):
            xTrain  =   np.array([self.xTrain[0], self.xTrain[ceil(0.5*self.TrainMax)], self.xTrain[self.TrainMax-1]])
            yTrainX  =   np.array([self.yTrainX[0], self.yTrainX[ceil(0.5*self.TrainMax)], self.yTrainX[self.TrainMax-1]])
            yTrainY  =   np.array([self.yTrainY[0], self.yTrainY[ceil(0.5*self.TrainMax)], self.yTrainY[self.TrainMax-1]])
            yTrainZ  =   np.array([self.yTrainZ[0], self.yTrainZ[ceil(0.5*self.TrainMax)], self.yTrainZ[self.TrainMax-1]])

            XTrain = xTrain.reshape(-1,1)
            yTrain = np.concatenate((yTrainX.reshape(-1,1), yTrainY.reshape(-1,1), yTrainZ.reshape(-1,1)), axis=1)

            self.gpr.fit(XTrain, yTrain)
            self.GPRupdtFLAG = 1

        pass


    def GPR_estimate(self, xTemp, testSize, dt):

        if (self.GPRupdtFLAG == 1):
            
            # start = time.time()
            # Get test points

            if(testSize > 1):
                xTest = xTemp + dt * np.arange(testSize)
            else:
                xTest = np.array([xTemp])
                
            XTest = xTest.reshape(-1,1)
            
            self.yPred, self.var = self.gpr.predict(XTest, return_std=True)

            # print("time :", time.time() - start)

        pass