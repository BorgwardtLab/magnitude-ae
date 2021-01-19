import numpy as np
import torch
from torch.utils.data import TensorDataset
from tadasets import dsphere

class Spheres():
    def __init__(self,d=100,n_spheres=10,r=5,tensor=True):
        self.d = d
        self.n_spheres = n_spheres
        self.r =r
        self.tensor = tensor
        self.shifts = np.random.randn(self.n_spheres,self.d+1)

    def generate(self,datapoints):
        if self.tensor:
            return torch.as_tensor(self._generate_sphere_data(datapoints),dtype=torch.float)
        else:
            return self._generate_sphere_data(datapoints)


    def _generate_sphere_data(self,datapoints):
        sigma = np.sqrt(10.0/np.sqrt(float(self.d+1)))*np.eye(self.d+1)
        shifts = [np.matmul(sigma,self.shifts[i].reshape(-1,1)).reshape(1,-1) for i in range(self.n_spheres)]
        spheres = [shifts[i] + dsphere(n=datapoints,d=self.d,r=self.r) for i in range(self.n_spheres)]
        envelope = dsphere(n=datapoints,d=self.d,r=5*self.r)
        spheres.append(envelope)
        return np.vstack(spheres)
