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

    # def generate_more(self):
    #     if self.tensor:
    #         return torch.as_tensor(self._generate_sphere_data(more=True),dtype=torch.float)
    #     else:
    #         return self._generate_sphere_data(more=True)

# def generate_sphere_data(d,n_spheres,r):
#     shifts = np.random.randn(n_spheres,d+1)
#     sigma = np.sqrt(10.0/np.sqrt(float(d+1)))*np.eye(d+1)
#     shifts = [np.matmul(sigma,shifts[i].reshape(-1,1)).reshape(1,-1) for i in range(n_spheres)]
#     spheres = [shifts[i] + dsphere(n=1000,d=d,r=r) for i in range(n_spheres)]
#     envelope = dsphere(n=1000,d=d,r=5*r)
#     spheres.append(envelope)
#     return torch.as_tensor(np.vstack(spheres))
