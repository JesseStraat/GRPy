import numpy as np
from sympy import *

class metric:
    def __init__(self, tensor = Matrix.diag([-1,1,1,1])):
        if not isinstance(tensor, Matrix):
            raise TypeError("tensor must be sympy matrix")
        tens_shape = tensor.shape
        if len(tens_shape) != 2 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 2-array")
        if not tensor.T.equals(tensor):                                         # Checks whether tensor is symmetric
            raise ValueError("tensor is not symmetric")
        
        # Tensor is a square symmetric np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)
    
    def christoffel(self, variables):
        n = self.tensor.shape[0]                                                # Dimension of manifold
        if len(variables) != n:
            raise ValueError(f"an inappropriate number of variables was given, must be {n}")
        for var in variables:
            if not isinstance(var, Symbol):
                raise TypeError("variables should be of class sympy.Symbol")
        chris = [[[0]*n for _ in range(n)] for __ in range(n)]
        invtensor = self.tensor.inv()
        
        for mu in range(n):
            for rho in range(n):
                for sig in range(n):
                    sumlist = [invtensor[mu,a]*(diff(self.tensor[sig,a],variables[rho]) + diff(self.tensor[rho,a],variables[sig]) - diff(self.tensor[rho,sig],variables[a])) for a in range(n)]
                    chris[mu][rho][sig] = 0.5*sum(sumlist)
        
        return christoffel(Array(chris))

class christoffel:
    def __init__(self, symbol = Array([[[0]*4 for _ in range(4)] for __ in range(4)]) ):
        if not isinstance(symbol, Array):
            raise TypeError("symbol must be sympy array")
        tens_shape = symbol.shape
        if len(tens_shape) != 3 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("symbol is not a square 3-array")
        for mu in range(tens_shape[0]):
            for rho in range(tens_shape[0]):
                for sig in range(rho+1, tens_shape[0]):
                    if not symbol[mu,rho,sig] == symbol[mu,sig,rho]:
                        raise ValueError("symbol is not symmetric in the bottom two indices")
        
        # Tensor is a properly anti-symmetric square np array
        self.symbol = symbol
    
    def __repr__(self):
        return str(self.symbol)

class riemann:
    def __init__(self, tensor = Array([[[[0]*4 for _ in range(4)] for __ in range(4)] for ___ in range(4)]) ):
        if not isinstance(tensor, Array):
            raise TypeError("tensor must be sympy array")
        tens_shape = tensor.shape
        if len(tens_shape) != 4 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 4-array")
        for rho in range(tens_shape[0]):
            for sig in range(tens_shape[0]):
                for mu in range(tens_shape[0]):
                    for nu in range(mu+1, tens_shape[0]):
                        if not tensor[rho,sig,mu,nu] == -1*tensor[rho,sig,nu,mu]:
                            raise ValueError("tensor is not anti-symmetric in the last two indices")
        
        # Tensor is a properly anti-symmetric square np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)

class ricci:
    def __init__(self, tensor = Matrix([[0]*4 for _ in range(4)]) ):
        if not isinstance(tensor, Matrix):
            raise TypeError("tensor must be sympy matrix")
        tens_shape = tensor.shape
        if len(tens_shape) != 2 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 2-array")
        
        # Tensor is a square np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)