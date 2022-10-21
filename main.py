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
        
        return christoffel(np.array(chris))

class christoffel:
    def __init__(self, symbol = np.zeros((4,4,4))):
        if not isinstance(symbol, np.ndarray):
            raise TypeError("symbol must be numpy array")
        tens_shape = symbol.shape
        if len(tens_shape) != 3 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("symbol is not a square 3-array")
        for T in symbol:
            if not np.array_equal(T.T, T):
                raise ValueError("symbol is not symmetric in the bottom two indices")
        
        # Tensor is a properly anti-symmetric square np array
        self.symbol = symbol
    
    def __repr__(self):
        return str(self.symbol)

class riemann:
    def __init__(self, tensor = np.zeros((4,4,4,4))):
        if not isinstance(tensor, np.ndarray):
            raise TypeError("tensor must be numpy array")
        tens_shape = tensor.shape
        if len(tens_shape) != 4 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 4-array")
        for T in tensor:
            for U in T:
                if not np.array_equal(U.T, -U):
                    raise ValueError("tensor is not anti-symmetric in the last two indices")
        
        # Tensor is a properly anti-symmetric square np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)

class ricci:
    def __init__(self, tensor = np.zeros((4,4))):
        if not isinstance(tensor, np.ndarray):
            raise TypeError("tensor must be numpy array")
        tens_shape = tensor.shape
        if len(tens_shape) != 2 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 4-array")
        
        # Tensor is a square np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)