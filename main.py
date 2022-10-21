import numpy as np
from sympy import *

class metric:
    def __init__(self, tensor = np.diag([-1,1,1,1])):
        if not isinstance(tensor, np.ndarray):
            raise TypeError("tensor must be numpy array")
        tens_shape = tensor.shape
        if len(tens_shape) != 2 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 2-array")
        if not np.array_equal(tensor.T, tensor):                                # Checks whether tensor is symmetric
            raise ValueError("tensor is not symmetric")
        
        # Tensor is a square symmetric np array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)

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