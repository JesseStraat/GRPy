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
        
        # Tensor is a square symmetric sympy array
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
    
    def geodesics(self, variables, parameter):
        return self.christoffel(variables).geodesics(variables, parameter)
    
    def riemann(self, variables):
        return self.christoffel(variables).riemann(variables)
    
    def ricci(self, variables):
        return self.riemann(variables).ricci()
    
    def rscal(self, variables):
        return self.ricci(variables).rscal(self)

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
        
        # Tensor is a properly anti-symmetric square sympy array
        self.symbol = symbol
    
    def __repr__(self):
        return str(self.symbol)
    
    def geodesics(self, variables, parameter):
        l = parameter
        x = variables
        output = [0] * len(x)
        
        for mu in range(len(x)):
            s = Function(x[mu])(l).diff(l).diff(l)
            for a in range(len(x)):
                for b in range(len(x)):
                    s += self.symbol[mu,a,b]*Function(x[a])(l).diff(l)*Function(x[b])(l).diff(l)
            output[mu] = Eq(s, 0)
        
        return output
    
    def riemann(self, variables):
        n = self.symbol.shape[0]
        if len(variables) != n:
            raise ValueError(f"an inappropriate number of variables was given, must be {n}")
        for var in variables:
            if not isinstance(var, Symbol):
                raise TypeError("variables should be of class sympy.Symbol")
        riem = [[[[0]*n for _ in range(n)] for __ in range(n)] for ___ in range(n)]
        
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(n):
                        riem[rho][sig][mu][nu] = diff(self.symbol[rho,nu,sig],variables[mu]) - diff(self.symbol[rho,mu,sig],variables[nu]) + sum(self.symbol[rho,mu,a]*self.symbol[a,nu,sig] - self.symbol[rho,nu,a]*self.symbol[a,mu,sig] for a in range(n))
        
        return riemann(Array(riem))
    
    def ricci(self, variables):
        return self.riemann(variables).ricci()
    
    def rscal(self, variables, g):
        return self.ricci(variables).rscal(g)

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
        
        # Tensor is a properly anti-symmetric square sympy array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)
    
    def ricci(self):
        n = self.tensor.shape[0]
        ric = Matrix.zeros(n,n)
        for mu in range(n):
            for nu in range (n):
                ric[mu,nu] = sum(self.tensor[l,mu,l,nu] for l in range(n))
        
        return ricci(ric)
    
    def rscal(self, g):
        return self.ricci().rscal(g)

class ricci:
    def __init__(self, tensor = Matrix([[0]*4 for _ in range(4)]) ):
        if not isinstance(tensor, Matrix):
            raise TypeError("tensor must be sympy matrix")
        tens_shape = tensor.shape
        if len(tens_shape) != 2 or not all(n == tens_shape[0] for n in tens_shape):
            raise ValueError("tensor is not a square 2-array")
        
        # Tensor is a square sympy array
        self.tensor = tensor
    
    def __repr__(self):
        return str(self.tensor)
    
    def rscal(self, g):
        if not isinstance(g, metric):
            return TypeError("metric g must be of class metric")
        n = self.tensor.shape[0]
        ginv = g.tensor.inv()
        return sum(ginv[mu]*self.tensor[mu] for mu in range(n**2))

if __name__ == "__main__":
    chi, phi, k = symbols('chi phi k')
    g = metric(Matrix([[1, 0], [0, (sin(sqrt(k)*chi))**2/k]]))
    Chr = g.christoffel([chi, phi])
    Riem = Chr.riemann([chi, phi])
    Ric = Riem.ricci()
    Rscal = Ric.rscal(g)
    
    print(f"g = {g},\nChr = {Chr},\nRiem = {Riem},\nRic = {Ric},\nRscal = {Rscal}")