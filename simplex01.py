import numpy as np
from itertools import combinations

def getCombinations(Matrix):
    rows,cols = Matrix.shape
    return list(combinations(range(cols),rows ))


def getBasisSolution(A,b,Objective):
    ''' Return Basis Solution with Trial Error Method For Simplex
    A: Coefficient Matrix
    b: Right-hand Coefficient Column Vector
    Objective: Coefficients of Objective Function
    '''
    variables = getCombinations(A)
    solutions = {}
    best = 0
    for variable in variables:
        try:
            X = np.zeros(A.shape[1]) # Number of columns
            basis = np.linalg.solve(A[ : ,[v for v in variable]] , b)
            if np.any(basis < 0):continue # unfeasible
            X[[v for v in variable]] = basis
            Z = X.dot(Objective)
            solutions[variable] = [X,Z]
            print("Solving For Basic {}".format(variable) , basis, X ,Z)
            if best < Z:
                best = Z
                sol = solutions[variable]
        except:
            print("Solving for Basic {}".format(variable), "Singular Matrix")
    return sol
            


D = np.array([[5,4,2,1],[2,3,8,1]])
bD = np.array([100,75])
zD = np.array([12,8,14,10])

print(getBasisSolution(D,bD,zD))
