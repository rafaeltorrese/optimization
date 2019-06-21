import numpy as np

A1 = np.array([[2,3,2,1,0,0],[4,0,3,0,1,0],[2,5,0,0,0,1] ])
C1 = np.array([4,3,6,0,0,0])      # Objective Function Coefficients
B1 = np.array([440,470,430])

A2 = np.array([[-1,1,0,1,0,0,0],
               [ 0,-1,2,0,1,0,0],
               [ 1, 1,1,0,0,1,0]])

C2 = np.array([12,15,14,0,0,0])



A3 = np.array([[3,-1,2,1,0,0,7],
               [-2,-4,0,0,1,0,12],
               [-4,3,8,0,0,1,10]])
C3 = np.array([1,-3,3,0,0,0])



A4 = np.array([[2,3,2,1,0,0,440],
               [4,0,3,0,1,0,470],
               [2,5,0,0,0,1,430]])
C4= np.array([4,3,6,0,0,0])

def iterate():
    # Define Key
    ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
    ratios[ratios < 0] = np.infty # if there exists negative ratios
    leave = np.argmin(ratios) # get the index with minimum value
    pivot = Amatrix[leave,entry]
    # Updte row with pivot and row leaving
    rowKey = Amatrix[leave,:] / pivot
    Amatrix[leave,:] = rowKey
    
    # Solve Equations
    for row in range(Amatrix.shape[0]):
        if(row == leave): continue
        Amatrix[row, : ] =Amatrix[row, : ] - (Amatrix[row,entry] * rowKey   )        

    #updates values
    basics[leave] = entry  
    cb  = CoefObject[basics]
    Zj = cb.dot(Amatrix)
    NetProfit = CoefObject-Zj[:-1] # cj - Zj, Except last column (RHS)

def simplex(Amatrix,CoefObject,RHS,direction="max"):
    #Initialize
    Amatrix = Amatrix.astype(float)
    CoefObject = CoefObject.astype(float)
    RHS = RHS.astype(float) # Right Hand Side
    RHS = RHS[:,np.newaxis] # convert to column vector
    Amatrix = np.hstack((Amatrix,RHS))
    basics = np.where(CoefObject == 0)[0] #indexes 
    cb  = CoefObject[basics] # Basic Coefficients
    Zj = cb.dot(Amatrix) 
    NetProfit = CoefObject-Zj[:-1] # cj - Zj
    iteration = 0
    #Iteration


    if direction == "max":
        while np.any(NetProfit > 0):
            iteration += 1
            entry = np.argmax(NetProfit)
            #iterate()
            # Define Key
            ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
            ratios[ratios < 0] = np.infty # if there exists negative ratios
            leave = np.argmin(ratios) # get the index with minimum value
            pivot = Amatrix[leave,entry]
            # Updte row with pivot and row leaving
            rowKey = Amatrix[leave,:] / pivot
            Amatrix[leave,:] = rowKey
        
            # Solve Equations
            for row in range(Amatrix.shape[0]):
                if(row == leave): continue
                Amatrix[row, : ] =Amatrix[row, : ] - (Amatrix[row,entry] * rowKey   )
        
            #updates values
            basics[leave] = entry  
            cb  = CoefObject[basics]
            Zj = cb.dot(Amatrix)
            NetProfit = CoefObject-Zj[:-1] # cj - Zj, Except last column (RHS)
            
    else:
        while np.any(NetProfit < 0):
            iteration += 1
            entry = np.argmin(NetProfit)
            
            # Define Key
            ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
            ratios[ratios < 0] = np.infty # if there exists negative ratios
            leave = np.argmin(ratios) # get the index with minimum value
            pivot = Amatrix[leave,entry]
            # Updte row with pivot and row leaving
            rowKey = Amatrix[leave,:] / pivot
            Amatrix[leave,:] = rowKey
        
            # Solve Equations
            for row in range(Amatrix.shape[0]):
                if(row == leave): continue
                Amatrix[row, : ] =Amatrix[row, : ] - (Amatrix[row,entry] * rowKey   )
        
            #updates values
            basics[leave] = entry  
            cb  = CoefObject[basics]
            Zj = cb.dot(Amatrix)
            NetProfit = CoefObject-Zj[:-1] # cj - Zj, Except last column (RHS)
            
    return dict(zip(basics,Amatrix[:, -1])), Zj[-1]




solutions,z = simplex(A1,C1,B1, direction="max")

print(solutions)
print(z)
