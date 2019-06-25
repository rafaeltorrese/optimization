import numpy as np

A1 = np.array([[2,3,2,1,0,0],[4,0,3,0,1,0],[2,5,0,0,0,1] ])
C1 = np.array([4,3,6,0,0,0])      # Objective Function Coefficients
B1 = np.array([440,470,430])

#minimize Example 2.16-5
A2 = np.array([[3,-1,2,1,0,0],[-2,-4,0,0,1,0],[-4,3,8,0,0,1]])
C2 = np.array([1,-3,3,0,0,0])
B2 = np.array([7,12,10])

#maximize
A3 = np.array([[-1,1,0,1,0,0],[0,-1,2,0,1,0],[1,1,1,0,0,1]])
C3 = np.array([12,15,14,0,0,0])
B3 = np.array([0,0,100])


#maximize
A4 = np.array([[1,1,1,1,0,0],[2,3,5,0,1,0],[2,-1,-1,0,0,1]])
C4 = np.array([3,2,5,0,0,0])
B4 = np.array([9,30,8])


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
            columnEntry = Amatrix[:,entry]
            ratios[columnEntry < 0] = np.infty # if there exists negative ratios
            leave = np.argmin(ratios) # get the index with minimum value
            pivot = Amatrix[leave,entry]
            # Updte row with pivot and row leaving
            rowKey = Amatrix[leave,:] / pivot

            
            print(entry,leave, "\n")
        

            Amatrix[leave,:] = rowKey
            
            # Solve Equations
            for row in range(Amatrix.shape[0]):
                if(row == leave): continue
                Amatrix[row, : ] =Amatrix[row, : ] - (Amatrix[row,entry] * rowKey)
                #print(Amatrix,"\n")
        
            #updates values
            basics[leave] = entry  
            cb  = CoefObject[basics]
            Zj = cb.dot(Amatrix)
            NetProfit = CoefObject-Zj[:-1] # cj - Zj, Except last column (RHS)
            print(f"Iteration {iteration}")
            print(Amatrix,"\n")
            
            
            
    else:
        while np.any(NetProfit < 0):
            iteration += 1
            entry = np.argmin(NetProfit)
            
            # Define Key
            ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
            columnEntry = Amatrix[:,entry]
            ratios[columnEntry < 0] = np.infty # if there exists negative ratios
            leave = np.argmin(ratios) # get the index with minimum value
            pivot = Amatrix[leave,entry]
            # Updte row with pivot and row leaving
            rowKey = Amatrix[leave,:] / pivot


            print(entry,leave, "\n")

            
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
            print(f"Iteration{iteration}")
            print(Amatrix, "\n")
            
    return dict(zip(basics,Amatrix[:, -1])), Zj[-1]




solutions,z = simplex(A4,C4,B4, direction="max")

print(solutions)
print(z)
