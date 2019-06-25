import numpy as np

X = np.array([[3,1,-1,1,0,0,0],
               [-1,4,1,0,-1,0,1],
               [0,1,1,0,0,1,0]]) # Body coefficients
C = np.array([-1,2,-1])      # Objective Function Coefficients
S = np.array([0,0,0]) # slack coefficients
A = np.array([-10000]) # # Artificial coefficients
b = np.array([10,6,4]) # Right Hand Coefficients





def simplex(Amatrix,CoefObject,RHS,Slack,Artificial=None,direction="max"):
    #Initialize
    CoefObject = CoefObject.astype(float)
    RHS = RHS.astype(float)
    if Artificial:
        Cj = np.concatenate((CoefObject,Slack,Artificial))
    else:
        Cj = np.concatenate((CoefObject,Slack))
    Amatrix = Amatrix.astype(float)
    
    RHS = RHS.astype(float) # Right Hand Side
    RHS = RHS[:,np.newaxis] # convert to column vector
    
    n = len(Slack) + len(Artificial)
    print(n)
    Amatrix = np.hstack((Amatrix,RHS))
    basics = np.sort(np.where(Amatrix == 1)[1][-n:]) #indexes 
    print(np.sort(np.where(Amatrix == 1)[1]))
    print(basics)
    cb  = Cj[basics] # Basic Coefficients    
    print(cb)
    Zj = cb.dot(Amatrix) 
    NetProfit = Cj-Zj[:-1] # cj - Zj
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
            cb  = Cj[basics]
            Zj = cb.dot(Amatrix)
            NetProfit = Cj-Zj[:-1] # cj - Zj, Except last column (RHS)
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
            cb  = Cj[basics]
            Zj = cb.dot(Amatrix)
            NetProfit = Cj-Zj[:-1] # cj - Zj, Except last column (RHS)
            print(f"Iteration{iteration}")
            print(Amatrix, "\n")
            
    return dict(zip(basics,Amatrix[:, -1])), Zj[-1], Amatrix




solutions,z,body = simplex(X,C,b,S,A, direction="max")

print(solutions)
print(z)
print("\n",body)

#np.savetxt("matrix.csv",body,delimiter=",")
