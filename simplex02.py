import numpy as np




C = np.array([3,4,1,5])      # Objective Function Coefficients

X = np.array("8 3 2 2 2 5 1 4 1 2 5 1".split(" ") , dtype=float).reshape((3,4)) # Body coefficients
nslacks = 3
b = np.array([10,5,6]) # Right Hand Coefficients


I =np.eye(nslacks)
S = np.zeros(nslacks) # slack coefficients

A = np.array([10000]) # # Artificial coefficients

X = np.hstack((X,I))





def simplex(Amatrix,CoefObject,RHS,Slack,Artificial=None,direction=1):
    #Initialize
    CoefObject = CoefObject.astype(float)
    nvars = len(CoefObject) # number of decision variables
    RHS = RHS.astype(float)
    if Artificial:
        Cj = np.concatenate((CoefObject,Slack,Artificial))
    else:
        Cj = np.concatenate((CoefObject,Slack))
    Amatrix = Amatrix.astype(float)
    
    RHS = RHS.astype(float) # Right Hand Side
    RHS = RHS[:,np.newaxis] # convert to column vector
    
    Amatrix = np.hstack((Amatrix,RHS))
    cols= np.sort(np.where(Amatrix == 1)[1]) #indexes 
    cols.sort() # sort Column Indexes
    basics = cols[cols >= nvars] # select only slack and artificial indexes
    
    cb  = Cj[basics] # Basic Coefficients    
    
    Zj = cb.dot(Amatrix) 
    NetProfit = Cj-Zj[:-1] # cj - Zj
    iteration = 0
    

    with open("results.csv" , "ba") as file:
        while np.any((direction*NetProfit )> 0):
            iteration += 1
            
            entry = np.argmax((direction*NetProfit)) # For Maximization Problem
            #iterate()
            # Define Key
            ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
            columnEntry = Amatrix[:,entry]
            if np.all(columnEntry < 0):break # Entry Column with all negative elements
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
            np.savetxt(file,np.vstack((Amatrix,Zj)),delimiter=",")
        
    return basics, Zj, Amatrix, iteration, NetProfit




basics,Zj,finalMatrix, iterations, netprofit = simplex(X,C,b,S,direction=1)


print(f"No Iterations: {iterations}")
print(f"Basics {basics}")
print(np.hstack((basics[:,np.newaxis],finalMatrix)))
print(f"Z value: {Zj}")
print(f"\n Net Profit {netprofit}")
print("\n",finalMatrix)

#np.savetxt("matrix.csv",body,delimiter=",")
