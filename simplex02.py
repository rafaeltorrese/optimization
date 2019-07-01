import numpy as np


C = np.array([1,-3,2])      # Objective Function Coefficients

X = np.array([[3,-1,2],
               [-2,4,0],
               [-4,3,8]]) # Body coefficients
nslacks = 3
b = np.array([7,12,10]) # Right Hand Coefficients


I = np.array([[-1,0,0,1],
              [0,1,0,0],
              [0,0,1,0]
              ])

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
    print(Cj)
    RHS = RHS.astype(float) # Right Hand Side
    RHS = RHS[:,np.newaxis] # convert to column vector
    
    Amatrix = np.hstack((Amatrix,RHS))
    cols= np.sort(np.where(Amatrix == 1)[1]) #indexes 
    cols.sort() # sort Column Indexes
    basics = cols[cols >= nvars] # select only slack and artificial indexes
    print(basics)
    cb  = Cj[basics] # Basic Coefficients    
    print(cb)
    Zj = cb.dot(Amatrix) 
    NetProfit = Cj-Zj[:-1] # cj - Zj
    iteration = 0
    #Iteration

    
    #if direction == 1:
    while np.any((direction*NetProfit )> 0):
        iteration += 1
        
        entry = np.argmax((direction*NetProfit)) # For Maximization Problem
        #iterate()
        # Define Key
        ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
        columnEntry = Amatrix[:,entry]
        if np.all(columnEntry == np.Infinity):break
        ratios[columnEntry < 0] = np.infty # if there exists negative ratios
        if np.all(ratios == np.infty):break
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
        
            
            
            
#    else:
#        while np.any(NetProfit < 0):
#            iteration += 1
#            
#            entry = np.argmin(NetProfit)
#            
#            # Define Key
#            ratios = Amatrix[:, -1] / Amatrix[:,entry] # RHS / Entry column
#            
#            columnEntry = Amatrix[:,entry]
#            ratios[columnEntry < 0] = np.infty # if there exists negative ratios
#            leave = np.argmin(ratios) # get the index with minimum value
#            pivot = Amatrix[leave,entry]
#            # Updte row with pivot and row leaving
#            rowKey = Amatrix[leave,:] / pivot
#
#
#            print(entry,leave, "\n")
#
#            
#            Amatrix[leave,:] = rowKey
#        
#            # Solve Equations
#            for row in range(Amatrix.shape[0]):
#                if(row == leave): continue
#                Amatrix[row, : ] =Amatrix[row, : ] - (Amatrix[row,entry] * rowKey   )
#                
#        
#            #updates values
#            basics[leave] = entry  
#            cb  = Cj[basics]
#            Zj = cb.dot(Amatrix)
#            NetProfit = Cj-Zj[:-1] # cj - Zj, Except last column (RHS)
#            print(f"Iteration{iteration}")
#            print(Amatrix, "\n")
#            if iteration == 20:break
#            
            
    return dict(zip(basics,Amatrix[:, -1])), Zj[-1], Amatrix




solutions,z,finalMatrix = simplex(X,C,b,S,A, direction=-1)

print(solutions)
print(z)
print("\n",finalMatrix)

#np.savetxt("matrix.csv",body,delimiter=",")
