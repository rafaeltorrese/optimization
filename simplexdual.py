import numpy as np




C = np.array("-1 -2 -3".split(" ") , dtype=float)      # Objective Function Coefficients

X = np.array("-2 1 -1 1 1 2 0 -1 2".split(" ") , dtype=float).reshape((3,3)) # Body coefficients


nslacks = 3
b = np.array([-4, 8, -2]) # Right Hand Coefficients


I =np.eye(nslacks)
S = np.zeros(nslacks) # slack coefficients


X = np.hstack((X,I))





def simplex(Amatrix,CoefObject,RHS,Slack):
    #Initialize
    Slack = Slack.astype(float)
    CoefObject = CoefObject.astype(float)
    nvars = len(CoefObject) # number of decision variables
    RHS = RHS.astype(float)
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
    

    
    while np.any(NetProfit <= 0) and np.any(Amatrix[:, -1] < 0):
        iteration += 1
        leaving = np.argmin(Amatrix[:, -1]) # Choose index of row key
        rowKey = np.array(Amatrix[leaving , :-1])
        
        if np.all(rowKey >= 0):break # Row Key with all non-negative elements
        rowKey[rowKey >= 0] = np.Inf # For excluding non-negative elements
        
        ratios = NetProfit / rowKey #
        ratios[ratios == 0] = np.Inf
        #ratios[ratios <= 0] = np.infty
        entry = np.argmin(ratios) # 
        rowKey = Amatrix[leaving , :] # Get Original Values fror RowKey
        pivot = Amatrix[leaving,entry]    
        
        rowKey = rowKey / pivot                   
        
        print(f"Entry: x{entry+1} Leaving: x{leaving+1} \n")
    
        Amatrix[leaving,:] = rowKey
        
        # Solve Equations
        for row in range(Amatrix.shape[0]):
            if(row == leaving): continue
            Amatrix[row, : ] = Amatrix[row, : ] - (Amatrix[row,entry] * rowKey)
            
    
        #updates values
        basics[leaving] = entry  
        cb  = Cj[basics]
        Zj = cb.dot(Amatrix)
        NetProfit = Cj-Zj[:-1] # cj - Zj, Except last column (RHS)
        print(f"Iteration {iteration}")
        print(Amatrix,"\n")
        
    return basics, Zj, Amatrix, iteration, NetProfit




basics,Zj,finalMatrix, iterations, netprofit = simplex(X,C,b,S)


print(f"No Iterations: {iterations}")
print(f"Basics {basics}")

print(np.hstack((basics[:,np.newaxis],finalMatrix)))
print(f"Z values: {Zj}")
print(f"Z : {Zj[-1]}")
print(f"\n Net Profit {netprofit}")
print("\n",finalMatrix)

np.savetxt("matrix.csv",np.hstack((basics[:,np.newaxis],finalMatrix)),delimiter=",")


