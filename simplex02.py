import numpy as np
from itertools import combinations

Cj = np.array([4,3,6,0,0,0])       

basics = np.where(Cj == 0)[0] #indexes 
print(basics)

cb  = Cj[basics]


body = np.array([[2,3,2,1,0,0,440],[4,0,3,0,1,0,470],[2,5,0,0,0,1,430] ])



Zj = cb.dot(body)


cjZj = Cj-Zj[:-1]

entry = np.argmax(cjZj)

print(body[:,-1] / body[:,entry])
leave = np.argmin(body[:,-1] / body[:,entry])
pivot = body[leave,entry]


tm = body[leave,:] / np.float(pivot)
print(tm)
body[leave,:] = tm
print(body)


#update basics

basics[leave] = entry


#update Cb
cb  = Cj[basics]


# Iteration 2

Zj = cb.dot(body)

print("\n" , cb)
print(Zj)
