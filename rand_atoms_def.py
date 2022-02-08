#the version of this file is for Functions branch
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:53:36 2021

@author: Matt
"""
#This script returns the position of a random atom in a plane surrounding the defect
#This atom will be selected for calculating the RDF
def rand_atoms_def(atoms,ao,rc):
    import numpy as np
    layers=np.ceil(rc/(ao/np.sqrt(3)))*2
    pos=np.zeros((2*int(layers+1),3))
    k=0
    for i in np.arange(-layers-1,layers+1):        
        i=int(i)
        ind=atoms[:,2]==i*(ao/np.sqrt(3))
        ind2=atoms[:,0]<(max(atoms[:,0])/3)
        ind3=atoms[:,0]>(min(atoms[:,0])/3)
        ind4=atoms[:,1]<(max(atoms[:,1])/3)
        ind5=atoms[:,1]>(min(atoms[:,1])/3)
        ind=ind*ind2*ind3*ind4*ind5
        sub=atoms[ind,:]
        a=np.random.randint(0,np.shape(sub)[0])
        pos[k,:]=sub[a,:]
        k=k+1
    
    return pos