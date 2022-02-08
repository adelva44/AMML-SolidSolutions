# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:45:24 2021

@author: Matt
"""
#This script will return the single atom rdf and coordination number.
#rc is the cutoff radius
#pos is the position of the single atom


def rdf_coord(atoms,rc,pos):
  
    import numpy as np
            
    pos=np.ones((np.shape(atoms)[0],3))*pos
        
    atoms=atoms-pos
    dists=np.zeros((np.shape(atoms)[0]))
    
    for i in np.arange(0,np.shape(atoms)[0]):
        dists[i]=np.sqrt(sum(atoms[i,:]*atoms[i,:]))
        
    dists=np.sort(dists)
    ind=dists<=rc
    neigh=dists[ind]
    neigh=neigh[1:]    
    neigh=np.round(neigh,5)        
    uni=np.unique(neigh)
    
    cn=np.zeros((np.shape(uni)[0],2))
    rdf=np.zeros((np.shape(cn)[0],2))
    
    for i in np.arange(0,np.shape(uni)[0]):
        ind=neigh==uni[i]
        cn[i,0]=uni[i]
        cn[i,1]=sum(ind)
        
    rdf[:,0]=cn[:,0]/cn[0,0]
    rdf[:,1]=(cn[0,1]/cn[:,1])*(rdf[:,0]*rdf[:,0])
    
    rdf[:,1]=np.round(rdf[:,1],4)
  
    return rdf, cn

