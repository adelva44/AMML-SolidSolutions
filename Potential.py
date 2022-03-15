#the version of this file is to add annotations to the code
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:10:09 2021

@author: Matt
"""

# Using LAAMPS, this function extracts the Interatomic Potential information from Embedded-Atom Method style file and sets that information to corresponding variables
def potential_read(fname='NiCo-lammps-2014.alloy'):
#%%
    import numpy as np
    #fname='NiCo-lammps-2014.alloy'
    #fname='FeNiCr.eam.alloy'
    with open(fname) as f:
        lines=f.readlines()
    chem=lines[3]
    chem=int(str.split(chem)[0])
    Nrho=int(str.split(lines[4])[0])
    drho=float(str.split(lines[4])[1])
    Nr=int(str.split(lines[4])[2])
    dr=float(str.split(lines[4])[3])
    cols=np.shape(str.split(lines[6]))[0]
    
    Nrho_rows=int(Nrho/cols)
    Nr_rows=int(Nr/cols)
    
    rho=np.zeros((Nr,chem))
    Fr=np.zeros((Nrho,chem))
    Pp=np.zeros((Nr,sum(np.arange(0,chem+1))))

    k=6
    for i in np.arange(0,chem):
        m=0
        for j in np.arange(0,Nrho):
            Fr[j,i]=float(str.split(lines[i*Nrho_rows+k+i])[m])            
            m=m+1                
            if m==cols:
                m=0
                k=k+1    
       
    k=6
    for i in np.arange(0,chem):
        m=0
        for j in np.arange(0,Nr):            
            rho[j,i]=float(str.split(lines[(i+1)*Nr_rows+k+i])[m])                                
            m=m+1                
            if m==cols:
                m=0
                k=k+1
    
    
    k=chem*Nr_rows+chem*Nrho_rows+5+chem  
 
    for i in np.arange(0,np.shape(Pp)[1]):
        m=0
        for j in np.arange(0,Nr):
            Pp[j,i]=float(str.split(lines[k])[m])
            m=m+1                
            if m==cols:
                m=0
                k=k+1
  
    lists=[]
    m=0
    for i in np.arange(0,chem):
        for j in np.arange(0,chem):
            if i < j:
                continue
            else:
                lists.append([i,j,m])
                m=m+1
 
    Pp_new=np.zeros((Nr,chem,chem))  
    for i in np.arange(0,np.shape(lists)[0]):
        ind1=lists[i][0]
        ind2=lists[i][1]
        ind3=lists[i][2]
        Pp_new[:,ind1,ind2]=Pp[:,ind3]
        
        if ind2<ind1:
            Pp_new[:,ind2,ind1]=Pp[:,ind3]
    
    Pp=Pp_new
    
    rrange=np.arange(0,Nr)*dr
    rhorange=np.arange(0,Nrho)*drho
#%%    
    return rrange,rhorange,rho,Fr,Pp

def potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn):
#%%
    import numpy as np
    import pandas as pd
    
    rho_bar=0    
    rho_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))  
    
    for j in np.arange(0,np.shape(comp)[0]):
        for i in np.arange(0,np.shape(cn)[0]):    
            c=comp[j]
            r=cn[i,0]
            m=cn[i,1]
            Rho=np.interp(r,rrange,rho[:,j])
            rho_bar=rho_bar+c*m*Rho 
            rho_cn[i,j]=np.interp(r,rrange,rho[:,j])
    
    rho_std=np.zeros((np.shape(rho_cn)[0]))
 #%%  
    for i in np.arange(0,np.shape(rho_std)[0]):
        rho_std[i]=np.sum((rho_cn[i,:]-np.sum(rho_cn[i,:]*comp))*(rho_cn[i,:]-np.sum(rho_cn[i,:]*comp))*comp)*cn[i,1]
        rho_std[i]=np.sqrt(rho_std[i])
 #%%
    F_bar=0
    Fs=np.zeros((np.shape(comp)[0]))
            
    for j in np.arange(0,np.shape(comp)[0]):
        c=comp[j]
        Fs[j]=np.interp(rho_bar,rhorange,Fr[:,j])
        F_bar=F_bar+np.interp(rho_bar,rhorange,Fr[:,j])*c 
        
    F_std=np.sqrt(np.sum((Fs-F_bar)*(Fs-F_bar)*comp))
    
    Pp_bar=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))
    
    Pp_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0],np.shape(comp)[0]))
    
    for i in np.arange(0,np.shape(Pp_bar)[0]):
        for j in np.arange(0,np.shape(Pp_bar)[1]):
            Pp_ind=0
            for k in np.arange(0,np.shape(cn)[0]):
                Pp_cn[k,i,j]=np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]
                Pp_ind=Pp_ind+comp[i]*comp[j]*cn[k,1]*np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]
            Pp_bar[i,j]=Pp_ind

     #%%           
    form_E=np.zeros((2,4))
    
    form_E[0,0]=rho_bar
        
    form_E[1,0]=np.sqrt(np.sum(rho_std*rho_std))
    
    form_E[0,1]=F_bar
    form_E[1,1]=F_std
    
    form_E[0,2]=sum(sum(Pp_bar))*0.5
    
    Pp_std_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))
    Pp_std_avg=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))
    for k in np.arange(0,np.shape(cn)[0]):
        for j in np.arange(0,np.shape(comp)[0]):
            avg=np.sum(Pp_cn[k,j,:]*comp)
            Pp_std_avg[k,j]=avg*cn[k,1]
            Pp_std_cn[k,j]=np.sum((Pp_cn[k,j,:]-avg)*(Pp_cn[k,j,:]-avg)*comp)*cn[k,1]                
  #%%  

    #%%
    Pp_std_cn2=np.sum(Pp_std_cn,axis=0)
    Pp_std_cn2=np.sqrt(Pp_std_cn2)
    
    Pp_std_avg2=np.sum(Pp_std_avg,axis=0)    
    
    Pp_std=np.sum(comp*(np.square(Pp_std_cn2)+np.square((Pp_std_avg2-sum(sum(Pp_bar))))))
    Pp_std=np.sqrt(Pp_std)
    
    aa=np.matmul(np.array([comp]).T,np.array([comp]))
    bb=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))
    
    for i in np.arange(0,np.shape(bb)[0]):
        for j in np.arange(0,np.shape(bb)[1]):
            bb[i,j]=Pp_std_avg2[i]-Pp_std_avg2[j]
    
    cc=np.square(bb)

    form_E[1,2]=Pp_std*0.5
    
    form_E[0,3]=F_bar+ sum(sum(Pp_bar))*0.5

    exp_FV=np.sum(comp*Fs*Pp_std_avg2*0.5) -F_bar*sum(sum(Pp_bar))*0.5
    covar2=exp_FV
    
    form_E[1,3]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar2)
    form_E=pd.DataFrame(form_E, columns=["rho","F","Pp","E"])
    form_E.index=["Mean", "Std"]
    
    covars=np.array([covar2])
    test=[Fs,Pp_std_avg2*0.5]
#%%    
    return form_E, test,covars
