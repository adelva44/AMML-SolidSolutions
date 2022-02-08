# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:10:09 2021

@author: Matt
"""

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
#%%
def potential_fetch(rrange,rhorange,rho,Fr,Pp,atoms,Neighbors,ind):
#%%
    import numpy as np
    
    rho_a=0
    Fr_a=0
    Pp_a=0    
    
    for i in np.arange(1,np.shape(Neighbors[1])[1]):    
        #print(i)
        r=Neighbors[0][ind,i]
        cent=int(atoms[ind,3]-1)
        pair=int(atoms[Neighbors[1][ind,i],3]-1)        
        rho_a=rho_a+np.interp(r,rrange,rho[:,pair])        
        Pp_a=Pp_a+(0.5*np.interp(r,rrange,Pp[:,cent,pair]))/r
        
    Fr_a=np.interp(rho_a,rhorange,Fr[:,cent])
 #%%
    return rho_a, Fr_a, Pp_a
#%%
def potential_stats(rrange,rhorange,rho,Fr,Pp,comp,cn,energies):
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
    form_E=np.zeros((2,6))
    
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
            #print(avg*cn[k,1])
            Pp_std_cn[k,j]=np.sum((Pp_cn[k,j,:]-avg)*(Pp_cn[k,j,:]-avg)*comp)*cn[k,1]                
  #%%  
    # test=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))    
    # for i in np.arange(0,np.shape(cn)[0]):
    #     for j in np.arange(0,np.shape(comp)[0]):
    #         avg=np.sum(Pp_cn[i,j,:]*comp)
    #         test[i,j]=np.sum((Pp_cn[i,j,:]-avg)*(Pp_cn[i,j,:]-avg)*comp)
    
    
    # cn_test=np.tile(cn[:,1],(np.shape(comp)[0],1))
    # cn_test=cn_test.T
    #%%
    # test2=np.sum(test*cn_test,axis=0)
    # test2=np.sqrt(test2)
    # # test2=0.5*test2
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
    
#%%
    import plot_pairpotentialdist as pltpot    
    pltpot.plot_pairpotendialdist(energies, Pp_std_avg2, Pp_std_cn2)
  #%%   
    # test=np.sum(comp*np.square(Pp_std_cn2))+np.sum(np.triu(aa,k=1)*cc)
    # test=np.sqrt(test)*0.5
    form_E[1,2]=Pp_std*0.5
    
    form_E[0,3]=F_bar+ sum(sum(Pp_bar))*0.5
   
    covar = (energies['F']*energies['Pp']).mean()-(energies['F'].mean()*energies['Pp'].mean()) 
    
    exp_FV=np.sum(comp*Fs*Pp_std_avg2*0.5) -F_bar*sum(sum(Pp_bar))*0.5
    covar2=exp_FV
    #print(comp*Fs*Pp_std_avg2*0.5)
    form_E[1,3]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar2)  
    form_E[0,4]=covar
    form_E[1,4]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar)
    form_E[0,5]=covar2
    form_E[1,5]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar2)
    form_E=pd.DataFrame(form_E, columns=["rho","F","Pp","E","E2","E_4"])
    form_E.index=["Mean", "Std"]
    #print(comp*Fs)
    #print(F_bar)
    #print(sum(Pp_std_avg2*0.5*comp)+sum(Fs*comp))
    test=Fs+Pp_std_avg2
    covars=np.array([covar,covar2])
    
#%%    
    return form_E, test,covars

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
            #print(avg*cn[k,1])
            Pp_std_cn[k,j]=np.sum((Pp_cn[k,j,:]-avg)*(Pp_cn[k,j,:]-avg)*comp)*cn[k,1]                
  #%%  
    # test=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))    
    # for i in np.arange(0,np.shape(cn)[0]):
    #     for j in np.arange(0,np.shape(comp)[0]):
    #         avg=np.sum(Pp_cn[i,j,:]*comp)
    #         test[i,j]=np.sum((Pp_cn[i,j,:]-avg)*(Pp_cn[i,j,:]-avg)*comp)
    
    
    # cn_test=np.tile(cn[:,1],(np.shape(comp)[0],1))
    # cn_test=cn_test.T
    #%%
    # test2=np.sum(test*cn_test,axis=0)
    # test2=np.sqrt(test2)
    # # test2=0.5*test2
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
    
#%%
    #import plot_pairpotentialdist as pltpot    
    #pltpot.plot_pairpotendialdist(energies, Pp_std_avg2, Pp_std_cn2)
  #%%   
    # test=np.sum(comp*np.square(Pp_std_cn2))+np.sum(np.triu(aa,k=1)*cc)
    # test=np.sqrt(test)*0.5
    form_E[1,2]=Pp_std*0.5
    
    form_E[0,3]=F_bar+ sum(sum(Pp_bar))*0.5
   
    #covar = (energies['F']*energies['Pp']).mean()-(energies['F'].mean()*energies['Pp'].mean()) 
    
    exp_FV=np.sum(comp*Fs*Pp_std_avg2*0.5) -F_bar*sum(sum(Pp_bar))*0.5
    covar2=exp_FV
    
    form_E[1,3]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar2)  
    #form_E[0,4]=covar
    #form_E[1,4]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar)
    #form_E[0,4]=covar2
    #form_E[1,4]=np.sqrt(F_std*F_std+(Pp_std*Pp_std/4)+2*covar2)
    form_E=pd.DataFrame(form_E, columns=["rho","F","Pp","E"])
    form_E.index=["Mean", "Std"]
    
    covars=np.array([covar2])
    test=[Fs,Pp_std_avg2*0.5]
    #print(Pp_std_avg2*0.5)
#%%    
    return form_E, test,covars
# def potential_avg_back(rrange,rhorange,rho,Fr,Pp,comp,cn):
# #%%
#     import numpy as np
#     import pandas as pd
    
#     rho_bar=0
#     #consts_rho=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))
#     #consts_F=np.zeros((np.shape(comp)[0]))
    
#     #var_rho_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))
#     #var_F=np.zeros((np.shape(comp)[0]))
    
#     #cov_chem=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))    
    
#     F_bar=0
#     Pp_bar=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))
#     Pp_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0],np.shape(comp)[0]))
       
#     for j in np.arange(0,np.shape(comp)[0]):
#         for i in np.arange(0,np.shape(cn)[0]):    
#             c=comp[j]
#             r=cn[i,0]
#             m=cn[i,1]
#             Rho=np.interp(r,rrange,rho[:,j])
#             rho_bar=rho_bar+c*m*Rho
#             #print(Rho)
#      #       consts_rho[i,j]=m*Rho
#       #      var_rho_cn[i,j]=m*m*Rho*Rho*c*(1-c)
    
#     for j in np.arange(0,np.shape(comp)[0]):
#         c=comp[j]
#         F_bar=F_bar+np.interp(rho_bar,rhorange,Fr[:,j])*c
#      #   consts_F[j]=np.interp(rho_bar,rhorange,Fr[:,j])
#       #  var_F[j]=consts_F[j]*consts_F[j]*c*(1-c)
    
#     #for i in np.arange(0,np.shape(comp)[0]):
#      #   for j in np.arange(0,np.shape(comp)[0]):
#       #      cov_chem[i,j]=-comp[i]*comp[j]
    
#     # lists=[]    
#     # for i in np.arange(0,np.shape(comp)[0]):
#     #     for j in np.arange(0,np.shape(comp)[0]):
#     #         if j <= i:
#     #             continue
#     #         else:
#     #             lists.append([i,j])
#     #             m=m+1
    
#     # covars_rho_cn=np.zeros((np.shape(cn)[0],np.shape(lists)[0]))
#     # covars_F=np.zeros((np.shape(lists)[0]))
    
#     # for j in np.arange(0,np.shape(lists)[0]):
#     #     covars_rho_cn[:,j]=consts_rho[:,lists[j][0]]*consts_rho[:,lists[j][1]]*cov_chem[lists[j][0],lists[j][1]]
#     #     covars_F[j]=consts_F[lists[j][0]]*consts_F[lists[j][1]]*cov_chem[lists[j][0],lists[j][1]]
        
#     # temp_rho=np.hstack((var_rho_cn,2*covars_rho_cn))
#     # temp_F=np.hstack((var_F,2*covars_F))
    
#     # var_rho_cn2=np.sum(temp_rho,axis=1)
#     # Var_F=np.sum(temp_F)
    
#     # Var_rho=sum(var_rho_cn2)    
    
#     # stdev_rho=np.sqrt(Var_rho)
#     # stdev_F=np.sqrt(Var_F)
            
#    #%% 
#     for i in np.arange(0,np.shape(Pp_bar)[0]):
#         for j in np.arange(0,np.shape(Pp_bar)[1]):
#             Pp_ind=0
#             print(i,j)
#             for k in np.arange(0,np.shape(cn)[0]):
#                 Pp_cn[k,i,j]=comp[i]*comp[j]*cn[k,1]*0.5*np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]
#                 Pp_ind=Pp_ind+comp[i]*comp[j]*cn[k,1]*0.5*np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]
            
#             Pp_bar[i,j]=Pp_ind
    
#     form_E=np.zeros((1,4))
    
#     form_E[0,0]=rho_bar
        
#     #form_E[1,0]=rho_bar
    
#     form_E[0,1]=F_bar
#     #form_E[1,1]=F_bar
    
#     form_E[0,2]=sum(sum(Pp_bar))
#     #form_E[1,2]=sum(sum(Pp_bar))
    
#     form_E[0,3]=F_bar+ sum(sum(Pp_bar))
#     #form_E[1,3]=F_bar+ sum(sum(Pp_bar))
    
#     form_E=pd.DataFrame(form_E, columns=["rho","F","Pp","E"])
#     #form_E.index=["Mean", "Std"]
#     form_E.index=["Mean"]
#  #%%   
    
#     return form_E

# def potential_F_std(cn,energies,rrange,form_E,rho,rhorange,Fr, comp, Neighbors, atoms, Pp_cn):
#     #%%
#     import numpy as np
#     rho_bar=form_E.at["Mean","rho"]
#     atomtypes=np.array([energies["type"]]).T
#     atomtypes=atomtypes[:,0]
    
#     types=np.zeros((np.shape(atomtypes)[0],int(max(atomtypes))))
    
#     for i in np.arange(0,int(max(atomtypes))):
#         ind=atomtypes==i+1
#         types[ind,i]=1
    
#     rhos=np.zeros((np.shape(cn)[0],int(max(atomtypes))))
    
#     for i in np.arange(0,np.shape(cn)[0]):
#         for j in np.arange(0,np.shape(rhos)[1]):
#             rhos[i,j]=np.interp(cn[i,0],rrange,rho[:,j])
    
#     rho_stat=types*rho_bar
    
#     rho_types=types*rhos
    
#     stats_rho=np.zeros((2,int(max(atomtypes))))
#     stats_F=np.zeros((2,int(max(atomtypes))))
    
#     for i in np.arange(0,int(max(atomtypes))):
#         stats_rho[0,i]=np.average(rho_stat[:,i])
#         stats_rho[1,i]=np.var(rho_stat[:,i])
    
#     Fs=np.zeros((int(max(atomtypes))))
    
#     for i in np.arange(0,np.shape(Fs)[0]):
#         Fs[i]=np.interp(rho_bar,rhorange,Fr[:,i])
    
#     F=types*Fs
#     F_comb=np.sum(F,axis=1)
    
#     for i in np.arange(0,int(max(atomtypes))):
#         stats_F[0,i]=np.average(F[:,i])
#         stats_F[1,i]=np.var(F[:,i])
    
#     lists=[]    
#     for i in np.arange(0,int(max(atomtypes))):
#         for j in np.arange(0,int(max(atomtypes))):
#             if j < i:
#                 continue
#             else:
#                 lists.append([i,j])
    
#     stats_F_comb=np.zeros((2,1))
#     stats_F_comb[0,0]=np.average(F_comb)
#     stats_F_comb[1,0]=np.std(F_comb)
#     std_F=np.sqrt(np.sum((Fs-stats_F_comb[0,0])*(Fs-stats_F_comb[0,0])*comp))
#     # cov_chem=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))    
    
#     # for i in np.arange(0,np.shape(comp)[0]):
#     #     for j in np.arange(0,np.shape(comp)[0]):
#     #         cov_chem[i,j]=-comp[i]*comp[j]
            
#     # covars_F=np.zeros((np.shape(lists)[0]))
#     # for j in np.arange(0,np.shape(lists)[0]):
#     #     covars_F[j]=Fs[lists[j][0]]*Fs[lists[j][1]]*cov_chem[lists[j][0],lists[j][1]]
    
#     # temp_F=np.hstack((stats_F[1,:],2*covars_F))
#     # Var_F=np.sum(temp_F)
#     # std_F=np.sqrt(Var_F)
#     # stats_F_comb[1,0]=np.sqrt(stats_F_comb[1,0])
#     rhos2=np.sum(rho_types,axis=1)
    
#     bonds=np.zeros((int(np.shape(atomtypes)[0]*cn[0,1]),sum(np.arange(0,np.shape(comp)[0]+1))))
    
#     for i in np.arange(0,np.shape(atomtypes)[0]):
#         for j in np.arange(1,np.shape(Neighbors[1][:,1:])[1]+1):
#             #print(j)
#             #print(int(i*cn[0,1]+j-1))
#             if atoms[Neighbors[1][i,0],3]==1 and atoms[Neighbors[1][i,j],3]==1:
#                 #print(1)
#                 bonds[int(i*cn[0,1]+j-1),0]=Pp_cn[0,0,0]
#             if atoms[Neighbors[1][i,0],3]==1 and atoms[Neighbors[1][i,j],3]==2:
#                 #print(2)
#                 bonds[int(i*cn[0,1]+j-1),1]=Pp_cn[0,0,1]
#             if atoms[Neighbors[1][i,0],3]==2 and atoms[Neighbors[1][i,j],3]==1:
#                 #print(2)
#                 bonds[int(i*cn[0,1]+j-1),1]=Pp_cn[0,0,1]
#             if atoms[Neighbors[1][i,0],3]==2 and atoms[Neighbors[1][i,j],3]==2:
#                 #print(3)
#                 bonds[int(i*cn[0,1]+j-1),2]=Pp_cn[0,1,1]
            
#     bonds_comb=np.sum(bonds,axis=1)
#     bonds_average=np.average(bonds_comb)*0.5*cn[0,1]
#     bonds_std=np.std(bonds_comb*0.5)*np.sqrt(12)
# #%%    
#     from numpy.random import default_rng
#     rng = default_rng()
#     ints=10000
#     samps=np.zeros((ints))
#     for i in np.arange(0,ints):
#         a=rng.choice(np.shape(bonds_comb)[0],12)
#         b=bonds_comb[a]
#         samps[i]=sum(b)*0.5
    
#     samps_avg=np.average(samps)
#     samps_std=np.std(samps)
    
#     ints2=int(np.shape(bonds_comb)[0]/12)
#     samps2=np.zeros((ints2))
#     k=12
#     for i in np.arange(0,np.shape(atoms)[0]):
#         samps2[i]=sum(bonds_comb[int(i*k):int(i*k+12)])
        
#     samps2=samps2*0.5
#     samps2_avg=np.average(samps2)
#     samps2_std=np.std(samps2)
# #%%    
#     bond_counts=np.zeros((2,2))
#     bond_totals=0
#     for i in np.arange(0,np.shape(atoms)[0]):
#         for j in np.arange(1,np.shape(Neighbors[1][:,1:])[1]+1):
#             cent=int(atoms[Neighbors[1][i,0],3]-1)
#             pair=int(atoms[Neighbors[1][i,j],3]-1)
#             bond_counts[cent,pair]=bond_counts[cent,pair]+1
#             bond_totals=bond_totals+1
            
#     #%%
#     import pandas as pd
#     temp2=pd.DataFrame.to_numpy(energies)
#     Pp_types=[]
#     for i in np.arange(0,np.shape(comp)[0]):
#         ind=temp2[:,1]==int(i+1)
#         Pp_types.append(np.array([temp2[ind,4]]).T)
        
#     #%%
#     return
#%%
# import numpy as np
# F_approx=np.zeros((np.shape(atoms)[0]),np.shape(comp))
# rho_bar3=Anal_df['rho']['Mean']
# #%%
# for i in np.arange(0,np.shape(atoms)[0]):
#     ind=energies[i,0]

#%%
# E_std=Anal_df['E']['Std']
# F_std=Anal_df['F']['Std']
# Pp_std=Anal_df['Pp']['Std']

# cov=(E_std*E_std-F_std*F_std-Pp_std*Pp_std)/2

# exp_xy=(energies['F']*energies['Pp']).mean()
# exp_x=Anal_df['F']['Mean']

# exp_y=Anal_df['Pp']['Mean']

# cov2=exp_xy-(exp_x*exp_y)

# corr=cov2/(F_std*Pp_std)

# E_std_check=F_std*F_std+Pp_std*Pp_std+2*corr*F_std*Pp_std

# E_std_check=np.sqrt(E_std_check)