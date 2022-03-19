# -*- coding: utf-8 -*-
#Here, necessary Python packages will be imported and other written scripts will be called to perform the energy statistics calculations herein
import numpy as np                                                       #Python package used commonly for data manipulation involving arrays, matrices, etc
import gen_cell_MD as gc                                                 #Generated HCP, BCC, or FCC lattice based on inputted dimensions and lattice parameter
import Potential as pot                                                  #Performs potential energy calculations for both Faulted and nonFaulted states
import rdf_coord as rc                                                   #Generated Coordination Number (CN) and Radial Distribution Function (RDF) relations based on lattice parameter and cutoff distance
import rand_atoms_def as rad                                             #

import time                                                              #Time function to determine how quickly this script will run, used to check efficiency in edits --will remove once codes are fully completed

start = time.time()  # start time

#Generates RDF for desired unit cell for chosen systems
# ao=3.512
ao=3.5225                                                               #Inputted lattice parameter value in Angstroms  (if user chooses no value, what should be the assumed value?)
#Input Dimensions for crystal generation
xdim=70
ydim=70
zdim=70

rcut=6.5                                                                #Cutoff distance which will restrict coordination shells generated

comp=np.array([0.33,0.33,0.34])
# comp=np.array([0.50,0.50])

[atoms, per]=gc.gen_cell_FCC(ao,xdim,ydim,zdim)                         #Generates list of
#%%
[rdf_FCC,cn_FCC]=rc.rdf_coord(atoms[:,:3],rcut,np.array([0,0,0]))
print(cn_FCC)
#Declaring a .alloy file name to generate Cohesive Energy statistics
fname='FeNiCr.eam.alloy'
# fname='NiCo-lammps-2014.alloy'

#Reading the potential datasets declared above

[rrange,rhorange,rho,Fr,Pp]=pot.potential_read(fname)
#%% Calculate the lammps results analytically           !!Remove?
#[Anal_df_fcc,energies_fcc]=an.Analytical_Energy(atoms,Neighbors_fcc,fname)     !!Remove?

#Performs Cohesive Energy Calculations and Statistics based on chosen system
[form_E_fcc,test_fcc,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_FCC)
print(form_E_fcc)
print(covars)
print(form_E_fcc['E']['Std'])

# [Lammps_df,Lammps_atoms]=Lammps_stats(fname='atoms.0.lammps')             !!Remove?

#Faulted System Calculations Hereon...
#%%USF
[atoms, per]=gc.gen_cell_USF(ao,xdim,ydim,zdim)                        #Generates Fault structure
#%%
#atoms=copy.deepcopy(atoms_usf[:,0:3])          !!Remove?
pos=rad.rand_atoms_def(atoms,ao,rcut)

layer_USF=[]
layer_USF1=[]
rdf_USF=[]
cn_USF=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_USF1.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_USF.append(cn_temp)
    else:
            cn_USF.append(cn_temp)

form_E_usf_store=[]
covars_usf_store=[] 
test_usf_store=[]               
for i1 in range(np.shape(cn_USF)[0]):                
    [form_E_usf,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_USF[i1])
    covars_usf_store.append(covars)
    form_E_usf_store.append(form_E_usf)
    test_usf_store.append(test)
    
    
pe_store_usf=[]
std_pe_store_usf=[]

for i2 in range(np.shape(form_E_usf_store)[0]):
    pe_store_usf.append(form_E_usf_store[i2]['E']['Mean'])
    std_pe_store_usf.append(form_E_usf_store[i2]['E']['Std'])  
    
ratio=1/np.shape(std_pe_store_usf)[0]    
covar_usf=0
for i in range(int(np.shape(std_pe_store_usf)[0])): 
   covar_usf=covar_usf+(np.sum(test_fcc[0]*np.array(test_usf_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_usf_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_usf_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_usf_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_usf[i]))
sum_usf_std= np.sqrt(sum(np.asarray(std_pe_store_usf)**2+(pe_store_usf-np.average(pe_store_usf))**2)/np.shape(std_pe_store_usf)[0])
std_usf=np.sqrt(sum_usf_std**2+form_E_fcc['E']['Std']**2-2*covar_usf)
print(std_usf)
#%%ISF
[atoms, per]=gc.gen_cell_ISF(ao,xdim,ydim,zdim) 
#%%
#atoms=copy.deepcopy(atoms_ISF[:,0:3])
pos=rad.rand_atoms_def(atoms,ao,rcut)


layer_ISF=[]
layer_ISF1=[]
rdf_ISF=[]
cn_ISF=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_ISF1.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_ISF.append(cn_temp)
    else:
            cn_ISF.append(cn_temp)

form_E_isf_store=[]
covars_isf_store=[] 
test_isf_store=[]               
for i1 in range(np.shape(cn_ISF)[0]):                
    [form_E_isf,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_ISF[i1])
    covars_isf_store.append(covars)
    form_E_isf_store.append(form_E_isf)
    test_isf_store.append(test)
    
    
pe_store_isf=[]
std_pe_store_isf=[]

for i2 in range(np.shape(form_E_isf_store)[0]):
    pe_store_isf.append(form_E_isf_store[i2]['E']['Mean'])
    std_pe_store_isf.append(form_E_isf_store[i2]['E']['Std'])  
    
ratio=1/np.shape(std_pe_store_isf)[0]    
covar_isf=0
for i in range(int(np.shape(std_pe_store_isf)[0])): 
   covar_isf=covar_isf+(np.sum(test_fcc[0]*np.array(test_isf_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_isf_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_isf_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_isf_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_isf[i]))
sum_isf_std= np.sqrt(sum(np.asarray(std_pe_store_isf)**2+(pe_store_isf-np.average(pe_store_isf))**2)/np.shape(std_pe_store_isf)[0])
std_isf=np.sqrt(sum_isf_std**2+form_E_fcc['E']['Std']**2-2*covar_isf)
print(std_isf)

#%%UTF1
[atoms, per]=gc.gen_cell_UTF1(ao,xdim,ydim,zdim) 
#%%
#atoms=copy.deepcopy(atoms_isf[:,0:3])
pos=rad.rand_atoms_def(atoms,ao,rcut)


layer_UTF1=[]
layer_UTF11=[]
rdf_UTF1=[]
cn_UTF1=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_UTF11.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_UTF1.append(cn_temp)
    else:
            cn_UTF1.append(cn_temp)

form_E_utf1_store=[]
covars_utf1_store=[] 
test_utf1_store=[]               
for i1 in range(np.shape(cn_UTF1)[0]):                
    [form_E_utf1,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_UTF1[i1])
    covars_utf1_store.append(covars)
    form_E_utf1_store.append(form_E_utf1)
    test_utf1_store.append(test)
    
    
pe_store_utf1=[]
std_pe_store_utf1=[]

for i2 in range(np.shape(form_E_utf1_store)[0]):
    pe_store_utf1.append(form_E_utf1_store[i2]['E']['Mean'])
    std_pe_store_utf1.append(form_E_utf1_store[i2]['E']['Std'])  
    
ratio=1/np.shape(std_pe_store_utf1)[0]    
covar_utf1=0
for i in range(int(np.shape(std_pe_store_utf1)[0])): 
   covar_utf1=covar_utf1+(np.sum(test_fcc[0]*np.array(test_utf1_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_utf1_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_utf1_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_utf1_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_utf1[i]))
sum_utf1_std= np.sqrt(sum(np.asarray(std_pe_store_utf1)**2+(pe_store_utf1-np.average(pe_store_utf1))**2)/np.shape(std_pe_store_utf1)[0])
std_utf1=np.sqrt(sum_utf1_std**2+form_E_fcc['E']['Std']**2-2*covar_utf1)
print(std_utf1)
#%%ESF
[atoms, per]=gc.gen_cell_ESF(ao,xdim,ydim,zdim) 
#%%
#atoms=copy.deepcopy(atoms_isf[:,0:3])
pos=rad.rand_atoms_def(atoms,ao,rcut)


layer_ESF=[]
layer_ESF1=[]
rdf_ESF=[]
cn_ESF=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_ESF1.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_ESF.append(cn_temp)
    else:
            cn_ESF.append(cn_temp)

form_E_esf_store=[]
covars_esf_store=[] 
test_esf_store=[]               
for i1 in range(np.shape(cn_ESF)[0]):                
    [form_E_esf,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_ESF[i1])
    covars_esf_store.append(covars)
    form_E_esf_store.append(form_E_esf)
    test_esf_store.append(test)
    
    
pe_store_esf=[]
std_pe_store_esf=[]

for i2 in range(np.shape(form_E_esf_store)[0]):
    pe_store_esf.append(form_E_esf_store[i2]['E']['Mean'])
    std_pe_store_esf.append(form_E_esf_store[i2]['E']['Std'])  
    
ratio=1/np.shape(std_pe_store_esf)[0]    
covar_esf=0
for i in range(int(np.shape(std_pe_store_esf)[0])): 
   covar_esf=covar_esf+(np.sum(test_fcc[0]*np.array(test_esf_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_esf_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_esf_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_esf_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_esf[i]))
sum_esf_std= np.sqrt(sum(np.asarray(std_pe_store_esf)**2+(pe_store_esf-np.average(pe_store_esf))**2)/np.shape(std_pe_store_esf)[0])
std_esf=np.sqrt(sum_esf_std**2+form_E_fcc['E']['Std']**2-2*covar_esf)

print(std_esf)
#%%UTF2
[atoms, per]=gc.gen_cell_UTF2(ao,xdim,ydim,zdim) 
#%%
#atoms=copy.deepcopy(atoms_isf[:,0:3])
pos=rad.rand_atoms_def(atoms,ao,rcut)


layer_UTF2=[]
layer_UTF21=[]
rdf_UTF2=[]
cn_UTF2=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_UTF21.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_UTF2.append(cn_temp)
    else:
            cn_UTF2.append(cn_temp)

form_E_utf2_store=[]
covars_utf2_store=[] 
test_utf2_store=[]               
for i1 in range(np.shape(cn_UTF2)[0]):                
    [form_E_utf2,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_UTF2[i1])
    covars_utf2_store.append(covars)
    form_E_utf2_store.append(form_E_utf2)
    test_utf2_store.append(test)
    
    
pe_store_utf2=[]
std_pe_store_utf2=[]

for i2 in range(np.shape(form_E_utf2_store)[0]):
    pe_store_utf2.append(form_E_utf2_store[i2]['E']['Mean'])
    std_pe_store_utf2.append(form_E_utf2_store[i2]['E']['Std']) 
    
ratio=1/np.shape(std_pe_store_utf2)[0]    
covar_utf2=0
for i in range(int(np.shape(std_pe_store_utf2)[0])): 
   covar_utf2=covar_utf2+(np.sum(test_fcc[0]*np.array(test_utf2_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_utf2_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_utf2_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_utf2_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_utf2[i]))
sum_utf2_std= np.sqrt(sum(np.asarray(std_pe_store_utf2)**2+(pe_store_utf2-np.average(pe_store_utf2))**2)/np.shape(std_pe_store_utf2)[0])
std_utf2=np.sqrt(sum_utf2_std**2+form_E_fcc['E']['Std']**2-2*covar_utf2)
print(std_utf2)
#%%TF
[atoms, per]=gc.gen_cell_TF(ao,xdim,ydim,zdim) 
#%%
#atoms=copy.deepcopy(atoms_isf[:,0:3])
pos=rad.rand_atoms_def(atoms,ao,rcut)


layer_TF=[]
layer_TF1=[]
rdf_TF=[]
cn_TF=[]
pe_store=[]
Neighbors_store=[]
atoms_ID_store=[]
for i in np.arange(0,np.shape(pos)[0]):
    [rdf_temp,cn_temp]=rc.rdf_coord(atoms,rcut,pos[i,:])      
    layer_TF1.append([pos[i,2], round(pos[i,2]/(ao/np.sqrt(3)))])
    if np.shape(rdf_temp)[0]==np.shape(rdf_FCC)[0]:        
        if abs(np.sum(rdf_temp-rdf_FCC))>1e-5:  
                cn_TF.append(cn_temp)
    else:
            cn_TF.append(cn_temp)

form_E_tf_store=[]
covars_tf_store=[] 
test_tf_store=[]               
for i1 in range(np.shape(cn_TF)[0]):                
    [form_E_tf,test,covars]=pot.potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn_TF[i1])
    covars_tf_store.append(covars)
    form_E_tf_store.append(form_E_tf)
    test_tf_store.append(test)
    
    
pe_store_tf=[]
std_pe_store_tf=[]

for i2 in range(np.shape(form_E_tf_store)[0]):
    pe_store_tf.append(form_E_tf_store[i2]['E']['Mean'])
    std_pe_store_tf.append(form_E_tf_store[i2]['E']['Std'])  
    
ratio=1/np.shape(std_pe_store_tf)[0]    
covar_tf=0
for i in range(int(np.shape(std_pe_store_tf)[0])): 
   covar_tf=covar_tf+(np.sum(test_fcc[0]*np.array(test_tf_store[i][0])*comp*ratio)+np.sum(test_fcc[0]*np.array(test_tf_store[i][1])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_tf_store[i][0])*comp*ratio)+np.sum(test_fcc[1]*np.array(test_tf_store[i][1])*comp*ratio)-form_E_fcc['E']['Mean']*(ratio*pe_store_tf[i]))
sum_tf_std= np.sqrt(sum(np.asarray(std_pe_store_tf)**2+(pe_store_tf-np.average(pe_store_tf))**2)/np.shape(std_pe_store_tf)[0])
std_tf=np.sqrt(sum_tf_std**2+form_E_fcc['E']['Std']**2-2*covar_tf)
print(std_tf)

for i in range(1000000):
    pass

end = time.time()
print("Elapsed time is  {}".format(end - start))