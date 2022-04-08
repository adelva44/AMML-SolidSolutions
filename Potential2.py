#the version of this file is to add annotations to the code
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:10:09 2021

@author: Matt
"""

#This function extracts the Interatomic Potential information from Embedded-Atom Method style file and sets that information to corresponding variables. More information about EAM style and variable names can be read here: https://docs.lammps.org/pair_eam.html
import numpy as np


def potential_read(fname):

    import numpy as np
    #fname='NiCo-lammps-2014.alloy'
    #fname='FeNiCr.eam.alloy'
    with open(fname) as f:
        lines=f.readlines()
    chem=lines[3]
    chem=int(str.split(chem)[0])                        #Number of atomic elements in system (e.g. NiCo will be 2 & FeNiCr will be 3)
    Nrho=int(str.split(lines[4])[0])                    #Number of atomic electron densities in file list
    drho=float(str.split(lines[4])[1])                  #Spacing in density for the rho array
    Nr=int(str.split(lines[4])[2])                      #Number of distances in file list
    dr=float(str.split(lines[4])[3])                    #Spacing in distance for the r array in Angstroms
    cols=np.shape(str.split(lines[6]))[0]               #Counts how many columns there are in the tabulated data

    Nrho_rows=int(Nrho/cols)
    Nr_rows=int(Nr/cols)

# Generates empty arrays for atomic electron densities, Embedding energies, and Pair potential interactions respectively that will be calculated based on parsed data from the file below
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

    rrange=np.arange(0,Nr)*dr                          #Array of the number of distances times the distance space
    rhorange=np.arange(0,Nrho)*drho                    #Array of the number of atomic electron densities times the density spacing
#%%
    return rrange,rhorange,rho,Fr,Pp                   #Generated values for atomic electron density, Embedding energy function and Pair potential


#Performs Energy calcualtions based on information extracted from file above.
def potential_stats2(rrange,rhorange,rho,Fr,Pp,comp,cn):
#%%
    import numpy as np
    import pandas as pd

#Rho Calculation
    rho_bar=0
    rho_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))

    for j in np.arange(0,np.shape(comp)[0]):
        for i in np.arange(0,np.shape(cn)[0]):
            c = comp[j]
            r=cn[i,0]
            m=cn[i,1]
            Rho=np.interp(r,rrange,rho[:,j])
            rho_bar=rho_bar+c*m*Rho                         #Equation 2 in Manuscript (Average per Atom Charge Density)
            rho_cn[i,j]=np.interp(r,rrange,rho[:,j])

#Calculates standard deviation for rho (?)
    rho_std=np.zeros((np.shape(rho_cn)[0]))
    for i in np.arange(0,np.shape(rho_std)[0]):
        rho_std[i] = np.sqrt(np.sum((rho_cn[i, :] - np.sum(rho_cn[i, :] * comp)) * (rho_cn[i, :] - np.sum(rho_cn[i, :] * comp)) * comp) * cn[i, 1])

#Embedding Energy Calculation
    F_bar=0
    Fs=np.zeros((np.shape(comp)[0]))

    for j in np.arange(0,np.shape(comp)[0]):
        c=comp[j]
        Fs[j]=np.interp(rho_bar,rhorange,Fr[:,j])           #F_x (Embedding Energy of Element X)
        F_bar = F_bar + Fs[j]*c                             #Located in Page 9 of Manuscript (Average per Atom Embedding Energy)

    F_std=np.sqrt(np.sum(((Fs-F_bar)**2)*comp))             #Equation 3 in Manuscript (Embeddeding Energy Standard Deviation)

#Pair Interaction Energy Calculation (referred to as V in the manuscript)
    Pp_bar=np.zeros((np.shape(comp)[0],np.shape(comp)[0]))
    Pp_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0],np.shape(comp)[0]))

    for i in np.arange(0,np.shape(Pp_bar)[0]):
        for j in np.arange(0,np.shape(Pp_bar)[1]):
            Pp_ind=0
            for k in np.arange(0,np.shape(cn)[0]):
                Pp_cn[k,i,j]= np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]                               #V_eta_xy (Pair interaction energies of solutes at distances corresponding to the various coordination shells)
                Pp_ind= Pp_ind + comp[i]*comp[j]*cn[k,1]*np.interp(cn[k,0],rrange,Pp[:,i,j])/cn[k,0]    #V_bar
            Pp_bar[i,j]=Pp_ind

    #Will be used to store variables
    form_E=np.zeros((2,4))
    form_E[0,0]=rho_bar                         #Mean per Atom Charge Density
    form_E[1,0]=np.sqrt(np.sum(rho_std**2))     #Standard Deviation per Atom Charge Density
    form_E[0,1]=F_bar                           #Mean per Atom Embedding Energy
    form_E[1,1]=F_std                           #Standard Deviation Embeddeding Energy
    form_E[0,2]=sum(sum(Pp_bar))*0.5            #Mean V

    Pp_std_cn=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))
    Pp_std_avg=np.zeros((np.shape(cn)[0],np.shape(comp)[0]))

    for k in np.arange(0,np.shape(cn)[0]):
        for j in np.arange(0,np.shape(comp)[0]):
            avg=np.sum(Pp_cn[k,j,:]*comp)                                                                       #V_bar_eta_x Located in Page 26 in Manuscript (Average Interaction Energy for a Solute X in Coordination Shell eta)
            Pp_std_avg[k,j]= avg*cn[k,1]                                                                        #V_bar_eta_x * N_eta
            Pp_std_cn[k,j]= np.sum((Pp_cn[k,j,:]-avg)*(Pp_cn[k,j,:]-avg)*comp)*cn[k,1]


    Pp_std_avg3 = np.zeros((np.shape(cn)[0], np.shape(comp)[0]))


    Pp_std_cn2 = np.sqrt(np.sum(Pp_std_cn,axis=0))                  #Equation A1 in Manuscript (Solute-level Interaction Energy Standard Deviation)
    Pp_std_avg2 = np.sum(Pp_std_avg,axis=0)                         #V_bar_x, Equation A2 in the Manuscript (Solute-level Interaction Energy Average)


    bb = np.zeros((np.shape(comp)[0], np.shape(comp)[0]))
    for i in np.arange(0, np.shape(bb)[0]):
        for j in np.arange(0, np.shape(bb)[0]):
            if i <= j:
                continue
            bb[i,j]= comp[i]*comp[j]*np.square(Pp_std_avg2[i] - Pp_std_avg2[j])

    Pp_std = np.sqrt(np.sum(comp*(np.square(Pp_std_cn2)))+sum(sum(bb)))                                     #Equation 4, Standard deviation of the Pair Interaction energy
    # print('this is E4',Pp_std)
    A3 = np.sqrt(np.sum(comp * (np.square(Pp_std_cn2) + np.square((Pp_std_avg2 - sum(sum(Pp_bar)))))))      #A3, replace with E4
    # print('this is A3',A3)

    print('E4-A3',Pp_std-A3)

    form_E[1,2]=Pp_std*0.5                                                  #1/2 Standard Deviation of Pair Interaction Energy (1/2 is to avoid double counting of interaction energies)

    form_E[0,3]=F_bar+ sum(sum(Pp_bar))*0.5                                 #Equation 1 in Manuscript (Average per atom binding energy)

    covar2 = np.sum(comp*Fs*Pp_std_avg2*0.5) - F_bar*sum(sum(Pp_bar))*0.5   #Equation 6 in Manuscript (cov(F, 1/2V))

    form_E[1,3] = np.sqrt(F_std**2 + (Pp_std/2)**2 + 2*covar2)              #Equation 5 in Manuscript (Standard Deviation of the per atom energy)
    form_E=pd.DataFrame(form_E, columns=["rho","F","Pp","E"])
    form_E.index=["Mean", "Std"]

    covars=np.array([covar2])
    test=[Fs,Pp_std_avg2*0.5]
#%%
    return form_E, test,covars              #form_E returns a tabulated version of the Mean and Standard Deviation for atomic electron density rho, Embedding Energy F (as a function of rho), the pair potential interaction Pp, and the total Energy E; test stores values that are used in covariance calculations needed for final GPFE calculation; covars returns the covariance
