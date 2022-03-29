#the version of this file is for Functions branch
# -*- coding: utf-8  -*-
"""
Created on Sun Mar 15 16:28:54 2020

@author: Matt
"""

import numpy as np
import math

#This function builds a perfect FCC crystal with a lattice parameter of ao
#The perfect crystal has periodic boundary conditions and is approximately
#xdim x ydim x zdim (in Angstroms) in size
#The crystal axes are as follows:
#x = 110
#y = 112
#z = 111
def gen_cell_FCC(ao=None, xdim=None, ydim=None, zdim=None):
    xo=ao/math.sqrt(2) #X repeat distance [ ]
    yo=3*ao/(2*math.sqrt(6)) #Y repeat distance
    zo=ao/math.sqrt(3) # Z layer distance
    
    #making sure the dimensions are integral numbers of the lattice vectors above
    xint=math.ceil(xdim/xo) 
    yint=math.ceil(ydim/yo)
    zint=math.ceil(zdim/zo)
    
    #Here we are ensuring periodicity of the system by making sure that the
    #overall dimensions align to periodic stacking along each axis
    mod=zint%3
    if mod==0:
        zint=zint
    elif mod==1:
        zint=zint-1
    elif mod==2:
        zint=zint+1
    
    mod=yint%2
    if mod==0:
        yint=yint
    else:
        yint=yint+1

    atoms=np.array([[0,0,0]])    
    
    #Centering the system cell on 0,0,0 (more or less)
    test=zint%2
    if test==0:
        aa=-zint/2
        bb=zint/2
    else:
        aa=math.floor(-zint/2)
        bb=math.floor(zint/2)

    cc=-yint/2
    dd=yint/2

    test=xint%2

    if test==0:
        ee=-xint/2
        ff=xint/2
    else:
        ee=math.floor(-xint/2)
        ff=math.floor(xint/2)
    
    #Placing the atoms
    for i in np.arange(aa,bb):
        bit2=i%3
        if bit2==0:
            shiftz=0
        elif bit2==1:
            shiftz=ao/math.sqrt(6)
        elif bit2==2:
            shiftz=2*ao/math.sqrt(6)
        
        for j in np.arange(cc,dd):
            bit=j%2
            if bit==0:
                shifty=0
            else: 
                shifty=xo/2
        
            for k in np.arange(ee,ff):
                if j==0 and k ==0 and i==0:
                    pass
                else:
                    temp=np.array([[k*xo+shifty, j*yo+shiftz, i*zo]])
                    atoms=np.vstack((atoms,temp))
    
    #Assigning the periodic dimensions
    per=np.array([[ee*xo, cc*yo, aa*zo],[ff*xo, dd*yo,bb*zo]])
    
    #Making sure that any atoms that fall outside the periodic X boundary are wrapped
    #back into the cell
    ind=np.where(atoms[:,1]>per[1,1])
    ind=ind[0]

    atoms[ind,1]=atoms[ind,1]-(per[1,1]-per[0,1])
    
    return atoms, per

#This function builds a perfect HCP crystal with a lattice parameter of ao
#The perfect crystal has periodic boundary conditions and is approximately
#xdim x ydim x zdim (in Angstroms) in size
#The crystal axes are as follows:
#x = 110 (FCC direction)
#y = 112 (FCC direction)
#z = 0001
def gen_cell_HCP(ao=None, xdim=None, ydim=None, zdim=None):
    xo=ao/math.sqrt(2)
    yo=3*ao/(2*math.sqrt(6))
    zo=ao/math.sqrt(3)

    xint=math.ceil(xdim/xo)
    yint=math.ceil(ydim/yo)
    zint=math.ceil(zdim/zo)
    
    mod=zint%2
    if mod==0:
        zint=zint
    elif mod==1:
        zint=zint-1
    
    
    mod=yint%2
    if mod==0:
        yint=yint
    else:
        yint=yint+1

    atoms=np.array([[0,0,0]])    

    test=zint%2
    if test==0:
        aa=-zint/2
        bb=zint/2
    else:
        aa=math.floor(-zint/2)
        bb=math.floor(zint/2)

    cc=-yint/2
    dd=yint/2

    test=xint%2

    if test==0:
        ee=-xint/2
        ff=xint/2
    else:
        ee=math.floor(-xint/2)
        ff=math.floor(xint/2)
    
    #This block holds only difference between this code and the FCC code.
    #here we ensure ABABAB stacking instead of ABCABCABC stacking
    for i in np.arange(aa,bb):
        bit2=i%2
        if bit2==0:
            shiftz=0
        elif bit2==1:
            shiftz=ao/math.sqrt(6)        
        
        for j in np.arange(cc,dd):
            bit=j%2
            if bit==0:
                shifty=0
            else: 
                shifty=xo/2
        
            for k in np.arange(ee,ff):
                if j==0 and k ==0 and i==0:
                   pass
                else:
                    temp=np.array([[k*xo+shifty, j*yo+shiftz, i*zo]])
                    atoms=np.vstack((atoms,temp))
                
    per=np.array([[ee*xo, cc*yo, aa*zo],[ff*xo, dd*yo,bb*zo]])

    ind=np.where(atoms[:,1]>per[1,1])
    ind=ind[0]

    atoms[ind,1]=atoms[ind,1]-(per[1,1]-per[0,1])
    
    return atoms, per

def gen_cell_BCC(ao=None, xdim=None, ydim=None, zdim=None):
    from ase.build import bulk
    from ase.calculators.lammps import Prism, convert

    ao = 3.5225
    xdim = 20
    ydim = 20
    zdim = 20

    a0 = bulk('Fe', 'bcc', cubic=True, a=ao)
    a0 *= [20, 20, 20]

    atoms = a0.get_positions()
    # rcut=6.72488400000000
    rcut = 20

    per = np.stack((np.array([0, 0, 0]), np.diagonal(np.asarray(a0.get_cell()))), 1)
    p = Prism(a0.get_cell())

    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance", "ASE", 'metal')
    per = np.array([[0, xhi], [0, yhi], [0, zhi]])
    atoms = gen_chem(atoms, types=2, comp=comp)

    return atoms, per

def gen_cell_USF(ao=None, xdim=None, ydim=None, zdim=None):
    #Here we are generating the atom coordinates of an USF defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the FCC cell as we normally would do
    [atoms, per]=gen_cell_FCC(ao, xdim, ydim, zdim)
    ind=atoms[:,2]>=0 #Selecting the part of the crystal with z >=0
    atoms[ind,1]=atoms[ind,1]+((ao/np.sqrt(6))/2) #adding the USF lattice shear to the selected half
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_cell_ISF(ao=None, xdim=None, ydim=None, zdim=None):
    #Here we are generating the atom coordinates of an ISF defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the FCC cell as we normally would do
    [atoms, per]=gen_cell_FCC(ao, xdim, ydim, zdim)
    ind=atoms[:,2]>=0 #Selecting the part of the crystal with z >=0
    atoms[ind,1]=atoms[ind,1]+(ao/np.sqrt(6)) #adding the ISF lattice shear to the selected half
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_cell_UTF1(ao=None, xdim=None, ydim=None, zdim=None):
    #here we are generating the atom coordinates of an UTF1 defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the ISF cell as we normally would do
    [atoms, per]=gen_cell_ISF(ao, xdim, ydim, zdim)
    ind=atoms[:,2]>=(ao/np.sqrt(3))-0.001 #Selecting the part of the crystal with z >= 1 lattice plane
    atoms[ind,1]=atoms[ind,1]+((ao/np.sqrt(6))/2) #adding the UTF1 lattice shear to the selected half
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_cell_ESF(ao=None, xdim=None, ydim=None, zdim=None):
    #here we are generating the atom coordinates of an ESF defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the ISF cell as we normally would do
    [atoms, per]=gen_cell_ISF(ao, xdim, ydim, zdim)
        
    ind=atoms[:,2]>=(ao/np.sqrt(3))-0.001 #Selecting the part of the crystal with z >= 1 lattice plane
    atoms[ind,1]=atoms[ind,1]+((ao/np.sqrt(6))) #adding the second part of the ESF lattice shear to the selected half
    
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_cell_UTF2(ao=None, xdim=None, ydim=None, zdim=None):
    #here we are generating the atom coordinates of an UTF2 defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the ESF cell as we normally would do
    [atoms, per]=gen_cell_ESF(ao, xdim, ydim, zdim)    
    
    ind=atoms[:,2]>=2*(ao/np.sqrt(3))-0.001 #Selecting the part of the crystal with z >= 2 lattice plane
    atoms[ind,1]=atoms[ind,1]+((ao/np.sqrt(6))/2) #adding the second part of the UTF2 lattice shear to the selected half
    
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_cell_TF(ao=None, xdim=None, ydim=None, zdim=None):
    #here we are generating the atom coordinates of an TF defect
    #We ensure that the atom at the origin is placed along the defect.
    #This makes our lives easier when calculating the RDF
    
    #Generating the ESF cell as we normally would do
    [atoms, per]=gen_cell_ESF(ao, xdim, ydim, zdim)    
    
    ind=atoms[:,2]>=2*(ao/np.sqrt(3))-0.001 #Selecting the part of the crystal with z >= 2 lattice plane
    atoms[ind,1]=atoms[ind,1]+((ao/np.sqrt(6))) #adding the second part of the TF lattice shear to the selected half
    
    ind2=atoms[:,1]>per[1,1] #Ensuring any atoms that pass through the periodic boundary are wrapped back into the cell
    atoms[ind2,1]=atoms[ind2,1]-(per[1,1]-per[0,1])
    return atoms, per

def gen_chem(atoms, types=2, comp=np.array([0.5,0.5])):   
    nums=np.arange(0, np.shape(atoms)[0])
    rng = np.random.default_rng()
    rng.shuffle(nums)
    start=0
    typearr=np.zeros((np.shape(atoms)[0],1))
    atoms=np.hstack((atoms,typearr))
    for i in np.arange(0, np.shape(comp)[0]):
        stop=start+comp[i]*np.shape(nums)[0]        
        stop=int(round(stop,0))        
        sub=nums[start:stop]
        sub=np.sort(sub)    
        atoms[sub,3]=i+1
        start=stop
    ind=np.where(atoms[:,3]==0)[0]    
    if np.shape(ind)[0]>0:
        rands=np.random.randint(1,types,np.shape(ind)[0])
        atoms[ind,3]=rands    
    return atoms
#%%

def gen_Neighbors(atoms, per,k):
  from periodic_kdtree import PeriodicCKDTree
  import numpy as np
  
  bounds = np.array([per[1,0]-per[0,0], per[1,1]-per[0,1], per[1,2]-per[0,2]]) 
  
  T = PeriodicCKDTree(bounds, atoms) # Input parameter for periodic function (PeriodicCKDTree)
  
  d, i = T.query(atoms, k ) #Outputs the distance between 12 nearest neighbors, and outputs     
                              #the indices of 12 nearest neighbors for each central atom. 
                              #Please note that these arrays are located under the name Neighbors 
                              #in the variable explorer.

  return d, i
#%%
def wc_parameter(atoms, Neighbors, comp): 
    #This function should now work for an arbitrary number of components with
    #arbitrary composition. It should work for beyond nearest neighbors if we
    #can provide the neighbor list. Seems to run pretty fast as well.
    #It returns alpha, a square matrix where the dimension is the number of components
    #The entries of alpha correspond to all i,j possible pairings between 
    #atom types
    

    atom_types=atoms[:,3]-1 #Storing the atom types but starting at 0 instead of 1.
    #The reason for this is the index in matrices start at 0. By setting the first
    #atom type to 0, we can assign it to the first column in our prob, chem, alpha, etc matrices

    neigh = Neighbors[1][:,1:] #Indices of neighbors not including the central atoms
            
    chem=np.zeros((np.shape(comp)[0],np.shape(comp)[0])) #This is the size of the matrix to hold the alpha terms
    
    comps=np.tile(comp,(np.shape(chem)[0],1)) #This is the c_j term for each entry in the matrix
    
    prob=chem #This is the probability matrix (starting as a matrix of 0s)
            
    
    #here I am cycling through all the atoms to find the type, and storing as 
    #cent
    for i in np.arange(0,np.shape(atoms)[0]):
        cent=int(atom_types[i])
        #Here i am cycling through the neighbors of each atom and storing its 
        #type in pair
        for j in np.arange(0,np.shape(neigh)[1]):          
            pair=int(atom_types[neigh[i,j]])
            #Here I am counting the number of i,j pairs and incrementing chem
            #by these counts. The atom type i,j determines the position in the
            #chem matrix. e,g. type 1,0 (or 2,1 originally before subtraction of 1)
            #will be stored at chem position 1,0
            chem[cent,pair]=chem[cent,pair]+1
             
    #here I am getting the probabilities. This is calculated by determining the
    #number of times that atom i is selected (comps*#of atoms), multiplied
    #by the coordination of the  neighbors (12 in this case)
    prob=chem/(comps.T*np.shape(atoms)[0]*np.shape(neigh)[1])
            
    #The kronecker delta matrix, sized to the system size
    delta=np.identity(np.shape(chem)[0])
    
    #Calculating alpha using the formula
    alpha=(prob-comps)/(delta-comps)

    return alpha

#%%
def gen_3dplotting(Neighbors, ao, atoms, per):
    import matplotlib.pyplot as plt
    dist=Neighbors[0][:,1:]
    dist=dist-ao/np.sqrt(2)
    dist=dist*10000
    dist=np.round(dist,0)
    dist=dist/10000

    if np.any(dist):
        print('Near neighbor not at ao/sqrt(2)')
    else:
        print('Passed')
        
    neigh=Neighbors[1][:,1:] 
        
    testpt=1 #Test point
        
    neighs_ind=neigh[testpt,:]
    
  
    #plotting
    # This line is commented
    plt.close('all') #Closing all open plots
    def_font= {'fontname':'Arial'} #Setting a font name
        
    plt.subplots(1,1, figsize=(6,6)) #Setting the size of our plot
    ax1=plt.subplot(1,1,1, projection='3d')
            
    ax1.scatter(atoms[testpt,0],atoms[testpt,1],atoms[testpt,2],'rs')
    ax1.scatter(atoms[neighs_ind,0],atoms[neighs_ind,1],atoms[neighs_ind,2],'ko')
    
    
    plt.ylabel('Y (Angstroms)', fontsize=22, **def_font, labelpad=5) #Setting the label for the y axis
    plt.yticks(**def_font, fontsize=12) #Setting the font for the x and y ticks
    plt.xticks(**def_font, fontsize=12)
    ax1.set_zlim([per[0,2],per[1,2]]) #Setting the plot limits on the z axis
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(13)) #Setting how many tick marks are on each axis
    ax1.yaxis.set_major_locator(plt.MaxNLocator(13))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(13))

    for t in ax1.zaxis.get_major_ticks(): t.label.set_fontsize(12), t.label.set_fontname('Arial') #Needed to align the font of the z axis ticks
    
    plt.xlabel('X (Angstroms)', fontsize=22, **def_font, labelpad=10) #Setting the font size and font for the x and y labels
    ax1.set_zlabel('Z (Angstroms)', fontsize=22, **def_font, labelpad=10) #Setting the font size and font for the z axis labels
    plt.ylim([per[0,1], per[1,1]]) #Setting the plot limits on the y axis
    plt.xlim([per[0,0],per[1,0]]) #Setting the plot limits on the x axis
    

    return dist
    
