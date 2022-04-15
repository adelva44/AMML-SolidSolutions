#Here, necessary Python packages will be imported and other written scripts will be called to perform the energy statistics calculations herein
import numpy as np                                                     #Python package used commonly for data manipulation involving arrays, matrices, etc
import Potential as pot                                               #Performs potential energy calculations
import rdf_coord as rc                                          #Generated coordination number (CN) and radial distribution function (RDF) relations based on lattice parameter and cutoff distance

ao = 3.5225                                                            #Inputted lattice parameter value in Angstroms
rcut = 6.5                                                             #Cutoff distance which will restrict coordination shells generated

comp = np.array([0.33, 0.33, 0.34])                                    #Refers to the composition of the system

[rdf, cn] = rc.rdf_coord_fcc(ao,rcut)                                  #Generates RDF and CN for perfect lattice based on ao and rc values, user can change fcc part to hcp or bcc based on material's lattice

#Declaring a .alloy file name to generate cohesive energy statistics as a string (between quotations '')
fname = 'FeNiCr.eam.alloy'

#Reading the EAM potential dataset declared above and extracting needed parameters from file
[rrange, rhorange, rho, Fr, Pp] = pot.potential_read(fname)

#Performs Cohesive Energy Calculations and Statistics based on chosen EAM file of chosen system
[form_E, test, covars] = pot.potential_stats2(rrange, rhorange, rho, Fr, Pp, comp, cn)
print(form_E)
