# the version of this file is for annotations branch
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:45:24 2021

@author: Matt
"""
# This script calculates the single atom Radial Distribution Function (RDF) and the Coordination Number (CN) for many lattice types in both Nonfaulted and Faulted States.
# Here, a0 refers to the lattice parameter and rc refers to the cutoff radius.


# In this function, Nonfaulted state CNs and RDFs are calculated for FCC, BCC, and HCP lattice types.
# These calculations are based on known atomic distance ratios and CNs that are specific to each lattice type. Using these, the RDF and CN values are determined up to the cutoff distance.
import numpy as np
def rdf_coord_fcc(a0,rc):
    atomic_dist_ratios = [0.7071070615,1,1.224743736,1.414214123,1.581138952,1.732050114,1.870828588,2,2.121321185,2.236067768,2.345207859,2.449490319,2.549510251,2.738613326,2.828428246,2.915489749,3,3.082203872,3.16227221,3.240375854,3.316628702,3.391173121,3.464094533,3.535535308,3.605552392,3.674231207,3.741657175,3.807887244,3.937015945,4,4.062015945,4.123092255,4.183314351,4.242653759,4.301167426,4.358912301,4.415888383,4.472124146,4.527705011,4.582574032,4.636816629,4.690404328,4.743422551,4.847693622,4.898974943,4.949743736,5,5.049743736,5.099031891,5.147807517,5.196156036,5.244048975,5.338553531,5.385165148,5.431378132,5.477220957,5.522665148,5.6125,5.656862187]
    coord_numbers = [12,6,24,12,24,8,48,6,36,24,24,24,72,48,12,48,30,72,24,48,24,48,8,84,24,96,48,24,96,6,96,48,48,36,120,24,48,24,48,48,120,24,120,96,24,108,30,48,72,72,32,144,96,72,72,48,120,144,12]
    atomic_dist = []
    cn = []
    for ratio in atomic_dist_ratios:                                            #For distance values less than the inputted cutoff distance, add atomic distance values that meet that criteria and corresponding CN values into a generated empty list to create a nested list of atomic distance values and their respective coordination numbers
        if float(ratio)*float(a0) <= rc:
            atomic_dist.append(round(float(ratio)*float(a0),5))
    for i in range(len(atomic_dist)):
        cn.append([atomic_dist[i],coord_numbers[i]])
    cn_array = np.asarray(cn)                                                   #Convert list data type into an array data type
    rdf = np.zeros((np.shape(cn_array)))                                        #Creates an array filled with zeros that is the same size of the cn_array that will be filled with calculated rdf values
    rdf[:, 0] = cn_array[:,0] / cn_array[0,0]                                   #First column of the rdf array is filled with values determined as the quotient between the all atomic distances and the first atomic distance value
    rdf[:, 1] = (cn_array[0,1] / cn_array[:, 1]) * (rdf[:, 0] * rdf[:, 0])      #Second column of the rdf array is filled with values determined as the product first coordination number divided by all the coordination numbers and the square of the atomic distance values determined above
    rdf[:, 1] = np.round(rdf[:, 1], 4)                                          #Rounds all the values in the second column of rdf up to 4 digits after the decimal place
    return rdf,cn_array

def rdf_coord_hcp(a0,rc):
    atomic_dist_ratios = [0.7071057487579844,1.0,1.1546997870830376,1.2247437899219304,1.3540070972320795,1.4142143364088005,1.58113839602555,1.6832505322924058,1.7320511000709722,1.779511710432931,1.8257430801987227,1.8708275372604686,1.9148530872959546,2.0412405961674946,2.121320085166785,2.1984840312278213,2.2360681334279633,2.2730305180979418,2.309402413058907,2.345206529453513,2.380476933995742,2.4152306600425835,2.449490418736693,2.483276082327892,2.5495102909865155,2.6140638750887155,2.6770617459190915,2.708014194464159,2.738611781405252,2.7688743789921935,2.7988105039034776,2.8284258339247694,2.857737402413059,2.915474804826118,2.9720936834634495,3.0000000000000004,3.027650816181689,3.05505180979418,3.0822058197303055,3.135815471965933,3.1885223562810503,3.240369056068133,3.291403832505323,3.3166245564229953,3.341655074520937,3.3665010645848117,3.3911653655074523,3.415650816181689,3.439960255500355,3.4641022001419444,3.5355344215755857,3.6055500354861603,3.6285904897090138,3.651483321504613,3.674234208658623,3.696845990063875,3.719318665720369,3.7416579134137686,3.763863733144074,3.807886444286728,3.8514066713981547,3.8944414478353444,3.915781405251952,3.937004968062456,3.958114975159688,3.979111426543648,4.020780695528744,4.0620184528034065,4.102844570617459,4.123105748757984,4.183298793470547,4.222952448545067,4.24264017033357]
    coord_numbers = [12.0,6.0,2.0,18.0,12.0,6.0,12.0,12.0,6.0,6.0,12.0,24.0,6.0,12.0,12.0,24.0,12.0,12.0,2.0,12.0,6.0,24.0,6.0,12.0,24.0,12.0,6.0,24.0,12.0,12.0,24.0,6.0,12.0,24.0,24.0,18.0,12.0,12.0,24.0,12.0,12.0,36.0,24.0,12.0,18.0,12.0,24.0,12.0,48.0,2.0,36.0,24.0,12.0,12.0,42.0,6.0,12.0,24.0,12.0,12.0,36.0,12.0,24.0,72.0,12.0,24.0,12.0,48.0,24.0,24.0,24.0,12.0,18.0]
    atomic_dist = []
    cn = []
    for ratio in atomic_dist_ratios:
        if float(ratio) * float(a0) <= rc:
            atomic_dist.append(round(float(ratio) * float(a0), 5))
    for i in range(len(atomic_dist)):
        cn.append([atomic_dist[i], coord_numbers[i]])
    cn_array = np.asarray(cn)
    rdf = np.zeros((np.shape(cn_array)))
    rdf[:, 0] = cn_array[:, 0] / cn_array[0, 0]
    rdf[:, 1] = (cn_array[0, 1] / cn_array[:, 1]) * (rdf[:, 0] * rdf[:, 0])
    rdf[:, 1] = np.round(rdf[:, 1], 4)
    return rdf,cn_array

def rdf_coord_bcc(a0,rc):
    atomic_dist_ratios = [0.8660241305890702,1.0,1.4142143364088005,1.6583136976579134,1.7320511000709722,2.0,2.1794492547906317,2.2360681334279633,2.449490418736693,2.5980752306600423,2.8284258339247694,2.9580411639460613,3.0000000000000004,3.1622767920511,3.27871965933286,3.3166245564229953,3.4641022001419444,3.570713981547197,3.6055500354861603,3.7416579134137686,3.840573456352023,4.0,4.0926756564939675,4.123105748757984,4.24264017033357,4.330126330731015,4.358898509581263,4.472136266855927,4.555216465578425,4.582574875798438,4.690415897799857,4.769694819020582,4.898980837473386,4.974938254080908,5.0,5.099020581973031,5.172039744499645,5.196153300212917,5.361902058197303,5.385163946061036,5.477226401703336,5.545268985095813,5.65685450674237]
    coord_numbers = [8.0,6.0,12.0,24.0,8.0,6.0,24.0,24.0,24.0,32.0,12.0,48.0,30.0,24.0,24.0,24.0,8.0,48.0,24.0,48.0,72.0,6.0,24.0,48.0,36.0,56.0,24.0,24.0,72.0,48.0,24.0,48.0,24.0,72.0,30.0,72.0,72.0,32.0,48.0,72.0,48.0,48.0,12.0]
    atomic_dist = []
    cn = []
    for ratio in atomic_dist_ratios:
        if float(ratio)*float(a0) <= rc:
            atomic_dist.append(round(float(ratio)*float(a0),5))
    for i in range(len(atomic_dist)):
        cn.append([atomic_dist[i],coord_numbers[i]])
    cn_array = np.asarray(cn)
    rdf = np.zeros((np.shape(cn_array)))
    rdf[:, 0] = cn_array[:, 0] / cn_array[0, 0]
    rdf[:, 1] = (cn_array[0, 1] / cn_array[:, 1]) * (rdf[:, 0] * rdf[:, 0])
    rdf[:, 1] = np.round(rdf[:, 1], 4)
    return rdf,cn_array

#CN and RDF Calculations for Faulted States, based on nearest neighbors, generated lattice type and atom positions:
def rdf_coord(atoms, rc, pos):
    import numpy as np
# Based on a lattice filled with atoms from the gen_cell_MD.py script,
    pos = np.ones((np.shape(atoms)[0], 3)) * pos                #Generate a position array filled with 1s that is the same size as the lattice
    atoms = atoms - pos                                         #Finds distance between starting position and atom you are parsing through
    dists = np.zeros((np.shape(atoms)[0]))

    for i in np.arange(0, np.shape(atoms)[0]):
        dists[i] = np.sqrt(sum(atoms[i, :] * atoms[i, :]))      #Calculates distances between the atoms for each atom in the lattice

    dists = np.sort(dists)
    ind = dists <= rc                                           #Selects atoms within the cut off distance and appends the atoms as a nearest neighbor to record the distance between each neighbor
    neigh = dists[ind]
    neigh = neigh[1:]
    neigh = np.round(neigh, 5)                                  #Rounds the neighbors value upto 5 decimal places
    uni = np.unique(neigh)                                      #Records only unique values

    cn = np.zeros((np.shape(uni)[0], 2))                        #Generates coordination numbers array based on how many unique neighbors there are and fills the entries with zeros
    rdf = np.zeros((np.shape(cn)[0], 2))                        #Generates rdf array based on the size of the CN array and fills the entries with zeros

    for i in np.arange(0, np.shape(uni)[0]):                    #For all unique neighbors, append the distances in the first column
        ind = neigh == uni[i]                                   #if the neighbor value and ind value match, then append that value to the uni array
        cn[i, 0] = uni[i]                                       #set the first column of cn entries as the distance values
        cn[i, 1] = sum(ind)                                     #set the second column of cn entries to how many neighbors there are


    rdf[:, 0] = cn[:, 0] / cn[0, 0]
    rdf[:, 1] = (cn[0, 1] / cn[:, 1]) * (rdf[:, 0] * rdf[:, 0])
    rdf[:, 1] = np.round(rdf[:, 1], 4)

    return rdf, cn
