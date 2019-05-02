
# coding: utf-8

# # Memristor Weight Mapping Code
# ### By Michael Elimu - LY2016030

# Import of the necessary Python files required

# In[ ]:


import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import h5py
from time import perf_counter as my_timer


# ## Get Weight Values and Features
# Obtain the weights needed for the memristor model input and some of it's features such as the rows and columns

# ## Conductance
# We need to find the conductance values of the Ron and Roff Resistance values of the memristor
# These are necessary, since our weight mappings are entirely based on the usuable conductance range of the memristor element that we shall be using

# In[ ]:


def conductanceValues(Ron, Roff):
    '''
    Calculates the conductance values from the resistance vaues provided
    
    Parameters:
    -------------
    
    Ron: Integer value
        
        Usually 2000 for TiO2 memristors
        The minimum resistance value of the memristor that when reached indicates that 
        the memristor is in a on state (1)
        
    Roff: Integer value 
        
        Usually 2000000 for TiO2 memristors
        The maximum resistance value of the memristor considered to be the value when reached, 
        the memristor is in off state (0)
        
    Returns:
    ----------
    conductance_values: a dictionary
        
        This containes the values of conductances calculated as a reciprocal of the resistances
        
        conductance_values["c_min"]  ------------------ Minimum conductance(Reciprocal of maximum resistance)
        conductance_values["c_max"]  ------------------ Maximum conductance(Reciprocal of minimum resistance)
        conductance_values["c_range"]------------------ The difference between Minimum and Maximum conductance
    
    '''
    
    Cmax, Cmin = (1/Ron, 1/Roff)
    cond_range = Cmax - Cmin
    
    conductance_values = {
        "c_min": Cmin,
        "c_max": Cmax,
        "c_range": cond_range
    }
    
    return conductance_values


# ## Weight and Bias Splits
# Seperate the weights to help handle the positive and negative weights and biases differently
# This is necessary because, we can not litterally write negative weights to the memristors and we don't want to ignore them as either. 
# It is therefore ideal to split them up but however get the absolute values of the negative weights and treat them as positive weights while writing to the memristor. However, during manipulation, the kernel output from the negative weights and bias section can be passed through a substractor together with the corresponding output kernel of the positive weights. This therefore gives us the differnce value that can be fed to the next layers in the neural network
# We also desire to obtain a maximum absolute value of either the weights or biases that can be used as the maximum value to be fit in the conductance raange of the memristor. This will help accommodate all the existing values for both the weights and the biases
# 

# In[ ]:


def weight_split(w_rows, w_cols, weight, mode='normal'):
    
    '''
    Makes a weight split into positives and negatives grouped differently allow easy handling
    Corresponding positive and negative weights are passed to the substractor to obtain the 
    total weight values onn the final memristor.
    
    Parameters:
    -------------
    w_rows: Integer value
        
        This is the number of row in the original weight
        
    W_cols: integer value
        
        This is the number of columns in the original weight. This is used to create two columns
        to represent positive and negavite values of the memristor weights
        
    weight: numpy array or matrix
        
        This is a matrix of the weights from the weight file(pretrained weights)
        
    mode: can be normal or proposed
        
        normal: The weights be split as all postive values to the left and negative values
                to the right
        proposed: The weights be split in a 4X5 manner, each with two positive columns, followed by
                two corresponding negative column weights and a redundant cell for overcoming the 
                defects
                
    Returns:
    ----------
    W: n-d array or matrix
        
        The splited weight values
    
    max_val: integer value
        
        The maximum absolute value in the weight matrix to be used in determining the conductance
        split
    
    '''

    # Obtain the absolute value of the stacked array
    w_b = np.abs(weight)
    
    max_val = max(w_b.flatten())

#     # Get the index of the maximum value
#     max_index = w_b.argmax()

#     # Retrieve the maximum value
#     max_val = w_b.ravel()[max_index]

    w_r, w_c = weight.shape
    # print(w_cols)

    w_pos =np.zeros([w_r,w_cols]).reshape(w_r, w_cols)
    w_neg = np.zeros([w_r,w_cols]).reshape(w_r, w_cols)

    w_pos[weight > 0] = weight[weight > 0]
    w_neg[weight <= 0] = weight[weight <= 0]*-1

    W = np.hstack((w_pos, w_neg))
    
    if mode == 'proposed':
        P = []
        for i in range(0, w_c, 2):
            if(i+1<w_c):
                P.append(w_pos[:,i])
                P.append(w_neg[:,i])
                P.append(w_pos[:,i+1])
                P.append(w_neg[:,i+1])
           
        W = np.array(P)
        W = W.T
        
    # print(W)

    # print(f"Positive Weights\n{W[:w_rows,:w_cols]}\n")
    # print(f"Negative Weights\n{W[:w_rows,w_cols:]}")
    # print(f"Positive Bias\n{W[w_rows:,:w_cols]}\n")
    # print(f"Negative Bias\n{W[w_rows:,w_cols:]}")
    # assert w_pos.shape == w_b_T.shape
    
    return W, max_val


# ## Weights and Bias Conductance
# We use the previously calculated minimum, maximum conductances and conductance range in the calculation of the weight and bias conductance equivalence in relation to the memristor conductance
# ### Note:
# The weights and biases have been put up together to help ease the manipulation of the whole learned parameters as a crossbar - a reflection of the operation used in memristor crossbar array

# In[ ]:


def weight_bias_cond(w, conductance_values, w_b_max):
    
    '''
    Convert the weights so as to be represented as conductance values
    The output weights will be conductance values as a representative of  the existing weights
    
    Parameters:
    -----------
    w: an array or matrix of weight values
    
        The weights that we need to convert to conductance values
        
    conductance_values: dictionary of memristor resistance and conductance range
    
        Contains the calculated conductance values and range
        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
        
    max_val: float or double
        The maximum absolute value in the weight matrix to be used in determining the conductance
        split
        
    Returns:
    ----------
    w_cond: n-d array or matrix
        
        Weight values converted as conductances of the memristor
        These weight values have to be in within the range of the conductance values
    
    '''
    
    # Extract Conductance Values
    cond_range = conductance_values["c_range"]
    Cmin = conductance_values["c_min"]
    Cmax = conductance_values["c_max"]
    
    w_cond = (w/w_b_max)*cond_range + Cmin
    
    return w_cond


# In[ ]:


def adjust_con_weights(weighted_bits, cond_split):
    
    # Move the weight values to the conductance levels
    r, c = weighted_bits.shape
    for i in range(r):
        for j in range(c):
            for k in range(len(cond_split)):
                weighted_bits[i, j] = cond_split[np.argmax(cond_split>=weighted_bits[i, j] )]
                
#     for con_val in cond_split:
#         if weighted_bits < con_val:
#             weighted_bits = con_val
#             break    
    
    return weighted_bits


# ## Redundunt Rows and Columns
# The main objective of the research is to identify a fault toleranct architecture to the reknown unrecoverable faults called "Stack-at" faults mostly caused by open and closed defects in the memristor crossbar due to manufacture imaturity that exixst for this new technology
# We therefore need to simulate some of these faults i.e. stack-at-0 and stack-at-1 faults abbrevaited as SA0 and SA1 respectively.
# To help overconme this problem, there are two scenarios that will be considered concurrently
# 1. If the positive side of the crossbar is faulty but the correspondiing negative cell is not faulty, we can use this to overcome the positive cell fault
# 2. If the negative side of the crossbar is faulty but the correspondiing positive cell is not faulty, we can use this to overcome the negative cell fault
# 3. If both the positive and the corresponding negative cells are faulty, we would need to use the redundant rows and columns that are provided in the crossbar array.
# Note: The random faults will be applied inclusive of the exixting redandunt cells introduced in the crossbar array, since they may be defected too

# In[ ]:


def add_red_array(w, per_def, conductance_values, mode, filter_window):
    
    '''
    A function to add redundat rows and columns to the existing crossbar array
    These columns are used for overcoming the faults that may exist on the memristor.
    The rows and columns are dynamic and entirely based on the percentage defect to minimize memory wastage
    
    Parameters:
    -----------
    w: an n-d array of conductance weight values
    
        The crossbar array with weighted conductance values
        We add redundant rows and columns to helf overcome the existing faults
    
    perc_def: an integer
    
        This number indicates the percentage of the faults on the memristor device.
        It is used in dynamically calculating the required redundant rows and columns for the defected memristor
        
    conductance_values: dictionary of memristor resistance and conductance range
    
        Contains the calculated conductance values and range
        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
        
    mode: can be normal or proposed
        
        normal: The weights be split as all postive values to the left and negative values
                to the right
        proposed: The weights be split in a 4X5 manner, each with two positive columns, followed by
                two corresponding negative column weights and a redundant cell for overcoming the 
                defects
        
    Returns:
    ----------
    
    crossbar: n-d array or matrix
        
        A matrix of weights with redundant rows and columns depending on the fault percentage
    
    '''
    
    # Get the minimum conductance from the conductance values
    cond_min = conductance_values['c_min']
    
    # Expand the array to include fault tolerant cells
    w_row, w_col = w.shape
    cols = 0
    endval = 0
    iter_start = 0
    remainder = 0
    # Percentage of rows and columns to add
    #rows = int(np.ceil((per_def/100)*w_row))
    
    if mode =='normal':
        cols = int(np.ceil((per_def/100)*w_col))

        if cols%2 != 0:
            cols = cols+1
    
        red_cols = np.zeros((w_row, cols))+cond_min

        w2 = np.concatenate((w, red_cols), axis=1)

        # red_rows = np.zeros((rows, w2.shape[1]))+cond_min

        crossbar = w2
    
    if mode == 'proposed':
        P = []
        red_cols = np.zeros((w_row, 1))+cond_min
        
        num_iters = w_col // filter_window
        remainder = w_col % filter_window
        
        for i in range(num_iters):
            for j in range(iter_start, filter_window * (i+1)):
                if j%filter_window == 0 and j !=0:
                    # Append two redundant columns after the last set of positive and negative cells after the selection
                    P.append(red_cols[:, 0])
                    P.append(red_cols[:, 0])
                    # print('added first')
                    # print('j up: ', j)
                    # Append the next set of weights after the redundant columns
                    P.append(w[:, j])
                else:
                    # Apply the normal set of weights to the crossbar
                    P.append(w[:, j])
                    
                    # print('j down: ', j)
            iter_start = filter_window * (i+1)
        # Append a redundant column to the last set of column cells Lat cells in the crossbar
        endval = iter_start+remainder
        if remainder != 0:
            for i in range(iter_start, endval):
                if i % filter_window == 0 and i != 0:
                    P.append(red_cols[:, 0])
                    P.append(red_cols[:, 0])
                    # print('added remainder')
                    P.append(w[:, i])
                else:
                    P.append(w[:, i])
#                     print('i remainder up: ', i)
                    
                    # print('i remainder down: ', i)
        
        P.append(red_cols[:, 0])
        P.append(red_cols[:, 0])
        # print('added last')
                    
        
        
#         for j in np.arange(0, w_col, 1):
#             if j%filter_window == 0 and j !=0:
#                 # Append two redundant columns after the last set of positive and negative cells after the selection
#                 P.append(red_cols[:, 0])
#                 P.append(red_cols[:, 0])
#                 # Append the next set of weights after the redundant columns
#                 P.append(w[:, j])
#             else:
#                 # Apply the normal set of weights to the crossbar
#                 P.append(w[:, j])
#         # Append a redundant column to the last set of column cells Lat cells in the crossbar
#         P.append(red_cols[:, 0])
#         P.append(red_cols[:, 0])
        
#         for j in np.arange(0, w_col, 1):
#             if j%filter_window == 0 and j !=0:
#                 # Append two redundant columns after the last set of positive and negative cells after the selection
#                 P.append(red_cols[:, 0])
#                 P.append(red_cols[:, 0])
#                 # Append the next set of weights after the redundant columns
#                 P.append(w[:, j])
#             else:
#                 # Apply the normal set of weights to the crossbar
#                 P.append(w[:, j])
#         # Append a redundant column to the last set of column cells Lat cells in the crossbar
#         P.append(red_cols[:, 0])
#         P.append(red_cols[:, 0])
        
        P = np.array(P)
        crossbar = P.T
        
        # print("crossbar: ",crossbar.shape)

    return crossbar,iter_start, remainder


# ### Modified Weight Split
# This is the proposed Weight split. We split our weights to form a *4X6* matrix were we have two columns for *Positive weights* followed by two columns of *Negative Weights* and one redundant column for each setup of the weights

# In[ ]:


def groupings(cells, endval, remainder, filter_size=4):
    
    '''
    Takes in the existing cells and modifies their arrangments
    
    Parameters:
    -----------
    cells: list of cells
    
        list containing all the crossbar cells arranged in order
        
    Returns:
    ---------
    crossbar_split: dictionary
    
        Outputs a dictionary containing the crossbar split into the positive, negative,
        redundant and grouping of the cells that need to be handled together
        
        crossbar_split['positive']  ................. Positive Crossbar Cells 
        crossbar_split['negative']  ................. Negative Crossbar Cells
        crossbar_split['redundant'] ................. Redundant Columns
        crossbar_split['groups']    ................. Crossbar Cell groups
    
    '''
#     print('endval: ', endval)
#     print('remainer: ', remainder)
    step = 0
    xbr_pos = []
    xbr_neg = []
    xbr_red_pos = []
    xbr_red_neg = []
    groups = []
    for i in cells:
        if filter_size == 4:
            if i[1]%6 == 0:
                # groups.append([i, (i[0], i[1]+2), (i[0], i[1]+1), (i[0], i[1]+3), (i[0], i[1]+split_factor)])
                groups.append([i, (i[0], i[1]+1), (i[0], i[1]+filter_size), (i[0], i[1]+filter_size + 1)])
                groups.append([(i[0], i[1]+2), (i[0], i[1]+3), (i[0], i[1]+filter_size),(i[0], i[1]+filter_size + 1)])
                for filter_cell in range(filter_size):
                    if filter_cell%2 == 0:
                        xbr_pos.append((i[0], i[1]+filter_cell))
                    else:
                        xbr_neg.append((i[0], i[1]+filter_cell))
                xbr_red_pos.append((i[0], i[1]+filter_size))
                xbr_red_neg.append((i[0], i[1]+filter_size + 1))
        
        elif filter_size == 8:
            if i[1]%10 == 0:
                # groups.append([i, (i[0], i[1]+2), (i[0], i[1]+1), (i[0], i[1]+3), (i[0], i[1]+split_factor)])
                if i[1] > endval:
                    filter_size = remainder
                else:
                    filter_size = filter_size
                    
                groups.append([i, (i[0], i[1]+1), (i[0], i[1]+filter_size), (i[0], i[1]+filter_size + 1)])
                groups.append([(i[0], i[1]+2), (i[0], i[1]+3), (i[0], i[1]+filter_size),(i[0], i[1]+filter_size + 1)])
                groups.append([(i[0], i[1]+4), (i[0], i[1]+5), (i[0], i[1]+filter_size),(i[0], i[1]+filter_size + 1)])
                groups.append([(i[0], i[1]+6), (i[0], i[1]+7), (i[0], i[1]+filter_size),(i[0], i[1]+filter_size + 1)])
                for filter_cell in range(filter_size):
                    if filter_cell%2 == 0:
                        xbr_pos.append((i[0], i[1]+filter_cell))
                    else:
                        xbr_neg.append((i[0], i[1]+filter_cell))
                xbr_red_pos.append((i[0], i[1]+filter_size))
                xbr_red_neg.append((i[0], i[1]+filter_size + 1))
                
#     print('groups: ', groups)
            
    crossbar_split = {
        "positive":xbr_pos,
        "negative":xbr_neg,
        "redundant_pos":xbr_red_pos,
        "redundant_neg":xbr_red_neg,
        "groups":groups
    }
            
    return crossbar_split


# ## Adding The SAFs
# This section adds the defects randomly to the crossbar. We need to add these defects inclusive of the redundant rows and columns that we added for overcoming these faults too.
# This is because, these cells too are likely to suffer from these defects and we do not have to treat them as a special case. Introducing faults to them renders some cells also not usuable, which is an ideal case

# In[ ]:


def add_defects(norm_weight, cxbar, perc_def, conductance_values, mode="normal", mask=4, case='crossbar_with_split', seed_val=0):
    
    '''
    Adds defects to an existing memristor crossbar array. The distribution of the defects varies.
    A random pick of the required number od cells is picked based on the percentage defect
    
    Parameters:
    -----------
    cxbar: an array of conductance weight values
    
        The original crossbar array with weighted conductance values
    
    perc_def: an integer
    
        This number indicates the percentage of the faults on the memristor device.
        It is used in dynamically calculating the required redundant rows and columns for the defected memristor
        
    conductance_values: dictionary of memristor resistance and conductance range
    
        Contains the calculated conductance values and range
        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
        
    Returns:
    ----------
    
    defects: dictionary
        A dictionary containing the generated faulty crossbar, the different faulty and none faulty cells
        
        defects['xbar']       .............. Generated Faulty crossbar
        defects['all_cells']  .............. All Cells
        defects['f_cells']    .............. Faulty Cells
        defects['SA_0']       .............. Stuck at 0 Cells
        defects['SA_1']       .............. Stuck at 1 Cells
    
    '''
    defected = {}
    
    f_xbar = np.array(cxbar)
    xbar = []
    r, c = cxbar.shape
    
    # print("cxbar.shape: ",cxbar.shape)
    
    r_norm, c_norm = norm_weight.shape
    all_cells = []
    all_cells_norm = []
    global cells_to_pick
    
    for i in range(r):
        for j in range(c):
            all_cells.append((i, j))
            
    for i in range(r_norm):
        for j in range(c_norm):
            all_cells_norm.append((i, j))
            
    # TWO CASES CONSIDERED IN THIS IMPLEMENTATION
    # Considering that each split can be faulty, and picking up the likely faulty cells
    # Use the fault on the general crossbar and get the number of likely affected cells
    # Choose the number of cells from the likely faulty cells

    SA_0 = []
    SA_1 = []
    defected = {}
    pad = 0;
    h_stride = mask
    w_stride = mask + 2

    if perc_def > 0:

        np.random.seed(seed_val)
        faulty_sample =[]
        faulty_cells = []

        n_H = int(np.ceil((r - h_stride + 2*pad)/h_stride) + 1)
        n_W = int(np.ceil((c - w_stride + 2*pad)/w_stride) + 1)

        for h in range(n_H):            # Loop over the vertical axis
            for w in range(n_W):        # Loop over the horizontal axis
                vert_start = h*h_stride
                v_end = vert_start + h_stride
                if v_end >= r:
                    vert_end = r
                else:
                    vert_end = v_end

                horiz_start = w*w_stride
                h_end = horiz_start + w_stride

                if h_end >= c:
                    horiz_end = c
                else:
                    horiz_end = h_end

                # Use the corners to define the 3D slice of Weight Matrix
                a_slice_prev = f_xbar[vert_start:vert_end, horiz_start:horiz_end]

                # print("Slice h{} w{}\n {}\n".format(h, w, a_slice_prev))

                cell = []
                for y in range(horiz_start, horiz_end):

                    for x in range(vert_start, vert_end):
                        cell.append((x, y))
                    # Apply the faults using the filter - Convolution Manner
                    # print("h_stride*w_stride: ", h_stride*w_stride)
                    # print("cell:\n",cell)
                    # Randomly Pick out cells from a random selection pattern
                    cell_choosing = np.random.permutation(cell)  
                    # my_cells = set(cells)
                    # Find out the number of cells to change according to the defect percentage
                    cells_to_change = int(np.ceil((perc_def/100)*(h_stride*w_stride)))
                    # print("cell_choosing:\n",cells_to_change)
                    # Choose from the random picks
                    # faulty_cells = faulty_chosen(cell_choosing, cells_to_change)
                    faulty_cells = faulty_chosen(norm_weight, cell_choosing, cells_to_change)

                faulty_sample.append(faulty_cells)

        likely_faulty = cell_list(faulty_sample)
        # print("likely_faulty:\n",likely_faulty)

        # Permute all the cells that have been chosen
        permuted = np.random.permutation(likely_faulty)

        # Three case Scenario:
        # Considering the fault to be distributed in the entire crossbar with split
        
        # Defined by the number of cells in the crossbar excluding the proposed crossbar arrangement
        cells_to_pick = int(np.ceil((perc_def/100)*(r_norm*c_norm)))
        cells_considered = [] 
        
        if case == 'crossbar_no_split':               
            permuted = np.random.permutation(all_cells)

        # Pick the cells
        cells_considered = faulty_chosen(norm_weight, permuted, cells_to_pick)

        # print("permuted:\n",permuted)
        
        if mode=='normal':
            f_xbar = np.array(norm_weight)
            defected = distribute_fault(cells_considered, conductance_values, f_xbar, norm_weight, all_cells_norm, mode)
    
        elif mode=='proposed':
            defected = distribute_fault(cells_considered, conductance_values, f_xbar, norm_weight, all_cells, mode)
        
        
    # No Fault
    else:
        if mode=='normal':
            f_xbar = np.array(norm_weight)
            all_cells = all_cells_norm
        
        defected = {
        "xbar":f_xbar,
        "all_cells":all_cells,
        "f_cells":[],
        "SA_0": [],
        "SA_1":[] 
        }
        
        # if perc_def == 0:
            # print(f'Fault Distribution\n{"#"*40}\nSA1 - Fault: {perc_def}%\nSA0 - Fault: {perc_def}%\n')
        # else:
            # print(f'Fault Distribution\n{"#"*40}\nSA1 - Fault: {(len(SA1)/ len(faulty_cells))*100}%\nSA0 - Fault: {(len(SA0)/ len(faulty_cells))*100}%\n')
#     print("defected: ",defected)
    return defected

def faulty_chosen(or_xbar, choosen_cells, cells_number):
    
    # print("choosen_cells: ",choosen_cells)
    # print("cells_number: ",cells_number)
    r, c = or_xbar.shape
    cells = []
    
    for cell in choosen_cells:
        cells.append(tuple(cell))
    
    f_cells = []    
    
    # Check if there are any faulty Cells
    if len(cells) > 1:
        for i in range(len(cells)):
            if len(f_cells) < cells_number:
                if cells[i][0] < r and cells[i][1]<c:
                    f_cells.append((cells[i][0],cells[i][1]))
            else:
                break
    else:
        # Convert none list cells to form a tupple
        for cell in cells:
            if type(cell) == list:
                if cell[0] < r and cell[1]<c:
                    f_cells.append((cell[0],cell[1]))
            else:
                if cell[0] < r and cell[1]<c:
                    f_cells.append(cell)
    
    return f_cells

def cell_list(cells):
    
    perm_cells = []
                    
    for i in cells:
        for j in i:
            perm_cells.append(j)
            
    return perm_cells

def distribute_fault(faulty_cells, conductance_values, f_xbar, cxbar, all_cells, mode):
    distribution = {}
    r, c = cxbar.shape
    # Extract Conductance Values
    c_min = conductance_values["c_min"]
    c_max = conductance_values["c_max"]
    # Fault Distribution
    SA0 = []
    SA1 = []
    
    N = len(faulty_cells)
    if N > 0:
        for i in range(N):
            if i < N//2:
                f_xbar[faulty_cells[i][0],faulty_cells[i][1]] = c_min
                SA0.append((faulty_cells[i][0],faulty_cells[i][1]))
                
            else:
                f_xbar[faulty_cells[i][0],faulty_cells[i][1]] = c_max
                SA1.append((faulty_cells[i][0],faulty_cells[i][1]))
    else:
        f_xbar = cxbar
    
    distribution = {
        "xbar":f_xbar,
        "all_cells":all_cells,
        "f_cells":faulty_cells,
        "SA_0": SA0,
        "SA_1":SA1 
    }
    return distribution


# In[ ]:


# def add_defects(cxbar, perc_def, conductance_values, mask_size=4, mode="normal"):
    
#     '''
#     Adds defects to an existing memristor crossbar array. The distribution of the defects varies.
#     A random pick of the required number od cells is picked based on the percentage defect
    
#     Parameters:
#     -----------
#     cxbar: an array of conductance weight values
    
#         The original crossbar array with weighted conductance values
    
#     perc_def: an integer
    
#         This number indicates the percentage of the faults on the memristor device.
#         It is used in dynamically calculating the required redundant rows and columns for the defected memristor
        
#     conductance_values: dictionary of memristor resistance and conductance range
    
#         Contains the calculated conductance values and range
#         c_min      = conductance_values['c_min']  .............. Minimum Conductance
#         c_max      = conductance_values['c_max']  .............. Maximum Conductance
#         cond_range = conductance_values['c_range'].............. Conductance Range
        
#     Returns:
#     ----------
    
#     defects: dictionary
#         A dictionary containing the generated faulty crossbar, the different faulty and none faulty cells
        
#         defects['xbar']       .............. Generated Faulty crossbar
#         defects['all_cells']  .............. All Cells
#         defects['f_cells']    .............. Faulty Cells
#         defects['SA_0']       .............. Stuck at 0 Cells
#         defects['SA_1']       .............. Stuck at 1 Cells
    
#     '''
#     defected = dict()
    
#     f_xbar = np.array(cxbar)
#     r, c = cxbar.shape
#     all_cells = []
    
#     if mode == "normal":
#         for i in range(r):
#             for j in range(c):
#                 all_cells.append((i, j))

#         defected = choose_cells(f_xbar, r, c, all_cells, perc_def, conductance_values)
        
#     elif mode == "proposed":
#         pad = 0;
#         h_stride = mask_size
#         w_stride = mask_size + 1 

#         n_H = int((r - h_stride + 2*pad)/h_stride) + 1
#         n_W = int((c - w_stride + 2*pad)/w_stride) + 1

#         for h in range(n_H):            # Loop over the vertical axis
#             for w in range(n_W):        # Loop over the horizontal axis
#                 vert_start = h*h_stride
#                 vert_end = vert_start + h_stride
#                 horiz_start = w*w_stride
#                 horiz_end = horiz_start + w_stride

#                 # Use the corners to define the 3D slice of Weight Matrix
#                 a_slice_prev = f_xbar[vert_start:vert_end, horiz_start:horiz_end]

#                 # print("Slice h{} w{}\n {}\n".format(h, w, a_slice_prev))

#                 cell = []
#                 for y in range(horiz_start, horiz_end):

#                     for x in range(vert_start, vert_end):

#                         cell.append((x, y))

#                 # Apply the faults using the filter - Convolution Manner
                
#                 defected = choose_cells(f_xbar, h_stride, w_stride, cell, perc_def, conductance_values)
    
#         # Fault Distibution

#         # if perc_def == 0:
#             # print(f'Fault Distribution\n{"#"*40}\nSA1 - Fault: {perc_def}%\nSA0 - Fault: {perc_def}%\n')
#         # else:
#             # print(f'Fault Distribution\n{"#"*40}\nSA1 - Fault: {(len(SA1)/ len(faulty_cells))*100}%\nSA0 - Fault: {(len(SA0)/ len(faulty_cells))*100}%\n')

#     return defected

# def choose_cells(f_xbar, r, c, all_cells, perc_def, conductance_values):
#     defects = {}
    
#     # Extract Conductance Values
#     c_min = conductance_values["c_min"]
#     c_max = conductance_values["c_max"]
    
#     # Choose Cells Randomly
#     random.seed(123)
#     cells = random.choices(all_cells, k=len(all_cells))

#     ###### my_cells = set(cells)
#     if perc_def > 0:
#         cells_to_change = int((perc_def/100)*(r*c))
#     else:
#         cells_to_change = 0

#     # Choose from all_cells randomly

#     faulty_cells = remove_duplicates(cells, cells_to_change)

#     ###############################################################################################    
#         # faulty_cells = []

#         # for i in range(cells_to_change):
#             # faulty_cells.append((rows[i], cols[i]))
#     ###############################################################################################

#     # Assign SA0 and SA1 randomly to the randomly chosen cells
    
#     # Fault Distribution
#     SA0 = []
#     SA1 = []

#     # for i in faulty_cells:
#     #     cxbar[i] = random.choice([c_min, c_max])
#     faulty_cells = list(faulty_cells)

#     N = len(faulty_cells)
#     for i in range(N//2):
#         f_xbar[faulty_cells[i][0],faulty_cells[i][1]] = c_min
#         SA0.append((faulty_cells[i][0],faulty_cells[i][1]))
#     for j in range(N//2, N):
#         f_xbar[faulty_cells[j][0],faulty_cells[j][1]] = c_max
#         SA1.append((faulty_cells[j][0],faulty_cells[j][1]))


#     defects["xbar"] = f_xbar
#     defects["all_cells"] = all_cells
#     defects["f_cells"] = faulty_cells
#     defects["SA_0"] = SA0
#     defects["SA_1"] = SA1
    
#     print("Xbar\n",defects["xbar"])
#     print("Fualty\n",defects["all_cells"])
    
#     return defects


# def remove_duplicates(choosen_cells, cells_number):
#     f_cells = set()
    
#     for cell in choosen_cells:
#         if cell not in f_cells:
#             yield(cell)
#             if len(f_cells)!=cells_number:
#                 f_cells.add(cell)
#             else:
#                 break
#     return f_cells


# ## Defect Distribution Analysis
# A plot of the defect distribution in the crossbar array
# 
# Considerations should however be made on the structure of the crossbar array, for simplicity, the positive weighted values have been put to a blue crossbar, while the negative weighted values have been put to yellow crossbars.
# Black crossbars are used as the redundant rows and columns
# 
#     The green cells indicate cells which do not have faults in them
#     Red cells are those that have been affected

# In[ ]:


def xbr_visualization(defects, or_xbar_cols, or_xbar_rows, c_min, mod_pos, mod_neg, mode):
    
    '''
    Visualize the crossbar array with or without defects for every leyer
    
    Parameters:
    -----------
    
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
        
    or_xbar_cols: integer value
        
        The number of columns in the original pretrained weight that we are mapping to the memristor
    
    or_xbar_rows: integer value
        
        The number of rows in the original pretrained weight that we are mapping to the memristor
    
    c_min: float or double
        
        The minimum conductance value of the memristor
    
    mod_pos: an array
    
        Array containing all memristor cells considered to be on the positive side of the crossbar
    
    mod_neg: an array
    
        Array containing all memristor cells considered to be on the negative side of the crossbar
    
        
    mode: can be normal or proposed
        
        normal: The weights be split as all postive values to the left and negative values
                to the right
        proposed: The weights be split in a 4X5 manner, each with two positive columns, followed by
                two corresponding negative column weights and a redundant cell for overcoming the 
                defects
        
     Returns:
    ----------
    Returns a plot of the graph containing the memristor cells
    Those in red are fault cells
    Those in green are the postive cells
    Those in green are the negative cells
    
    '''
    
    # Extract Features from a defected crossbar
    xbar = defects['xbar']
    aff_cells = defects['f_cells']
    all_xbar_cells = defects['all_cells']
    sa0 = defects['SA_0']
    sa1 = defects['SA_1']
    
    r, c = xbar.shape
    # print(f'rows = {r}\nColumns = {c}')
    
    xbar_cells = np.array(all_xbar_cells)
    faulty_ones = np.array(aff_cells)
    
    
    # Distinguish positive and Negative crossbars
    pos_xbr = []
    rec = []
    neg_xbr = []
    for x in all_xbar_cells:
        if xbar[x] == c_min:
            rec.append(x)
            
        if x[1] < or_xbar_cols and x[0]<or_xbar_rows:
            pos_xbr.append(np.array(x))

        if or_xbar_cols <= x[1] < (or_xbar_cols*2) and x[0]<or_xbar_rows:
            neg_xbr.append(np.array(x))    
 
    if mode == 'normal':
        pos__xbr = np.array(pos_xbr)
        neg__xbr = np.array(neg_xbr)
    
    if mode == 'proposed':
        pos__xbr = np.array(mod_pos)
        neg__xbr = np.array(mod_neg)
        
    rec = np.array(rec)
    
    # Demacate SA0 and SA1 Faults
    SA0 = np.array(sa0)
    SA1 = np.array(sa1)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(50, 20))
    
    mpl.rcParams['grid.color'] = 'brown'
    mpl.rcParams['grid.linestyle'] = 'solid'
    mpl.rcParams['grid.linewidth'] = 0.5

    plt.grid(True)
    plt.scatter(xbar_cells[:,1], xbar_cells[:,0],s = 20, marker='s',c='white', label='Redundant cells' )
    
    plt.scatter(pos__xbr[:,1], pos__xbr[:,0],s = 20, c='green', marker= 's', label='Positive Weight cells' )
    plt.scatter(neg__xbr[:,1], neg__xbr[:,0],s = 20,c='blue', marker= 's', label='Negative Weight cells' )
    plt.scatter(rec[:,1], rec[:,0],s = 20, c='white', marker= 's', label='Available for recovery')
#     plt.scatter(faulty_ones[:,1],faulty_ones[:,0], c='red', marker='o')
    if SA0.size > 0:
        plt.scatter(SA0[:,1],SA0[:,0],s = 20, c='red', marker='s', label='SA0 Weight cells')
    if SA1.size > 0:
        plt.scatter(SA1[:,1],SA1[:,0],s = 20, c='yellow', marker='s', label='SA1 Weight cells')
    
    
    # plt.scatter(usable_cells[:,1], usable_cells[:,0],s = 20, marker= 's', label='Usable cells' )
    
    plt.ylim(r,-1)
    plt.xlim(-1,c)
    
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    return plt.show()


# ## Fault Handling
# Many scenarios will be handled in this section depending on the cell behavior and whether or it it is affected
# Both cells in the positive and negative sides of the crossbar array need to be considered, since a fault in one needs to be overcome by the non-faulty cell in the corresponding cell if it is not faulty
# #### SA1:
# #### SA0:

# ### Using the Corresponding Rows and Columns

# In[ ]:


def corr_rows_cols(xbr, defects, c, conductance_values):
    
    '''
    This process uses the existing cells without the need for the redundant rows and columns
    to overcome the faults in the memristor
    
    Parameters:
    ------------
    xbr: matrix or numpy array
    
        This is the crossbar array with additional redundant columns without defects added. 
        It is used for comparison purposes to help find the original values of the affected cells
        
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
    
    c: integer value
    
        The number of columns in the original weight without split to help determine where positive
        and negative values are seperated.
    
    conductance_values: dictionary of memristor resistance and conductance range
            
        Contains the calculated conductance values and range

        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
            
    Return:
    ---------
        
    xbr_f:matrix or numpy array
        The finally rectified crossbar
    
    need_rc:array
        This is an array containing the cells that require redundant columns to overcome the faults
        
    '''
    
    # Extract conductance Values
    c_min = conductance_values["c_min"]
    c_max = conductance_values["c_max"]
    
    # Extract the defects
    xbr_f = defects['xbar']
    SA0 = defects['SA_0']
    SA1 = defects['SA_1']
    
    SA = SA0 + SA1
    need_rc = []
    rc = []
    
    for i in SA:
        # SA0 FAULT HANDLING CRITERIA
        if xbr_f[i]==c_min:
            if i[1] < c:
                if xbr[i] != c_min:
                    if (i[0], i[1]+c) not in SA:
                        xbr_f[i[0],i[1]+c] = c_min
            
            if c<=i[1] < c*2:
                if xbr[i] != c_min:
                    if (i[0], i[1]-c) not in SA:
                        xbr_f[i[0],i[1]-c] = c_min

        # SA1 FAULT HANDLING CRITERIA
        if xbr_f[i] == c_max:
            if i[1]<c:
                if xbr[i] > c_min:
                    if (i[0], i[1]+c) not in SA:
                        
                        if xbr_f[(i[0], i[1]+c)] < c_max - xbr[i] + c_min:
                            
                            xbr_f[(i[0], i[1]+c)] = c_max - xbr[i] + c_min
                
            if c<=i[1]<c*2:
                if xbr[i] > c_min:
                    if (i[0], i[1]-c) not in SA:
                        
                        if xbr_f[(i[0], i[1]-c)] < c_max - xbr[i] + c_min:
                        
                            xbr_f[(i[0], i[1]-c)] = c_max - xbr[i] + c_min
# WORKING CODE
#     for i in SA:
        
#         # SA0 FAULT HANDLING CRITERIA
#         if xbr_f[i]==c_min:
#             if i[1] < c:
#                 if xbr[i] != c_min:
#                     if (i[0], i[1]+c) not in SA:
#                         xbr_f[i[0],i[1]+c] = c_min
                        
#             if c<=i[1] < c*2:
#                 if xbr[i] != c_min:
#                     if (i[0], i[1]-c) not in SA:
#                         xbr_f[i[0],i[1]-c] = c_min

#         # SA1 FAULT HANDLING CRITERIA
#         if xbr_f[i] == c_max:
#             if i[1]<c:
#                 if xbr[i] != c_max:
#                     if (i[0], i[1]+c) not in SA:
#                         if xbr[i] > c_min: 
#                             xbr_f[i[0],i[1]+c] = c_max - xbr[i] + c_min
#                 else: 
#                     xbr_f[i[0],i[1]+c] = c_max
                                

#             if c<=i[1]<c*2:
#                 if xbr[i] != c_max:
#                     if (i[0], i[1]-c) not in SA:
#                         xbr_f[i[0],i[1]-c] = c_max - xbr[i] + c_min
#                 else: 
#                     xbr_f[i[0],i[1]-c] = c_max
    
    return xbr_f


# ### Using the Redundant Rows and Columns

# In[ ]:


def red_rows_cols(xbr, defects, c, conductance_values, seed_val=0):
    
    '''
    This process uses the approach of both redundant columns and the existing cells which
    is an approach tending to use the advantges of both methods while taking advantage of the
    pitfalls of the other method
    
    Parameters:
    ------------
    xbr: matrix or numpy array
    
        This is the crossbar array with additional redundant columns without defects added. 
        It is used for comparison purposes to help find the original values of the affected cells
        
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
    
    c: integer value
    
        The number of columns in the original weight without split to help determine where positive
        and negative values are seperated.
    
    conductance_values: dictionary of memristor resistance and conductance range
            
        Contains the calculated conductance values and range

        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
            
    Return:
    ---------
        
    xbr_f:matrix or numpy array
        The finally rectified crossbar
    
    mapping:array
        This is an array containing the cells and their corresponding redundant columns used in overcoming
        the fault
        
    '''
    
    # Extract conductance Values
    c_min = conductance_values["c_min"]
    c_max = conductance_values["c_max"]
    
    # Extract the defects
    xbr_f = defects['xbar']
    SA_0 = defects['SA_0']
    SA_1 = defects['SA_1']
    
    SA = SA_0 + SA_1
    xbr_cols = xbr_f.shape[1]
    xbr_or = xbr.shape[1]
    red_cols = xbr_cols - c*2
    red_pos = int(red_cols/2)
    mappings = []
    r_c_rec = []
    r_c = 0
    
    if red_pos!=0:
        for i in SA:
            
            np.random.seed(seed_val)

            # SA0 FAULT HANDLING CRITERIA
            if xbr_f[i]==c_min:
                if i[1] < c:
                    if xbr[i] > c_min:
                        r_c = random.choice(np.arange(red_pos))
                        # Corresponding negative cell not faulty or SA_0
                        if(i[0], i[1]+c) not in SA or (i[0], i[1]+c) in SA_0:
                            
                            # Recovery Positive Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) not in SA or (i[0], c*2+r_c+red_pos) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell not faulty, Recovery Negative Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell Faulty-SA_1, Recovery Negative Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c+red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c+red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c+red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])
                                    
                        
                        # Corresponding negative cell SA_1
                        if(i[0], i[1]+c) in SA_1:
                            
                            # Recovery Positive Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA:
                                
                                # Recovery Positive Cell less than Stuck At Value

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])


                if c<=i[1] < c*2:
                    if xbr[i] > c_min:
                        r_c = random.choice(range(red_pos, red_cols))
                        
                        # Corresponding Positive cell not faulty or SA_0
                        if(i[0], i[1]-c) not in SA or (i[0], i[1]-c) in SA_0:
                            
                            # Recovery Negative Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) not in SA or (i[0], c*2+r_c-red_pos) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell not faulty, Recovery Positive Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell Faulty-SA_1, Recovery Positive Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c-red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c-red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c-red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])
                                    
                                    
                        # Corresponding Positive cell SA_1
                        if(i[0], i[1]-c) in SA_1:
                            
                            # Recovery Negative Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) not in SA or (i[0], c*2+r_c-red_pos) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell not faulty, Recovery Positive Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell Faulty-SA_1, Recovery Positive Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c-red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c-red_pos] > c_min:
                                    xbr_f[i[0],c*2+r_c-red_pos] = c_min

                                    mappings.append([i, r_c])

            # SA1 FAULT HANDLING CRITERIA
            if xbr_f[i] == c_max:
                if i[1] < c:
                    if xbr[i] > c_min:
                        r_c = random.choice(np.arange(red_pos))
                        # Corresponding negative cell not faulty or SA_0
                        if(i[0], i[1]+c) not in SA or (i[0], i[1]+c) in SA_0:
                            
                            # Recovery Positive Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) not in SA or (i[0], c*2+r_c) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c+red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c+red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell not faulty, Recovery Negative Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell Faulty-SA_1, Recovery Negative Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c+red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c+red_pos] < c_max:
                                    xbr_f[i[0],c*2+r_c+red_pos] = c_max

                                    mappings.append([i, r_c])
                                    
                        # Corresponding negative cell SA_1
                        if(i[0], i[1]+c) in SA_1:
                            
                            # Recovery Positive Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) not in SA or (i[0], c*2+r_c) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell not faulty, Recovery Negative Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c+red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Positive Cell Faulty-SA_1, Recovery Negative Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c+red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c+red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c+red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])


                if c<=i[1] < c*2:
                    if xbr[i] > c_min:
                        
                        r_c = random.choice(range(red_pos, red_cols))
                        
                        # Corresponding Positive cell not faulty or SA_0
                        if(i[0], i[1]-c) not in SA or (i[0], i[1]-c) in SA_0:
                            
                            # Recovery Negative Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) not in SA or (i[0], c*2+r_c) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c-red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c-red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell not faulty, Recovery Positive Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell Faulty-SA_1, Recovery Positive Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c-red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c-red_pos] < c_max:
                                    xbr_f[i[0],c*2+r_c-red_pos] = c_max

                                    mappings.append([i, r_c])
                        
                        # Corresponding Positive cell SA_1            
                        if(i[0], i[1]-c) in SA_1:
                            
                            # Recovery Negative Cell or Recovery Negative Cell not Faulty
                            if (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) not in SA or (i[0], c*2+r_c-red_pos) in SA_0:
                                
                                # Recovery Positive Cell less than Cell Value

                                if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                    xbr_f[i[0],c*2+r_c] = xbr[i]

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell not faulty, Recovery Positive Cell Faulty
                            elif (i[0], c*2+r_c) not in SA and (i[0], c*2+r_c-red_pos) in SA_1:                      

                                if xbr_f[i[0],c*2+r_c] < c_max:
                                    xbr_f[i[0],c*2+r_c] = c_max

                                    mappings.append([i, r_c])
                                    
                            # Recovery Negative Cell Faulty-SA_1, Recovery Positive Cell  not Faulty
                            elif (i[0], c*2+r_c) in SA_1 and (i[0], c*2+r_c-red_pos) not in SA:                      

                                if xbr_f[i[0],c*2+r_c-red_pos] < c_max - xbr[i] + c_min:
                                    xbr_f[i[0],c*2+r_c-red_pos] = c_max - xbr[i] + c_min

                                    mappings.append([i, r_c])

            
    return xbr_f, mappings 


# ### Combined Redundant Rows and Columns and the Corresponding Rows and Columns

# In[ ]:


def combined(xbr, defects,  c, conductance_values, seed_val=0):
    
    '''
    This process uses the approach of both redundant columns and the existing cells which
    is an approach tending to use the advantges of both methods while taking advantage of the
    pitfalls of the other method
    
    Parameters:
    ------------
    xbr: matrix or numpy array
            This is the crossbar array with additional redundant columns without defects added. 
            It is used for comparison purposes to help find the original values of the affected cells
        
    defects:dictionary
            This contains the faulty crossbars, the cells with specific faults
            defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
            defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
            defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
    
    c: integer value
            The number of columns in the original weight without split to help determine where positive
            and negative values are seperated.
    
    conductance_values: dictionary of memristor resistance and conductance range
            Contains the calculated conductance values and range
            
            c_min      = conductance_values['c_min']  .............. Minimum Conductance
            c_max      = conductance_values['c_max']  .............. Maximum Conductance
            cond_range = conductance_values['c_range'].............. Conductance Range
            
    Return:
    ---------
    output: a dictionary for output
        
        output["f_xbr"]:      ----------- The finally rectified crossbar
        output["both_faulty"]:----------- Cells which are faulty on both the positive and negaative sides
        output["mapping"]:    ----------- This is an array containing the cells and their corresponding 
                                          redundant columns used in overcoming the fault        
    '''
    
    # Extract conductance Values
    c_min = conductance_values["c_min"]
    c_max = conductance_values["c_max"]
    
    # Extract the defects
    xbr_f = defects['xbar']
    SA_0 = defects['SA_0']
    SA_1 = defects['SA_1']
    
    SA = SA_0 + SA_1
    xbr_cols = xbr_f.shape[1]
    xbr_or = xbr.shape[1]
    red_cols = xbr_cols - c*2
    red_pos = int(red_cols/2)
    
    np.random.seed(seed_val)
    mappings = []
    faluty_rc = []
    both_faulty = []
    r_c_rec = []
    r_c = 0
    
    if len(SA)!=0:
        
#         for x in SA:
#         # Identifying Faulty Cells and their corresponding ones
#             for j in SA:
#                 # Find Cells in the same row which are faulty
#                 if x[0] == j[0]:

#                     # Check if the cell falls under redundant Columns
#                     if j[1] >= c*2:
#                         faluty_rc.append(j)

#                     # Check for Faulty Cells Whose Corresponding cells are faulty too
#                     if x[1] == j[1]-c or x[1] == j[1]+c or x[1]-c == j[1] or x[1]+c == j[1]:
#                         if x not in both_faulty and j not in both_faulty:
#                             both_faulty.append(x)
#                             both_faulty.append(j)
    
        for i in SA:

            # SA0 FAULT HANDLING CRITERIA
            if xbr_f[i]==c_min:
                # Consider A Positive Cell Stuck At Zero
                if i[1] < c:
                    if xbr[i] > c_min:
                      
                        r_c = random.choice(np.arange(red_pos))
                        # Case Where the Negative Cell is not Stuck.
                        # Can be used to overcome a case where the positive redundant cell is faulty
                        if (i[0], i[1]+c) not in SA:
                            if (i[0], c*2+r_c) not in SA:
                                if (i[0], c*2+r_c+ red_pos) not in SA or (i[0], c*2+r_c+ red_pos) in SA_0:
                                    if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                        xbr_f[i[0],c*2+r_c] = xbr[i]
                                        mappings.append([i, r_c])
                                
                                # elif (i[0], c*2+r_c+ red_pos) in SA_1:
                                    # xbr_f[i[0],c*2+r_c] = c_max
                                    # mappings.append([i, r_c])
                            
                            elif (i[0], c*2+r_c) in SA_1:

                                if (i[0], c*2+r_c + red_pos) not in SA or (i[0], c*2+r_c + red_pos) in SA_0:
                                    if xbr_f[i[0], i[1]+c] < c_max - xbr[i] + c_min:
                                        xbr_f[i[0], i[1]+c] = c_max - xbr[i] + c_min
                                        mappings.append([i, r_c])

                        elif (i[0], i[1]+c) in SA_1 and ((i[0], c*2+r_c+ red_pos) not in SA or (i[0], c*2+r_c+ red_pos) in SA_0):

                            if (i[0], c*2+r_c) not in SA:
                                xbr_f[i[0],c*2+r_c] = c_max
                                mappings.append([i, r_c])
                            
                            elif (i[0], c*2+r_c) in SA_1:
                                mappings.append([i, r_c])
                                            
                        elif (i[0], i[1]+c) in SA_0:

                            if (i[0], c*2+r_c) not in SA:

                                if (i[0], c*2+r_c+ red_pos) not in SA or (i[0], c*2+r_c+ red_pos) in SA_0:

                                    if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                        xbr_f[i[0],c*2+r_c] = xbr[i]
                                        mappings.append([i, r_c])
                                
                                # elif (i[0], c*2+r_c+ red_pos) in SA_1:
                                    # if xbr_f[i[0],c*2+r_c] < c_max:
                                        # xbr_f[i[0],c*2+r_c] = c_max
                                        # mappings.append([i, r_c])
                            
                            elif (i[0], c*2+r_c) in SA_1:

                                if (i[0], c*2+r_c + red_pos) not in SA:

                                    if xbr_f[i[0],c*2+r_c+ red_pos] < c_max - xbr[i] + c_min:
                                        xbr_f[i[0],c*2+r_c+ red_pos] = c_max - xbr[i] + c_min
                                        mappings.append([i, r_c])
                
                # Consider A Negative Cell Stuck at Zero
                if c<=i[1] < c*2:
                    if xbr[i] > c_min:

                        r_c = random.choice(np.arange(red_pos, red_cols))
                        
                        # Check Out The corresponding Positive cell
                        if (i[0], i[1]-c) not in SA:

                            if (i[0], c*2+r_c) not in SA:

                                if (i[0], c*2+r_c - red_pos) not in SA or (i[0], c*2+r_c - red_pos) in SA_0:

                                    if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                        xbr_f[i[0],c*2+r_c] = xbr[i]
                                        mappings.append([i, r_c])
                                
                                # elif (i[0], c*2+r_c - red_pos) in SA_1:
                                    # if xbr_f[i[0],c*2+r_c] < c_max:
                                        # xbr_f[i[0],c*2+r_c] = c_max
                                        # mappings.append([i, r_c])
                            
                            elif (i[0], c*2+r_c) in SA_1:

                                if (i[0], c*2+r_c - red_pos) not in SA or (i[0], c*2+r_c - red_pos) in SA_0:
                                    
                                    if xbr_f[i[0], i[1]-c] < c_max - xbr[i] + c_min:
                                        
                                        xbr_f[i[0], i[1]-c] = c_max - xbr[i] + c_min
                                        mappings.append([i, r_c])

                        elif (i[0], i[1]-c) in SA_0:

                            if (i[0], c*2+r_c) not in SA:

                                if (i[0], c*2+r_c - red_pos) not in SA or (i[0], c*2+r_c - red_pos) in SA_0:

                                    if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                        xbr_f[i[0],c*2+r_c] = xbr[i]
                                        mappings.append([i, r_c])
                                
                                # elif (i[0], c*2+r_c - red_pos) in SA_1:
                                    # xbr_f[i[0],c*2+r_c] = c_max
                                    # mappings.append([i, r_c])
                            elif (i[0], c*2+r_c) in SA_1:

                                if (i[0], c*2+r_c - red_pos) not in SA:

                                    if xbr_f[i[0],c*2+r_c - red_pos] < c_max - xbr[i] + c_min:
                                        xbr_f[i[0],c*2+r_c - red_pos] = c_max - xbr[i] + c_min
                                        mappings.append([i, r_c])

                        elif (i[0], i[1]-c) in SA_1:

                            if (i[0], c*2+r_c) not in SA:
                                
                                if (i[0], c*2+r_c - red_pos) not in SA and  xbr_f[i[0], c*2+r_c - red_pos] == c_min:

                                    if xbr_f[i[0], c*2+r_c - red_pos] >= c_min:
                                        xbr_f[i[0],c*2+r_c] = c_max
                                        mappings.append([i, r_c])
                                    
                            elif (i[0], c*2+r_c) in SA_1:

                                # if (i[0], c*2+r_c - red_pos) not in SA and xbr_f[i[0], c*2+r_c - red_pos] == c_min:
                                mappings.append([i, r_c])

            # SA1 FAULT HANDLING CRITERIA
            if xbr_f[i] == c_max:
                if i[1] < c:
                    if xbr[i] > c_min:
                        if red_pos == 0:
                            pass
                        else:
                            r_c = random.choice(np.arange(red_pos))

                            # All the other recovery cells are not faulty
                            if (i[0], i[1]+c) not in SA:

                                xbr_f[i[0], i[1]+c] = c_max - xbr[i] + c_min

                            else:                                
                                # Corresponding Negative cell is not faulty
                                # if (i[0], i[1]+c) in SA_0:

                                    # Non Faulty Redundant Positive Column
                                    # if (i[0], c*2+r_c) not in SA:
                                        # Faulty Redundant Negative Column - SA-1
                                        # if (i[0], c*2+r_c+ red_pos) in SA_1:

                                            # if xbr_f[i[0], c*2+r_c] < xbr[i]:
                                                # xbr_f[i[0], c*2+r_c] = xbr[i]
                                                # mappings.append([i, r_c])
                                                
                                # Corresponding Negative Cell Faulty SA-0
                                if (i[0], i[1]+c) in SA_0:
                                    # Redundant Negative Column Cell not Faulty
                                    if (i[0], c*2+r_c+ red_pos) not in SA:
                                        # Redundant Positive Column Cell Not Faulty
                                        if (i[0], c*2+r_c) not in SA or (i[0], c*2+r_c) in SA_0:
                                            if xbr_f[i[0],c*2+r_c + red_pos] < c_max - xbr[i] + c_min:
                                                xbr_f[i[0],c*2+r_c+ red_pos] = c_max - xbr[i] + c_min
                                                mappings.append([i, r_c])

                                    # Redundant Negative Column Cell Faulty - SA-1
                                    elif (i[0], c*2+r_c+ red_pos) in SA_1:
                                        if (i[0], c*2+r_c) not in SA:
                                            if xbr_f[i[0], c*2+r_c] < xbr[i]:
                                                xbr_f[i[0], c*2+r_c] = xbr[i]
                                                mappings.append([i, r_c])
                                                
                                            elif (i[0], c*2+r_c) in SA_0:
                                                mappings.append([i, r_c])

                                    # Redundant Positive Column Cell Faulty - SA-1
                                    # elif (i[0], c*2+r_c) in SA_1:

                                        # Redundant Negative Column Cell Not Faulty
                                        # if (i[0], c*2+r_c + red_pos) not in SA:

                                            # if xbr_f[i[0],c*2+r_c+ red_pos] < c_max:
                                                # xbr_f[i[0],c*2+r_c+ red_pos] = c_max
                                                # mappings.append([i, r_c])

                                    # Redundant Positive Column Cell Faulty - SA-0
                                    elif (i[0], c*2+r_c) in SA_0:

                                        if (i[0], c*2+r_c + red_pos) not in SA:

                                            if xbr_f[i[0],c*2+r_c+ red_pos] < c_max - xbr[i] + c_min:
                                                xbr_f[i[0],c*2+r_c+ red_pos] = c_max - xbr[i] + c_min
                                                mappings.append([i, r_c])
                                                
                                        elif (i[0], c*2+r_c + red_pos) in SA_1:
                                            mappings.append([i, r_c])

                                # Corresponding Negative Cell Faulty - SA-1
                                elif (i[0], i[1]+c) in SA_1:

                                    # Redundant Positive Cell not Faulty
                                    if (i[0], c*2+r_c) not in SA:

                                        # Redundant Negative Cell Not Faulty or SA-0
                                        if (i[0], c*2+r_c+ red_pos) not in SA or (i[0], c*2+r_c+ red_pos) in SA_0:

                                            if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                                xbr_f[i[0],c*2+r_c] = xbr[i]
                                                mappings.append([i, r_c])

                                        # Redundant Negative Cell Faulty - SA-1
                                        # if (i[0], c*2+r_c+ red_pos) in SA_1:

                                            # if xbr_f[i[0],c*2+r_c] < c_max:

                                                # xbr_f[i[0],c*2+r_c] = c_max
                                                # mappings.append([i, r_c])

                                    # Redundant Positive cell Faulty -  SA-1
                                    elif (i[0], c*2+r_c) in SA_1:

                                        # Redundant Negative Cell not Faulty
                                        if (i[0], c*2+r_c + red_pos) not in SA:

                                            if xbr_f[i[0],c*2+r_c+ red_pos] < c_max - xbr[i] + c_min:
                                                xbr_f[i[0],c*2+r_c+ red_pos] = c_max - xbr[i] + c_min
                                                mappings.append([i, r_c])


                if c<=i[1] < c*2:
                    # Negative Cell Faulty
                    if xbr[i] < c_max:
                        if red_pos == 0:
                            pass
                        else:
                            r_c = random.choice(np.arange(red_pos, red_cols))

                            # All other Cells are fault Free
                            if (i[0], i[1]-c) not in SA:

                                xbr_f[i[0], i[1]-c] = c_max - xbr[i] + c_min

                            else:
                                # Positive Cell Not Faullty
#                                 if (i[0], i[1]-c) not in SA:

#                                     # Negative Redundant Cell Not Faulty
#                                     if (i[0], c*2+r_c) not in SA:

#                                         # Positive Redundant Cell Faulty - SA-1
#                                         if (i[0], c*2+r_c - red_pos) in SA_1:
#                                             if xbr_f[i[0],c*2+r_c] < xbr[i]:                                                
#                                                 xbr_f[i[0],c*2+r_c] = c_max - xbr[i] + c_min
#                                                 mappings.append([i, r_c])

#                                         # Positive Redundant Cell Faulty - SA-0
#                                         if (i[0], c*2+r_c - red_pos) in SA_0:
#                                             if xbr_f[i[0], i[1]-c] < c_max - xbr[i] + c_min:
#                                                 xbr_f[i[0], i[1]-c] = c_max - xbr[i] + c_min

#                                     # Negative Redundant Cell Faulty - SA-1
#                                     elif (i[0], c*2+r_c) in SA_1:
#                                         if (i[0], c*2+r_c - red_pos) not in SA:
#                                             if xbr_f[i[0],c*2+r_c - red_pos] < c_max :
#                                                 xbr_f[i[0],c*2+r_c - red_pos] = c_max
#                                                 xbr_f[i[0], i[1]-c] = c_max - xbr[i] + c_min
#                                                 mappings.append([i, r_c])

#                                     # Negative Redundant Cell Faulty - SA-0
#                                     elif (i[0], c*2+r_c) in SA_0:

#                                         if (i[0], c*2+r_c - red_pos) not in SA:
#                                             xbr_f[i[0], i[1]-c] = c_max - xbr[i] + c_min
                                
                                # Positive Cell Faulty - SA-0
                                if (i[0], i[1]-c) in SA_0:
                                    # Positive Redundant Cell not Faulty
                                    if (i[0], c*2+r_c - red_pos) not in SA:
                                        # Redundant Negative Cell not Faulty
                                        if (i[0], c*2+r_c) not in SA or (i[0],c*2+r_c) in SA_0:
                                            if xbr_f[i[0],c*2+r_c - red_pos] < c_max - xbr[i] + c_min:
                                                xbr_f[i[0],c*2+r_c - red_pos] = c_max - xbr[i] + c_min
                                                mappings.append([i, r_c])

                                        # Positive Redundant Cell Faulty - SA-1
                                    elif (i[0], c*2+r_c - red_pos) in SA_1:
                                        if (i[0], c*2+r_c) not in SA:
                                            if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                                xbr_f[i[0],c*2+r_c] = xbr[i]
                                                mappings.append([i, r_c])
                                                
                                        elif (i[0], c*2+r_c) in SA_0:
                                            mappings.append([i, r_c])

                                        # Positive Redundant Cell Faulty - SA-0
                                        # elif (i[0], c*2+r_c - red_pos) in SA_0:

                                            # if xbr_f[i[0],c*2+r_c] > c_min:
                                                # xbr_f[i[0],c*2+r_c] = c_min
                                                # mappings.append([i, r_c])

                                    # Negative Redundant Cell Faulty - SA-1
                                    # elif (i[0], c*2+r_c) in SA_1:

                                        # Positive Redundant Cell Not Faulty
                                        # if (i[0], c*2+r_c - red_pos) not in SA:

                                            # if xbr_f[i[0],c*2+r_c - red_pos] < c_max :
                                                # xbr_f[i[0],c*2+r_c - red_pos] = c_max
                                                # mappings.append([i, r_c])


                                # Positive Cell Faulty - SA-1
                                if (i[0], i[1]-c) in SA_1:

                                    # Negative Redundant Cell Not Faulty
                                    if (i[0], c*2+r_c) not in SA:

                                        # Positive Redundant Cell Not Faulty or SA-0
                                        if (i[0], c*2+r_c - red_pos) not in SA or (i[0], c*2+r_c - red_pos) in SA_0:

                                            if xbr_f[i[0],c*2+r_c] < xbr[i]:
                                                xbr_f[i[0],c*2+r_c] = xbr[i]
                                                mappings.append([i, r_c])

                                        # Positive Redundant Cell Faulty - SA-1
                                        # if (i[0], c*2+r_c - red_pos) in SA_1:
                                            # if xbr_f[i[0],c*2+r_c] < c_max:
                                                # xbr_f[i[0],c*2+r_c] = c_max
                                                # mappings.append([i, r_c])

                                    # Negative Redundant Cell Faulty - SA-1
                                    if (i[0], c*2+r_c) in SA_1:

                                        # Positive Redundant Cell Not Faulty
                                        if (i[0], c*2+r_c - red_pos) not in SA:
                                            if xbr_f[i[0], c*2+r_c - red_pos] < c_max - xbr[i] + c_min:
                                                xbr_f[i[0],c*2+r_c - red_pos] = c_max - xbr[i] + c_min
                                                mappings.append([i, r_c])
                
    output = {
        "f_xbr":xbr_f,
        # "both_faulty":both_faulty,
        "mapping":mappings
    }
    
    return output


# In[ ]:


def proposed_approach(xbar_or, defects, recovery_cells, conductance_values, endval):# xbr_f, xbr, SA_0, SA_1,  c, c_min, c_max, group_cell, pos, neg, red):
    
    '''    
    Rectify the faults on the proposed split crossbar of 4X5. One redundant cell is used to overcome
    a fault that cannot be done by the four cells mostly if both the positive and negative corresponding
    cells are faulty
    
    Parameters:
    -------------
    
    defects:dictionary
            This contains the faulty crossbars, the cells with specific faults
            defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
            defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
            defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
            
    xbar: matrix or numpy array
            This is the crossbar array with 4X5 splits without defects added. 
            It is used for comparison purposes to help find the original values of the affected cells
            
    conductance_values: dictionary of memristor resistance and conductance range
            Contains the calculated conductance values and range
            
            c_min      = conductance_values['c_min']  .............. Minimum Conductance
            c_max      = conductance_values['c_max']  .............. Maximum Conductance
            cond_range = conductance_values['c_range'].............. Conductance Range
            
    Return:
    ---------
    
    xbar_f:matrix or numpy array
            
            Crossbar with cells that have been rectified
            
    cell_mapping:numpy array
            
    '''
    # Extract conductance Values
    c_min = conductance_values['c_min']
    c_max = conductance_values['c_max']
    
    # Extract the defects
    xbar_f = defects['xbar']
    SA_0 = defects['SA_0']
    SA_1 = defects['SA_1']
    
    fc = SA_0 + SA_1
    # print("defects['f_cells'] == fc: {}".format(faulty_cells == fc))
    # print("defects['f_cells']: {}\nfc: {}".format(faulty_cells,fc))
    # print("SA_1: {}, Cells: {}".format(len(SA_1),SA_1))
    cell_mapping = []
    
    # Writing a positive Value to a positive affected cell
    # print("Faulty grouped: ", cells_considered)
    
    if len(recovery_cells) !=0:
        
        for mod_cell in recovery_cells:
            pos_cell = mod_cell[0]
            neg_cell = mod_cell[1]
            rec_pos = mod_cell[2]
            rec_neg = mod_cell[3]
            
            if pos_cell[1] < endval:
#                 print('endval: ',endval)
#                 print('pos_cell: ',pos_cell)
#                 print('neg_cell: ',neg_cell)
#                 print('rec_pos: ',rec_pos)
#                 print('rec_neg: ',rec_neg)
                # POSITITVE CELL FAULTY
        
                # Positive Cell Faulty SA_0
                if pos_cell in SA_0:
                    if xbar_or[pos_cell] != c_min:
                        # Redundant Column Cell not Faulty
                        if rec_pos not in fc:
                            if rec_neg not in fc or rec_neg in SA_0:
                                # Use Redundant cell to overcome the fault
                                if xbar_f[rec_pos] < xbar_or[pos_cell]:
                                    xbar_f[rec_pos] = xbar_or[pos_cell]
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                        # Redundant Column Cell Faulty - SA-1
                        # No Changes can be preformed when the redundant cell is faulty at SA-0
                        elif rec_pos in SA_1:                            
                            if rec_neg not in fc or rec_neg in SA_0:
                                # Negative Corresponding Cell not Faulty
                                if neg_cell not in fc:
                                    if xbar_f[neg_cell] < c_max - xbar_or[pos_cell] + c_min:
                                        xbar_f[neg_cell] = c_max - xbar_or[pos_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                                # Negative Corresponding Cell Faulty - SA-0
                                elif neg_cell in SA_0:
                                    if xbar_f[rec_neg] < c_max - xbar_or[pos_cell] + c_min and rec_neg not in fc:
                                        xbar_f[rec_neg] = c_max - xbar_or[pos_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                                # Negative Corresponding Cell Faulty - SA-1
                                elif neg_cell in SA_1:
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                # Positive Cell Faulty - SA-1                    
                elif pos_cell in SA_1:
                    if xbar_or[pos_cell] != c_min:
                        if neg_cell not in fc:
                            # Use Redundant cell to overcome the fault
                            if xbar_f[neg_cell] < c_max - xbar_or[pos_cell] + c_min:
                                xbar_f[neg_cell] = c_max - xbar_or[pos_cell] + c_min

                        # When the Corresponding negative cell is faulty
                        # We can only reduce the error margin by setting the redundant cell to c_min
                        elif neg_cell in SA_1:
                            if rec_pos not in fc:
                                if rec_neg not in fc or rec_neg in SA_0:
                                    if xbar_f[rec_pos] < xbar_or[pos_cell]:
                                        xbar_f[rec_pos] = xbar_or[pos_cell]
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            elif rec_pos in SA_1:
                                if rec_neg not in fc:
                                    if xbar_f[rec_neg] < c_max - xbar_or[pos_cell] + c_min:
                                        xbar_f[rec_neg] = c_max - xbar_or[pos_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                        elif neg_cell in SA_0:
                            if rec_neg not in fc and (rec_pos not in fc or rec_pos in SA_0):
                                if xbar_f[rec_neg] < c_max - xbar_or[pos_cell] + c_min:
                                    xbar_f[rec_neg] = c_max - xbar_or[pos_cell] + c_min
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            elif rec_neg in SA_1:
                                if rec_pos not in fc:
                                    if xbar_f[rec_pos] < xbar_or[pos_cell]:
                                        xbar_f[rec_pos] = xbar_or[pos_cell]
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                                if rec_pos in SA_0:
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])


                # NEGATIVE CELL FAULTY

                # Negative Cell Faulty - SA-0
                elif neg_cell in SA_0:
                    if xbar_or[neg_cell] != c_min:
                        # Redundant Column Cell not Faulty
                        if rec_neg not in fc:
                            if rec_pos not in fc or rec_pos in SA_0:
                                # Use Redundant cell to overcome the fault
                                # Positive Corresponding Cell not Faulty
                                if pos_cell not in fc or pos_cell in SA_0:
                                    if xbar_f[rec_neg] < xbar_or[neg_cell]:
                                        xbar_f[rec_neg] = xbar_or[neg_cell]
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                                if pos_cell in SA_1:
                                    if xbar_f[rec_neg] < c_max:
                                        xbar_f[rec_neg] = c_max
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                        # Redundant Column Cell Faulty - SA-1
                        # No Changes can be preformed when the redundant cell is faulty at SA-0
                        elif rec_neg in SA_1:
                            if pos_cell not in fc:
                                if rec_pos not in fc or rec_pos in SA_0:
                                    if xbar_f[pos_cell] < c_max - xbar_or[neg_cell] + c_min:
                                        xbar_f[pos_cell] = c_max - xbar_or[neg_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            elif pos_cell in SA_0:
                                if rec_pos not in fc:
                                    if xbar_f[rec_pos] < c_max - xbar_or[neg_cell] + c_min:
                                        xbar_f[rec_pos] = c_max - xbar_or[neg_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            elif pos_cell in SA_1:
                                if rec_pos not in fc or rec_pos in SA_0:
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                # Negative Cell Faulty - SA-1                    
                elif neg_cell in SA_1:
                    if xbar_or[neg_cell] != c_min:
                        if pos_cell not in fc:
                            # Use Redundant cell to overcome the fault
                            if xbar_f[pos_cell] < c_max - xbar_or[neg_cell] + c_min:
                                xbar_f[pos_cell] = c_max - xbar_or[neg_cell] + c_min
                                # cell_mapping.append([pos_cell, neg_cell])

                        # Corresponding positive cell is faulty - SA-0
                        elif pos_cell in SA_0:
                            if rec_pos not in fc:
                                # Redundant Cell not Faulty
                                # No change can be made if the cell is faulty
                                if rec_neg not in fc or rec_neg in SA_0:
                                    if xbar_f[rec_pos] < c_max - xbar_or[neg_cell] + c_min: 
                                        xbar_f[rec_pos] = c_max - xbar_or[neg_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            elif rec_pos in SA_1:
                                if rec_neg not in fc:
                                    if xbar_f[rec_neg] < xbar_or[neg_cell]: 
                                        xbar_f[rec_neg] = xbar_or[neg_cell]
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                                elif rec_neg in SA_0:
                                    cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                        # Corresponding Positive Cell Faulty - SA-1
                        elif pos_cell in SA_1:
                            if rec_neg not in fc:
                                # Redundant Cell not Faulty
                                if rec_pos not in fc or rec_pos in SA_0:
                                    if xbar_f[rec_neg] < xbar_or[neg_cell]: 
                                        xbar_f[rec_neg] = xbar_or[neg_cell]
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])

                            # Redundant Cell not Faulty
                            elif rec_neg in SA_1:
                                if rec_pos not in fc:
                                    if xbar_f[rec_pos] < c_max - xbar_or[neg_cell] + c_min: 
                                        xbar_f[rec_pos] = c_max - xbar_or[neg_cell] + c_min
                                        cell_mapping.append([pos_cell, neg_cell, rec_pos, rec_neg])
                                    
    else:
    # No Faulty Cells, Keep the cell values
        xbar_f = xbar_or
        cell_mapping = []
                    
    return xbar_f, cell_mapping


# ## Number of bits
# If we need to consider the number of bits that our memristor should be able to store, we need to define the number of bits so that the weight values are reshaped to match the bit value in the memristor conductance values.
# Memristors have not fully been tested for multiple bit precision, however, this being part of the test issue in the reserach, we have to consider this fact and check for how well it can perform with regardss to the bit levels changed.
# This will generally change the weight values to fit into the bit ranges as defined by the number of bits

# In[ ]:


def bit_level_precision(conductance_values, 
                        w_con, 
                        n, 
                        max_w_b):
    
    '''
    This converts the weights passed to it to take up the bit levels as defined
    The number of bits alter the conductance value ranges and conductance values mapped to the crossbar
    
    Parameters:
    ------------
    conductance_values: dictionary of memristor resistance and conductance range
            
        Contains the calculated conductance values and range

        c_min      = conductance_values['c_min']  .............. Minimum Conductance
        c_max      = conductance_values['c_max']  .............. Maximum Conductance
        cond_range = conductance_values['c_range'].............. Conductance Range
        
    w_con: matrix or numpy array
            This is the final crossbar array with with faults fixed. 
            It is converted to correspond to the number of bits that we would like the memristor to hold
        
    n: integer value
            The number of bits to represent the memristor bits storage level.
            The more the number of bits, the better the accuracy as manifested in some experiments
            
    max_val: float or double
        The maximum absolute value in the weight matrix to be used in determining the conductance
        split
            
    Return:
    ---------
    w: matrix or numpy array

        The finally rectified crossbar

    div_pattern: array

        An array containing the division pattern of the memristor. This depends entirely on the number of
        bits that we specify. The number of divisions is equivalent to 2^n - 1 where n is the number of bits
        
    '''
    # Extract Conductance values
    min_cond = conductance_values['c_min']
    max_cond = conductance_values['c_max']
    cond_rng = conductance_values['c_range']
    
    divs = pow(2, n) - 1
    div_pattern = np.linspace(min_cond, max_cond, divs)
    F_w = (((w_con - min_cond)/cond_rng)* divs).round()
    
    w_bits_con = ((F_w* cond_rng)/divs)+min_cond

    # Alternatively
    # divs = pow(2, n)
    # div_pattern = np.linspace(min_cond, max_cond, divs)
    # F_w = ((((w_con - min_cond)/cond_rng)* divs)- 0.5).round()

    # w_bits_con = (((F_w + 0.5)* cond_rng)/divs)+min_cond 
    
    # w = ((w_bits_con - min_cond)/cond_rng)*max_w_b

    #Weight Adjustment - Corresponding to the Conductance Range
    # w_bits = adjust_con_weights(w_bits_con, div_pattern)
    
    w = ((w_bits_con - min_cond)/cond_rng)*max_w_b

    # return w_bits, div_pattern
    return w, div_pattern


# ## Final Stage of output
# This is the output of every model described for overcoming the faults. They are reshaped to take on dimensions equivalent to the original weight dimensions to allow for easy model running

# In[ ]:


def xbar_output(xba, w_r, w_c):
    
    '''
    Visualize the crossbar array with or without defects for every leyer
    
    Parameters:
    -----------
    
    w_r: integer value
        
        The number of rows in the original pretrained weight that we are mapping to the memristor
    
    w_c: integer value
        
        The number of columns in the original pretrained weight that we are mapping to the memristor
        
     Returns:
    ----------
    xba_final: An n-d array or matrix
        
        An array of conductance values that is of the same size as the originally pretrained weight
        ready for testing using the prediction model of the neural network during the evaluation of
        it's performance
    
    '''    
    tot_rows, tot_cols = xba.shape
    r_cols = tot_cols - w_c*2
    r_pos = r_cols//2
    
    xba_final = []
    
    pos_xbr_cells = slice(0, w_c)
    neg_xbr_cells = slice(w_c, w_c*2)
        
    xba_final = xba[:,pos_xbr_cells]-xba[:,neg_xbr_cells]
    
    return xba_final[:w_r+1,:]


# In[ ]:


def xbar_output_red_col(xba, w_r, w_c, red_cols, defects, all_cells):
    
    '''
    Visualize the crossbar array with or without defects for every leyer
    
    Parameters:
    -----------
    
    w_r: integer value
        
        The number of rows in the original pretrained weight that we are mapping to the memristor
    
    w_c: integer value
        
        The number of columns in the original pretrained weight that we are mapping to the memristor
        
    red_cols: an array
    
        Array containing all memristor cells which are the redundant columns used in overcoming the faults
        in the crossbar
    
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
           
    f_both: an array
    
        Array containing all memristor cells which are considered faulty on both the positive and negative side
        They are considered irrecoverable
        
    all_cells: an array
    
        Array containing all memristor cells in the crossbar
        
    
    f_gp_cells: an array
    
        Array containing all faulty memristor cells grouped with the redundant cells used for overcoming the fault
    
     Returns:
    ----------
    xba_final: An n-d array or matrix
        
        An array of conductance values that is of the same size as the originally pretrained weight
        ready for testing using the prediction model of the neural network during the evaluation of
        it's performance
    
    '''
    
    # Extract Defects
    SA0 = defects['SA_0']
    SA1 = defects['SA_1']
    
    tot_rows, tot_cols = xba.shape
    r_cols = tot_cols - w_c*2
    r_pos = int(r_cols/2)
    f_c = SA0 + SA1
    
    global xba_final
    
    pos_xbr_cells = slice(0, w_c)
    neg_xbr_cells = slice(w_c, w_c*2)
    
    xba_final = xbar_output(xba, tot_rows, w_c)
    
    if len(red_cols) != 0:
        
        # print("red_cols: ", red_cols)
        
        for r_cell in red_cols:
            
            # print("r_cell: ", r_cell)
        
        # Check if the cell is among the grouped cells which use redundant columns

            if r_cell[0][1] < w_c:
                if r_cell[1] < r_pos:

                    xba_final[r_cell[0][0], r_cell[0][1]] = xba_final[r_cell[0][0], r_cell[0][1]] + xba[r_cell[0][0],r_cell[1]+w_c*2]- xba[r_cell[0][0],r_cell[1]+w_c*2+r_pos]

                if r_pos <= r_cell[1] < r_cols:

                    xba_final[r_cell[0][0], r_cell[0][1]] = xba_final[r_cell[0][0], r_cell[0][1]] - xba[r_cell[0][0],r_cell[1]+w_c*2] + xba[r_cell[0][0],r_cell[1]+w_c*2-r_pos]

            if w_c <= r_cell[0][1] < 2*w_c:

                if r_cell[1] < r_pos:

                    xba_final[r_cell[0][0], r_cell[0][1]-w_c] = xba_final[r_cell[0][0], r_cell[0][1]-w_c] + xba[r_cell[0][0],r_cell[1]+w_c*2]- xba[r_cell[0][0],r_cell[1]+w_c*2+r_pos]

                if r_pos <= r_cell[1] < r_cols:

                    xba_final[r_cell[0][0], r_cell[0][1]-w_c] = xba_final[r_cell[0][0], r_cell[0][1]-w_c] - xba[r_cell[0][0],r_cell[1]+w_c*2] + xba[r_cell[0][0],r_cell[1]+w_c*2-r_pos]

    return xba_final[:w_r+1,:]


# In[ ]:


def xbar_output_combined(xba, w_r, w_c, cell_map, defects, all_cells):
    
    '''
    Visualize the crossbar array with or without defects for every leyer
    
    Parameters:
    -----------
    
    w_r: integer value
        
        The number of rows in the original pretrained weight that we are mapping to the memristor
    
    w_c: integer value
        
        The number of columns in the original pretrained weight that we are mapping to the memristor
        
    cell_map: an array
    
        Array containing all memristor cells which are the redundant columns used in overcoming the faults
        in the crossbar
    
    
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
        
    all_cells: an array
    
        Array containing all memristor cells in the crossbar
        
    
    f_gp_cells: an array
    
        Array containing all faulty memristor cells grouped with the redundant cells used for overcoming the fault
        
     Returns:
    ----------
    xba_final: An n-d array or matrix
        
        An array of conductance values that is of the same size as the originally pretrained weight
        ready for testing using the prediction model of the neural network during the evaluation of
        it's performance
    
    '''    
    # Extract Defects
    SA0 = defects['SA_0']
    SA1 = defects['SA_1']
    
    tot_rows, tot_cols = xba.shape
    r_cols = tot_cols - w_c*2
    r_pos = int(r_cols/2)
    f_c = SA0 + SA1
    
    pos_xbr_cells = slice(0, w_c)
    neg_xbr_cells = slice(w_c, w_c*2)
    
    xba_final = xbar_output(xba, xba.shape[0], w_c)
    
    for cell in all_cells:
        for corrected in cell_map:
            if cell == corrected[0]:
                if cell[1] < w_c:
                    if corrected[1] < r_pos:
                        xba_final[cell] = xba_final[cell[0],cell[1]] + xba[cell[0],corrected[1]+w_c*2]- xba[cell[0],corrected[1]+w_c*2+r_pos]

                    if r_pos <= corrected[1] < r_cols:
                        # print('test 2')
                        xba_final[cell] = xba_final[cell[0],cell[1]] - xba[cell[0],corrected[1]+w_c*2] + xba[cell[0],corrected[1]+w_c*2-r_pos]

                if w_c <= cell[1] < 2*w_c:

                    if corrected[1] < r_pos:
                        # print('test 3')
                        xba_final[cell[0],cell[1]-w_c] = xba_final[cell[0],cell[1]-w_c] + xba[cell[0],corrected[1]+w_c*2] - xba[cell[0],corrected[1]+w_c*2+r_pos]

                    if r_pos <= corrected[1] < r_cols:
                        # print('test 4')
                        xba_final[cell[0],cell[1]-w_c] = xba_final[cell[0],cell[1]-w_c] - xba[cell[0],corrected[1]+w_c*2] + xba[cell[0],corrected[1]+w_c*2-r_pos]

    return xba_final[:w_r+1,:]


# In[ ]:


def xbar_output_prop(xba, w_r, w_c, group_cells, f_gp_cell, filter_size, endcell):
    
    '''
    Visualize the crossbar array with or without defects for every leyer
    
    Parameters:
    -----------
    
    w_r: integer value
        
        The number of rows in the original pretrained weight that we are mapping to the memristor
    
    w_c: integer value
        
        The number of columns in the original pretrained weight that we are mapping to the memristor
    
    defects:dictionary
    
        This contains the faulty crossbars, the cells with specific faults
        defects['xbar'] ------------ Faulty crossbar with cells affected by the introduced faults
        defects['SA_0'] ------------ Cells with values stuck at high resistance or low conductance value
        defects['SA_1'] ------------ Cells with values stuck at low resistance or high conductance value
        
    all_cells: an array
    
        Array containing all memristor cells in the crossbar
        
    
    f_gp_cells: an array
    
        Array containing all faulty memristor cells grouped with the redundant cells used for overcoming the fault
    
    pos: an array
    
        Array containing all memristor cells considered to be on the positive side of the crossbar
        
    sep: integer value
        
        The number indicating the number of rows and number of columns to be used in the case of a proposed
        architecture.
        
    fault_tool: can be existing_cells, red_columns, mixed, or proposed
        
        existing_cells: Uses the exixting cells to overcome the fault in the memristor crossbar
                        It does not require any additional hardware in terms of additional memrostor cells
                        
        red_columns:    Uses the redundant cells to overcome the faults in the memristor crossbar
                        It requires an additional number of rows and columns however, the size depends on the 
                        fault defect percentage
                    
        mixed:          Combines the behavior of the two approaches above to attain a balanced and more
                        robust approach taking advantage of each other.
        
        proposed:       Use the proposed approach to overcome the fault on the memristor crossbar
        
     Returns:
    ----------
    xba_final: An n-d array or matrix
        
        An array of conductance values that is of the same size as the originally pretrained weight
        ready for testing using the prediction model of the neural network during the evaluation of
        it's performance
    
    '''
    # Extract Cells from the group
    # pos = group_cells['positive']
    # neg_cell_ = group_cells['negative']
    # red_cell_ = group_cells['redundant']
    
    all_cells = group_cells['groups']
    # print(group_cells)
    
    # print("all_cells\n", all_cells)
    
    tot_rows, tot_cols = xba.shape
    
    xba_final = np.zeros((tot_rows, w_c))
    j = 0
    
    # print(xba.shape)
    # print(w_r, w_c)

    # print(xba_final.shape)
#     print("Cells Xbar Output\n",all_cells)
    for cells in all_cells: # Cells which are unaffected due to the faults
        pos_cell = cells[0]
        neg_cell = cells[1]
        
#         print('endcell up: ',endcell)
        
        if pos_cell[1] < tot_cols:
            j = int(xbar_cols(pos_cell[1], filter_size,w_c))
            # Use Positive Cell for Value feeds
            # print('j up: ', j)
            # print('neg cell: ', neg_cell)
            # print('pos cell: ', pos_cell)
            xba_final[pos_cell[0], j] = xba[pos_cell[0], pos_cell[1]] - xba[pos_cell[0], pos_cell[1]+1]
            # print('pos_cell - Out: ', pos_cell)

            # print('Step 1')
        
    if len(f_gp_cell) > 0:
        # print("f_gp_cell: ",f_gp_cell)
        for f_cells in f_gp_cell: # Cells which are unaffected due to the faults
            pos_cell = f_cells[0]
            neg_cell = f_cells[1]
            rec_pos = f_cells[2]
            rec_neg = f_cells[3]
            
            if pos_cell[0] <= tot_cols:
                j = int(xbar_cols(pos_cell[1], filter_size, w_c))

                # print('j down: ', j)
                # print('neg cell: ', neg_cell)
                # print('pos cell: ', pos_cell)

                xba_final[pos_cell[0], j] = xba_final[pos_cell[0], j] + xba[rec_pos[0], rec_pos[1]] - xba[rec_pos[0], rec_pos[1]+1]
            
    # return xba_final[:w_r,:]
    return xba_final[:w_r+1,:]


# Define A function to pick up corresponding Columns to be written to
def xbar_cols(column_number, filters, w_c):
    if filters == 4:
        if column_number%6==0:
            col = (column_number//6)*2
        else:
            col = (column_number//6)*2 + 1
    
    elif filters == 8:
        col=0
        if column_number>filters:
            col_out = (column_number//2)- 1
            if col_out < w_c:
                col = col_out
        else:
            col = (column_number%10)//2
    # print("column_number: ", column_number, col)
    return col


# ### Pass the Weights to the Model For Evaluation

# ### Model Imports

# In[ ]:


# Import Models
from keras import layers
from keras.models import model_from_json
from keras import optimizers
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import Sequential
from keras.layers import Input, Dropout, BatchNormalization, Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten,MaxPooling2D, Activation,concatenate, GlobalMaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from keras.regularizers import l1,l2
import datetime


# ### Define Data Source

# In[ ]:


def Lenet5():
    tools = []
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Convert the Targets to Categorical values
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Reshape Training and Test Datasets
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Build The Model
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(120, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    #####################################
    #     MEMRISTOR WEIGHTS MODEL       #
    #####################################
    weights = np.load(weight_file)
    model.set_weights(weights)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

    #####################################
    #           MODEL SUMMARY           #
    #####################################
#     print(model.summary())
    
    # Test Model Accuracy
    Accuracy = model.evaluate(X_test, y_test_cat)
    Accuracy_retrained = Accuracy
    
    print(f'Evaluation Tool: {eval_tool}')
    
    res_or = model.predict_classes(X_test[:6])
    
    tools.append(res_or)
    
    # Retrain Model Using the Wiehts as initializers    
    if re_train == True:
        model.fit(X_train, y_train_cat, batch_size=128, validation_split=0.2)
        
        x_weights = weight_checker['x_weights']
        final_weight = weight_checker['weight']
        weight_faultfree = weight_checker['weight_faultfree']
        
        retrained_weights = np.array(model.get_weights())
        model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
        Accuracy_retrained_no_fault = model.evaluate(X_test, y_test_cat)
        print('Retrained Without Fault Consideration: {}**'.format(Accuracy_retrained_no_fault[1]))
        
        for i in range(x_weights.shape[0]):
            my_shape = x_weights[i].shape
#             print(my_shape)
            if len(my_shape) == 1:
                for j in range(x_weights[i].shape[0]):
                    if x_weights[i][j] == False:
                        retrained_weights[i][j] = final_weight[i][j]
                        
            elif len(my_shape) == 2:
                for j in range(my_shape[0]):
                    for k in range(my_shape[1]):                        
                        if x_weights[i][j][k] == False:
                            retrained_weights[i][j][k] = final_weight[i][j][k]
            elif len(my_shape) == 4:
                for j in range(my_shape[0]):
                    for k in range(my_shape[1]): 
                        for m in range(my_shape[2]):
                            for n in range(my_shape[3]): 
                                if x_weights[i][j][k][m][n] == False:
                                    retrained_weights[i][j][k][m][n] = final_weight[i][j][k][m][n]
                            
        model.set_weights(retrained_weights)
        model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
        Accuracy_retrained = model.evaluate(X_test, y_test_cat)
        
        res_retrained = model.predict_classes(X_test[:6])
        
        tools.append(res_retrained)
    
    # Visualize Results
    
#     for res in tools:
#         plt.style.use('ggplot')
#         plt.figure(figsize=(10, 10))

#         for i in range(6):
#             plt.subplot(1, 6, i+1)
#             plt.imshow(X_test[i, :,:].reshape((28,28)), cmap='gray')
#             plt.gca().get_xaxis().set_ticks([])
#             plt.gca().get_yaxis().set_ticks([])
#             plt.xlabel('Pred: %d' % res[i])
#         plt.show()
    
    return Accuracy[1], Accuracy_retrained[1]    


# In[ ]:


def FashionMnist(converted_weight, weight_file, eval_tool, re_train=False):
    
    tools = []
    
    # Check the operating System
    import os
    # surpress Tensorflow warning information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Number of classes
    num_classes = 10

    # Batch size and number of epochs
    batch_size = 128
    # epochs = 24

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Load the training and test data from keras
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Reshape the data as required by the backend
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Scale the pixel intensities
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # Change the y values to categorical values
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

#     # Compile the model
#     model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#     # Train the model
#     hist = model.fit(x_train, y_train_cat,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(x_test, y_test_cat), 
#     verbose=2)

#     # Evaluate the model on test data
#     score = model.evaluate(x_test, y_test_cat, verbose=2)
#     print('Test Loss: ', score[0])
#     print('Test Accuracy: ', score[1])

#     # Visualize the training progress
#     epoch_list = list(range(1, len(hist.history['acc'])+1))
#     plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
#     plt.legend(('Training Accuracy', 'Validation Accuracy'))
#     plt.show()

#     np.save('fashion_mnist_weight.npy', model.get_weights())

    #####################################
    #     MEMRISTOR WEIGHTS MODEL       #
    #####################################
    weights = np.load(weight_file)
    model.set_weights(weights)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

    #####################################
    #           MODEL SUMMARY           #
    #####################################
    # print(model.summary())
    
    # Test Model Accuracy
    Accuracy = model.evaluate(x_test, y_test_cat)
    
    print(f'Evaluation Tool: {eval_tool}')
    
    res_or = model.predict_classes(x_test[:6])
    
    tools.append(res_or)
    # Retrain Model Using the Wiehts as initializers
    accuracy = Accuracy[1]
    retrained = Accuracy[1]
    
    if re_train == True:
        
        model.fit(x_train, y_train_cat, batch_size=batch_size, validation_split=0.2)
    
        Accuracy_retrained = model.evaluate(x_test, y_test_cat)
        
        res_retrained = model.predict_classes(x_test[:6])
        
        tools.append(res_retrained)
        
        retrained = Accuracy_retrained[1]
    
    # Visualize Results
    
#     for res in tools:
#         plt.style.use('ggplot')
#         plt.figure(figsize=(10, 10))

#         for i in range(6):
#             plt.subplot(1, 6, i+1)
#             plt.imshow(x_test[i, :,:].reshape((28,28)), cmap='gray')
#             plt.gca().get_xaxis().set_ticks([])
#             plt.gca().get_yaxis().set_ticks([])
#             plt.xlabel('Pred: %d' % res[i].eval())
#         plt.show()
    
    return accuracy, retrained  


# ### Alexnet Model
# Testing Using the Alexnet Model

# In[ ]:


def Alexnet(weight_checker, weight_file, eval_tool, re_train=False, seed_val=0):
    # Check the operating System
    import os
    from keras.optimizers  import Adam
    # surpress Tensorflow warning information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    np.random.seed(seed_val)
    retrained_weights =[]
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #  #z-score
    # mean = np.mean(x_train,axis=(0,1,2,3))
    # std = np.std(x_train,axis=(0,1,2,3))
    # x_training = (x_train-mean)/(std+1e-7)

    x_train = x_train / 255.0
    x_test = x_test / 255.0


    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    x_training_gray = np.dot(x_train[:,:,:,:3], [0.299, 0.587, 0.114])
    x_test_gray = np.dot(x_test[:,:,:,:3], [0.299, 0.587, 0.114])

    trainset = int(x_training_gray.shape[0]*0.8)

    x_train_gray = x_training_gray[:trainset]
    x_val_gray = x_training_gray[trainset:]

    x_train_gray = x_train_gray.reshape(-1,32,32,1)
    x_val_gray = x_val_gray.reshape(-1,32,32,1)
    x_test_gray = x_test_gray.reshape(-1,32,32,1)

    num_classes = 10
    y_training = np_utils.to_categorical(y_train,num_classes)
    y_train = y_training[:trainset]
    y_val = y_training[trainset:]
    y_test = np_utils.to_categorical(y_test,num_classes)
    # print(x_train.shape)
    # print(x_val.shape)
    # print(y_train.shape)
    # print(y_val.shape)

    # weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=x_train_gray.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

#     model.summary()

    # model.load_weights('new_weights_alexnet_cifar10.h5')

    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Load model weights
    
    weights = np.load(weight_file)
    model.set_weights(weights)
    scores = model.evaluate(x_test_gray, y_test, verbose=1)
    
    if re_train == True:
        # #data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2
            )
        datagen.fit(x_train_gray)

        # #training
        batch_size = 64
        
        model.fit_generator(datagen.flow(x_train_gray, y_train, batch_size=batch_size),                        steps_per_epoch=x_train_gray.shape[0] // batch_size,epochs=1,                        verbose=1,validation_data=(x_val_gray,y_val))
        
        x_weights = weight_checker['x_weights']
        final_weight = weight_checker['weight']
        weight_faultfree = weight_checker['weight_faultfree']
        
        retrained_weights = np.array(model.get_weights())
        # Check the accuracy before Resetting the faulty cells
        model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        Accuracy_retrained_no_fault = model.evaluate(x_test_gray, y_test)
        print('Retrained Without Fault Consideration: {}**'.format(Accuracy_retrained_no_fault[1]))
        
        # Reset the Faulty Cells
        for i in range(x_weights.shape[0]):
            my_shape = x_weights[i].shape
#             print(my_shape)
            if len(my_shape) == 1:
                for j in range(x_weights[i].shape[0]):
                    if x_weights[i][j] == False:
                        retrained_weights[i][j] = final_weight[i][j]
                        
            elif len(my_shape) == 2:
                for j in range(my_shape[0]):
                    for k in range(my_shape[1]):                        
                        if x_weights[i][j][k] == False:
                            retrained_weights[i][j][k] = final_weight[i][j][k]
                            
            elif len(my_shape) == 4:
                for j in range(my_shape[0]):
                    for k in range(my_shape[1]):
                        for m in range(my_shape[2]):
                            for n in range(my_shape[3]):
                                if x_weights[i][j][k][m][n] == False:
                                    retrained_weights[i][j][k][m][n] = final_weight[i][j][k][m][n]
        
        retrained_weights = np.array(retrained_weights)
        model.set_weights(retrained_weights)
        
        model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        scores_retrained = model.evaluate(x_test_gray, y_test, verbose=1)
    else:
        scores_retrained = scores
        
    
    print(f'Evaluation Tool: {eval_tool}')    
        
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
    #                     steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
    #                     verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
    # #save to disk
    # model_json = model.to_json()
    # with open('model.json', 'w') as json_file:
    #     json_file.write(model_json)
    # np.save('weights_alexnet_cifar10.npy', model.get_weights()) 

    return scores[1], scores_retrained[1]


# In[ ]:


from keras.models import model_from_json
global json_model 
json_model= 'inception_model.json'

def Googlenet(weight_checker, weight_file, eval_tool, re_train=False, seed_val=0):
    
    from keras.optimizers import Adam, SGD
    
    use_norm = True
    lrate = 0.001
    json_file = open(json_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train_gray = np.dot(x_train[:,:,:,:3], [0.299, 0.587, 0.114])
    x_test_gray = np.dot(x_test[:,:,:,:3], [0.299, 0.587, 0.114])

    x_train_gray = x_train_gray.reshape(-1,32,32,1)
    x_test_gray = x_test_gray.reshape(-1,32,32,1)
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    weights = np.load(weight_file)
    model.set_weights(weights)
    
    model.compile(loss='binary_crossentropy',
              optimizer=Adam(lrate),
              metrics=['accuracy'])
    
    # Test Model Accuracy
    Accuracy = model.evaluate(x_test, y_test_cat)
    Accuracy_retrained = Accuracy
    
    print(f'Evaluation Tool: {eval_tool}')
    
    # Retrain Model Using the Wiehts as initializers    
    if re_train == True:
        model.fit(x_train_gray, y_train_cat, batch_size=128, validation_split=0.2)
        
        x_weights = weight_checker['x_weights']
        final_weight = weight_checker['weight']
        weight_faultfree = weight_checker['weight_faultfree']
        
        retrained_weights = np.array(model.get_weights())
#         model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
#         Accuracy_retrained_no_fault = model.evaluate(X_test, y_test_cat)
#         print('Retrained Without Fault Consideration: {}**'.format(Accuracy_retrained_no_fault[1]))
        
        for i in range(x_weights.shape[0]):
            
            # print(x_weights.shape)
            
            my_shape = x_weights[i].shape
            if len(my_shape) == 1:
                for j in range(x_weights[i].shape[0]):
                    if x_weights[i][j] == False:
                        retrained_weights[i][j] = final_weight[i][j]
                        
            if len(my_shape) == 2:
                for j in range(my_shape[0]):
                    for k in range(my_shape[1]):                        
                        if x_weights[i][j][k] == False:
                            retrained_weights[i][j][k] = final_weight[i][j][k]
                            
        model.set_weights(retrained_weights)
        model.compile(loss='binary_crossentropy',optimizer=Adam(lrate), metrics=['accuracy'])
        Accuracy_retrained = model.evaluate(x_test, y_test_cat)
    
    return Accuracy[1], Accuracy_retrained[1] 


# ## Main Program Flow
# This is the where the program starts the sequential operation

# In[ ]:


def convert_weight(cond_vals, num_bits, default_weight, perc_def, filter_size=4, model_used='normal', fix_used='none', case="crossbar_with_split", seed_val=0):
    
    acc_or =[]
    ideal_acc = []
    w_b_split = []
    
    # Retrained
    rt_acc_or =[]
    
    # filename = 'weights_MNIST_CNN_npy.npy'

    or_weights = np.load(default_weight)
    
#     print(or_weights.shape)

    W = []
    W_init = []
    
    # Get the weight shape of the original weight to help in looping
    # Through the weights. This helps determine the trained layers

    tot_weights = or_weights.shape[0]
#     print(or_weights)

    # General Weight Reader For all Layers in any network
    for w in range(0, tot_weights,2):
#     for w in range(0, tot_weights):
        # Input Layer
        # The Even indices correspond to the 
        l_weight = or_weights[w]
#         print(l_weight.shape)
        l_bias = or_weights[w+1]

        # Shape of the Weight Section
        l =  l_weight.shape

        weight = l_weight.reshape(l)

        dims = len(l)
#         print('Dimensions: ', dims)
        weights = []
        if dims > 2:
            for i in range(l[0]):
                for j in range(l[1]):
                    if dims == 4:
                        for k in range(l[2]):
                            weights.append(weight[i,j,k])
                    else:
                        weights.append(weight[i,j])               
        elif dims == 1:
            weights = weight.reshape(1,l[0])
        else:
            weights = weight

        weights = np.array(weights)

        w_inp, w_dim = weights.shape
        # print("weights Shape: {}".format(weights.shape))
        
        biases = l_bias.reshape(1, l_bias.shape[0])
        # print("biases Shape: {}".format(biases.shape))

        # Get the maximum value of the weights abd bias
        # Combine the weight and bias by stacking
        # w_b_c = np.concatenate((weights, biases), axis=1)
        w_b_C_T = np.vstack((weights, biases))
        
#         w_b_C_T = np.hstack((weights, biases))
        wb_r, wb_c = w_b_C_T.shape

        # Define the wieghts and biases from the split
        # w_b_split, max_w_b = weight_split(weights, w_b_C_T)
        
        # if model_used == 'normal':
        
        # Weight Split for the normal crossbar
        w_b_split_norm, max_w_b_norm = weight_split(w_inp, w_dim, w_b_C_T, mode='normal')
        # Calculate the Conductance values of the weights and Biases
        wb_cond_norm = weight_bias_cond(w_b_split_norm, cond_vals, max_w_b_norm)
        
        # elif model_used == 'proposed':
        
        # Weight Split for the proposed crossbar
        w_b_split_prop, max_w_b_prop = weight_split(w_inp, w_dim, w_b_C_T, mode='proposed')
    
        # Calculate the Conductance values of the weights and Biases
        wb_cond_prop = weight_bias_cond(w_b_split_prop, cond_vals, max_w_b_prop)

        # Introduce redundancy
        if model_used == 'normal':
            my_weights, endval, rem = add_red_array(wb_cond_prop, perc_def, cond_vals, 'proposed', filter_size)
            normal_dist, endval2, rem2 = add_red_array(wb_cond_norm, perc_def, cond_vals, model_used, filter_size)
            
        elif model_used == 'proposed':
            my_weights, endval, rem = add_red_array(wb_cond_prop, perc_def, cond_vals, model_used, filter_size)        
            normal_dist, endval2, rem2 = add_red_array(wb_cond_norm, perc_def, cond_vals, 'normal', filter_size)
        
        # print("Weight Shape after Redundancy: ", my_weights.shape)

        #################################################################################################
        #                              PROPOSED WEIGHT SPLIT PATTERN                                    #
        #################################################################################################
        # Finnd the shape of normal distributed Weight
        # print(normal_dist)
        # print(normal_dist.shape)
        
        # Faults introduced to the crossbar
        defected_weight = add_defects(normal_dist,my_weights, perc_def, cond_vals, model_used, filter_size, case, seed_val)
        
        # print("Weights After Applying Defects\n",defected_weight)
        # print(defected_weight)
        new_xbar = defected_weight["xbar"]
        all_xbr_cells = defected_weight["all_cells"]
        f_cell = defected_weight["f_cells"]
        sa_0 = defected_weight["SA_0"]
        sa_1 = defected_weight["SA_1"]
        
        # print(len(f_cell), ' Faulty Cells')
        # print(sa_0, ' SA_0')
        # print(sa_1, ' SA_1')
        # Cell grouping for Proposed approach
        
        grouped_cells = groupings(all_xbr_cells, endval, rem, filter_size)
        
        # print("defected_weight\n", defected_weight)
        
        # Visualization
        p_cells = ()
        n_cells = ()

        # TO DO!
        # FAULT handling
        # Regarding the affected Cells

        # Considering the bit levels
        global bits, L_weights
        
        # Obtain the Weights when Cells have not been recovered
        bits_init = bit_level_precision(cond_vals, normal_dist, num_bits, max_w_b_norm)
        # print("bits Prop\n{}".format(type(bits[0])))
        L_weights_init = xbar_output(bits_init[0],w_inp, w_dim)
        
        # Consider Each approach
        if fix_used == 'none':            
            bits = bit_level_precision(cond_vals, new_xbar, num_bits, max_w_b_norm)
            # print("bits Prop\n{}".format(type(bits[0])))
            L_weights = xbar_output(bits[0],w_inp, w_dim)
            # print("L_weights Normal\n{}".format(L_weights))
            
        elif fix_used == 'existing_cells':
            fault_tol = corr_rows_cols(my_weights, defected_weight, w_dim, cond_vals)
            bits = bit_level_precision(cond_vals, fault_tol, num_bits, max_w_b_norm)
            L_weights = xbar_output(bits[0], w_inp, w_dim)
        
        elif fix_used == 'redundant_cols':            
            tol_red_col, red_col_map = red_rows_cols(my_weights[0], defected_weight, w_dim, cond_vals, seed_val)
            bits = bit_level_precision(cond_vals, tol_red_col, num_bits, max_w_b_norm)
            L_weights = xbar_output_red_col(bits[0],  w_inp, w_dim, red_col_map, defected_weight, all_xbr_cells)

        # Fault Handled Weights Using Both Existing Cells and Redundant Rows and Columns
        elif fix_used == 'combined':
            combined_approach = combined(my_weights, defected_weight, w_dim, cond_vals, seed_val)
            tol_mixed = combined_approach["f_xbr"]
            # faulty_both = combined_approach["both_faulty"]
            cell_mapping = combined_approach["mapping"]
            bits = bit_level_precision(cond_vals, tol_mixed, num_bits, max_w_b_norm)
            L_weights = xbar_output_combined(bits,  w_inp, w_dim, cell_mapping, defected_weight, all_xbr_cells)
            
        elif fix_used == 'proposed': 
            endpoint = endval+rem 
            # print("Original\n{}\nCalculated\n{}\nCeel Groups\n{}\nConductance\n{}\n".format(my_weights, defected_weight, grouped_cells, cond_vals))
            # Make the Cell Groupings
            # print("my_weights Shape: ", my_weights.shape)
            # print("Defected Cells: ", f_cell)
            # print("grouped_cells: ", grouped_cells)
            
            # print("Grouped For Recovery Cells: ", f_grp)
            cells_considered = recoverable(f_cell, filter_size, endval,rem)
            tol_prop, f_grp_mapping = proposed_approach(my_weights, defected_weight, cells_considered, cond_vals, endpoint)
            
            bits = bit_level_precision(cond_vals, tol_prop, num_bits, max_w_b_prop)
            # print("bits Prop\n{}".format(type(bits[0])))
            L_weights = xbar_output_prop(bits[0], w_inp, w_dim, grouped_cells, f_grp_mapping, filter_size, endpoint)
            # print("L_weights Proposed\n{}".format(L_weights))
            # print("defected_weight\n{}".format(defected_weight['xbar'].shape))
        
        w = L_weights[:w_inp, :].reshape((l_weight.shape))
        b = L_weights[w_inp, :].reshape(l_bias.shape)
        # print(w.shape)
        # print(b.shape)
        
        w_init = L_weights_init[:w_inp, :].reshape((l_weight.shape))
        b_init = L_weights_init[w_inp, :].reshape(l_bias.shape)
        
        # Weight for cells not recovered
        W_init.append(w_init)
        W_init.append(b_init)
        
        # Weight of Cells if at all recovered
        W.append(w)
        W.append(b)
    
    W = np.array(W)
    W_init = np.array(W_init)
    
    converted={
        'weight':W,
        'weight_init':W_init,
        'fix_tool':fix_used
    }
    
    return converted
# Group the cells with their recovery redundant columns based on the split
def recoverable(faulty_cells, split, endval, rem):
    recovery_cells = []
    if split == 4:
        for gp_cell in faulty_cells:
            # Check Collumns of the cells to help determine whether they are positive or negative
            # if gp_cell[1]>=endval:
                # print('gp_cell: - ', gp_cell)
                        
            # else:
            if gp_cell[1]%6 == 0:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+split), (gp_cell[0], gp_cell[1]+split+1)])

            elif gp_cell[1]%6 == 1:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1),gp_cell,(gp_cell[0], gp_cell[1]+split-1), (gp_cell[0], gp_cell[1]+split)])

            elif gp_cell[1]%6 == 2:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+2), (gp_cell[0], gp_cell[1]+3)])

            elif gp_cell[1]%6 == 3:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+2)])

            # If the positive recovery cell is faulty, overcome using the existing cells close to it
            # To Avoid conflict for cells to overcome the fault, different cells are used to overcome the faults in the redundant columns

            elif gp_cell[1]%6 == 4:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-2), (gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+1)])

            else:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-3), (gp_cell[0], gp_cell[1]-2), (gp_cell[0], gp_cell[1]-1),(gp_cell[0], gp_cell[1])])
    
    elif split == 8:
        for gp_cell in faulty_cells:
            
            # if gp_cell[1]>=endval:
                # print('gp_cell: - ', gp_cell)
                
                
            # else:
            if gp_cell[1]%10 == 0:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+split), (gp_cell[0], gp_cell[1]+split+1)])

            elif gp_cell[1]%10 == 1:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1),gp_cell,(gp_cell[0], gp_cell[1]+split-1), (gp_cell[0], gp_cell[1]+split)])

            elif gp_cell[1]%10 == 2:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+split-2), (gp_cell[0], gp_cell[1]+split-1)])

            elif gp_cell[1]%10 == 3:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+split-3), (gp_cell[0], gp_cell[1]+split-2)])

            elif gp_cell[1]%10 == 4:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+split-4), (gp_cell[0], gp_cell[1]+split-3)])

            elif gp_cell[1]%10 == 5:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+split-5), (gp_cell[0], gp_cell[1]+split-4)])

            elif gp_cell[1]%10 == 6:
                recovery_cells.append([gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+split-6), (gp_cell[0], gp_cell[1]+split-5)])

            elif gp_cell[1]%10 == 7:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+1), (gp_cell[0], gp_cell[1]+2)])

            # Use the last set of positive negative cells to help overcome the redundant column faults

            elif gp_cell[1]%10 == 8:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-2), (gp_cell[0], gp_cell[1]-1), gp_cell, (gp_cell[0], gp_cell[1]+1)])

            else:
                recovery_cells.append([(gp_cell[0], gp_cell[1]-3), (gp_cell[0], gp_cell[1]-2), (gp_cell[0], gp_cell[1]-1),gp_cell])
                    
    print(recovery_cells)
    return recovery_cells


# In[ ]:


def accuracy_check(converted_weight, perc_def, model, new_weight_filename='memristorWeights/Memristor_weights.npy', re_train=False):
    tool = converted_weight['fix_tool']
    weight_init = converted_weight['weight_init']
    final_weight = converted_weight['weight']
    
    # print('Final Weights:\n',final_weight.shape)
    weight_check = []
    weight_checker = {}
    
    for i in range(weight_init.shape[0]):
        weight_check.append(weight_init[i] == final_weight[i])
        # print(weight_init[i].shape)
    weight_checker['weight_faultfree'] = weight_init # Cross referenced Weights    
    weight_checker['x_weights'] = np.array(weight_check) # Cross referenced Weights
    weight_checker['weight'] = final_weight              # Final Weights after recovery
    
    # print("Final: {}, Shape: {}".format(final_weight, final_weight.shape))
    # print("Initial: {}, Shape: {}".format(weight_init, weight_init.shape))
    
    # comp = np.all([weight_init, final_weight])
    # print(comp)
    
    with open (new_weight_filename, 'r+'):
        np.save(new_weight_filename, np.array(final_weight))
    
    return weight_checker, new_weight_filename    


# In[ ]:


def main():
    
    Ron, Roff = (int(input('R-on: ')), int(input('R-off: ')))

    # Get the Memristance values of the meristor
    cond_vals = conductanceValues(Ron, Roff)
    
    # Split of Conductance Values
    MINcond = cond_vals['c_min']
    MINcond = cond_vals['c_max']
    condRANGE = cond_vals['c_range']
    
    # Considering the bit levels
    num_bits = int(input('Enter the number of bits: '))
    
    default_weights = 'Lenet_weight.npy'
#     default_weights = 'fashion_mnist_weight.npy'
#     default_weights = 'alexnet_weights.npy'
#     default_weights = 'inception_cifar10.npy'
    
    # model = eval(input('Enter model (lenet5, FashionMnist, ..): '))
    
    model = eval(input('Enter model (lenet5, Alexnet, Googlenet): '))
    converted_weights = np.array([])
    software_based_accuracy = model(converted_weights, default_weights, 'Ideal')

    print(f'Ideal/Software Based Accuracy: {software_based_accuracy[1]}\n{"+"*40}')
    
    # Define Approach to Use
    approach = input("Enter Approach (normal, proposed): ").lower()
    
    # Define Fix Model to Use
    
    filter_size=4
    if approach != 'proposed':
        fix_model = input("Enter Fix Model (None, existing_cells, redundant_cols, combined): ").lower()
    
    else:
        fix_model = 'proposed'
        filter_size = int(input("Enter the column size for the split: "))
    test_case = input("Enter the test-case (crossbar_with_split, crossbar_no_split): ")
    
    # Introduce Fault Defects
    # percentage_defect = int(input("Enter Fault Percentage: "))
    with open ('accuracy_tracker.txt', 'a+') as f:
        f.write("{} - {} - {} bits\n".format(fix_model, test_case,num_bits))
        f.write("perc_def\tAccuracyies\tAverage\n")
        f.write("_"*40+"\n")
    
    # seed_vals = [123, 111, 155, 555, 100,789, 329, 500, 907, 644]
        
    seed_vals = [0, 123, 579, 111, 155, 555, 222, 100, 888, 456, 789, 329, 500, 579, 999, 234, 907, 644, 811, 440]
    
    for percentage_defect in range(3, 4):
        
        accs = []
        acc_retrain= []
        re_train = False
        
        for my_seed in seed_vals:
            converted_weights = convert_weight(cond_vals, num_bits, default_weights, percentage_defect, filter_size, approach, fix_model, test_case, seed_val=my_seed)
            # print(converted_weights)
            acc = accuracy_check(converted_weights, percentage_defect, model, new_weight_filename='memristorWeights/Memristor_weights.npy', re_train = re_train)
            accs.append(acc[0])
            acc_retrain.append(acc[1])

        with open ('accuracy_tracker.txt', 'a+') as f:
            f.write("{}\t{}\t{}\n".format(percentage_defect,accs, np.array(accs).mean()))
            f.write("{}\t{}\t{}\n".format(percentage_defect,acc_retrain, np.array(acc_retrain).mean()))

#     sendMail('accuracy_tracker.txt')


# In[ ]:


# import smtplib
# import email
# from email import encoders

# def sendMail(file_attachment):
#     fromaddr = "elimumicahel9@gmail.com"
#     toaddr = "elimumichael@hnu.edu.cn"

#     msg = email.mime.multipart.MIMEMultipart()

#     msg['From'] = fromaddr
#     msg['To'] = toaddr
#     msg['Subject'] = "Experiment Results"

#     body = """Dear Michael,
    
#     I have completed the run. Attached are your results
    
#     """

#     msg.attach(email.mime.text.MIMEText(body, 'plain'))

#     filename = file_attachment
#     attachment = open("/filename", "rb")

#     part = email.mime.base.MIMEBase('application', 'octet-stream')
#     part.set_payload((attachment).read())
#     encoders.encode_base64(part)
#     part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

#     msg.attach(part)

#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(fromaddr, "Sumi@Nel0330")
#     text = msg.as_string()
#     server.sendmail(fromaddr, toaddr, text)
#     server.quit()


# In[ ]:


if __name__ == '__main__':
   main()


# In[ ]:


# # Predicting the inputs
# import requests
# from PIL import Image

# url = 'http://postfiles7.naver.net/20160616_86/ejjun92_1466078439176U1Bsu_PNG/sample_digit.png?type=w580'
# response = requests.get(url, stream=True)
# img = Image.open(response.raw)
# plt.imshow(img, cmap=plt.get_cmap('gray'))


# In[ ]:


# # Resize the image and ensure that it is a binary image
# import cv2
# img = np.asarray(img)

# print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img, (28, 28))
# img = cv2.bitwise_not(img)
# plt.imshow(img, cmap=plt.get_cmap('gray'))


# In[ ]:


# model = lenet5()
# img = img/255
# print(img.shape)
# img = img.reshape(1, 28, 28, 1)
# prediction = model.predict_classes(img)

