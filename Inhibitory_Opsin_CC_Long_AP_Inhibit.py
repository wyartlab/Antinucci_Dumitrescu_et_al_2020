#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:31:06 2019
@author: adna.dumitrescu

Script opens abf file with current clamp data aquired with a protocol in which a current injection (200-800ms) which induced tonic firing was paired with a long (50-200ms) LED stimulation in the middle. 
 
How the script works: 
1. Asks user to provide a file path for the abf file to be analysed
2. Asks user for meta-data that cannot be automatically extracted from abf file (such as stimulation regime and cell type etc)
3. Finds all points where the current pulse is applied and uses these time-bins to extract data from the trace in which voltage values are recorded.
3. Finds all points where the LED is ON and uses these time-bins to extract data from the trace in which voltage values are recorded. 
4. Puts all extracted data in a single .csv file named after the trace
5. Adds data as a new row to a master .csv file for the whole experiment 
6. Outputs in line graphs showing cell response only during concomitent current injection and LED stimulation time period.

Metadata collected: file name, date, experimenter, protocol type, opsin type, LED wavelenght, power and stimulation time.
Data that is calculated by script: 
Baseline voltage level: mean of points across first 100ms of the trace
Value of current pulse in pA
Time of current pulse duration in ms 
Time of LED stimulation in ms 
Number of spikes in control (current pulse only) trace split in 3 time bins equal to: time in pre LED stim, during LED stim, and after LED stim. 
Number of spikes in experimental trace (current and LED stimulation) split into the same time bins: pre, mid and post LED stimulation
Also calculates the time to 1st spike at the end of the LED stimulation.

Data was normalised offline to a 50ms time bin. 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyabf
import pandas as pd
from scipy.signal import find_peaks
import os
wdir=os.getcwd() 


#### open file 
file_path = input('Please give me the complete file path of the trace you want to analyse below:\n')
abf = pyabf.ABF(file_path)

### extract main data
data = abf.data
voltage_trace = data[0,:] # extracts primary channel recording in CC which is voltage measurement 
voltage_data_baseline = np.mean(voltage_trace [0:1999])### calculate baseline pA response by averaging the first 100ms
current_trace = data[1,:] # extracts 2ndary channel recording in CC which is current measurement 
current_data_baseline = np.mean(current_trace [0:1999])### calculate baseline pA response by averaging the first 100ms
current_trace_baseline_substracted = current_trace - current_data_baseline

LED_trace = data[-1,:] # extracts 4th channel recording in CC which is LED analog signal (channel 3 = LED TTL, not needed)
date_time = abf.abfDateTime
protocol = abf.protocol
time = abf.sweepX
resting_potential = np.mean(voltage_trace [0:20000]) # takes mean of all values aquired in the 1st second which is used as baseline membrane resting potential 
sampling_rate = abf.dataPointsPerMs
file_name = abf.abfID ### extract filename 
protocol = abf.protocol


### select experimenter 
user_input = int(input('Which rig was used for this recording:   Rig 1 = 1   Rig 2 = 2\n'))

experimenter_dict = { 
       1 : 'Rig 1' , 
       2 : 'Rig 2' 
        } #dictionary for all opsin types numbered according to potential user input 

experimenter = experimenter_dict.get(user_input) #get the dictionary value based on key entered by user   

if experimenter != None:
    print ('Trace recorded at ' +str(experimenter)) #print this is choice selected is in the dictionary 
else:
    raise ValueError ('Wrong number entered for Rig used, please run script again. No data was saved') #print this if choice selected in not in the opsin dictionary 

### select opsin type 
user_input = int(input('What cell type is this? Type the corresponding number: \nWT = 0\nEXCITATORY OPSINS: ChR2(1)      CoChR(2)     Chrimson(3)         ReaChR(4)       Chronos(5)      Cheriff(6)     \nINHIBITORY OPSINS: GtACR1(7)       GtACR2(8)       NpHR(9)         Arch(10)\n\n'))

cell_type_dict = { 
       0 : 'WT' , 
       1 : 'ChR2' , 
       2 : 'CoChR',
       3 : 'Chrimson' , 
       4 : 'ReaChR' , 
       5 : 'Chronos' , 
       6 : 'Cheriff' , 
       7 : 'GtACR1' , 
       8 : 'GtACR2' , 
       9 : 'NpHR3.0' , 
       10 : 'Arch3.0' 
        } #dictionary for all opsin types numbered according to potential user input 

cell_type_selected = cell_type_dict.get(user_input) #get the dictionary value based on key entered by user   

if cell_type_selected != None:
    print ('Trace recorded from ' +str(cell_type_selected) + ' positive cell' ) #print this is choice selected is in the dictionary 
else:
    raise ValueError ('Wrong number entered for cell type, please run script again. No data was saved') #print this if choice selected in not in the opsin dictionary 


##### establish LED stimulation: LED wavelenght and power 

##extract wavelenght used 
LED_wavelenght_user = int(input('What LED wavelenght did you use for this trace? Chose from the following options: \n(1)  475nm (LED3)    \n(2)  520nm (LED4)  \n(3)  543nm (TRITC)\n(4)  575nm (LED5)\n(5)  630nm (cy5)\n'))

LED_wavelength_dict = { 
       1 : '475' , 
       2 : '520' ,
       3 : '543',
       4 : '575',
       5 : '630',
        } #dictionary for all opsin types numbered according to potential user input 

LED_wavelength = LED_wavelength_dict.get(LED_wavelenght_user) #get the dictionary value based on key entered by user   

if LED_wavelength != None:
    print ('Trace recorded with ' +str(LED_wavelength) + 'nm light stimulation' ) #print this is choice selected is in the dictionary 
else:
    raise ValueError ('Wrong number entered for LED wavelength, please run script again. No data was saved') #print this if choice selected in not in the opsin dictionary 


## extract power range 
LED_stim_type_user = int(input('What LED stim did you do? Chose from the following options: \n(1)  475nm 2% max irradiance\n(2)  475nm 20% max irradiance\n(3)  475nm 50% max irradiance\n(4)  475nm 100% max irradiance\n\n(5)  520nm 50% max irradiance\n(6)  520nm 100% max irradiance\n\n(7)  543nm 50% max irradiance\n(8)  543nm 100% max irradiance\n\n(9)  575nm 50% max irradiance\n(10)  575nm 100% max irradiance\n\n(11)  630nm 50% max irradiance\n(12)  630nm 100% max irradiance\n\n'))

LED_power_setup_dict = { 
       1 : 'LED_475_2%' , 
       2 : 'LED_475_20%' , 
       3 : 'LED_475_50%' , 
       4 : 'LED_475_100%' , 
       5 : 'LED_520_50%' , 
       6 : 'LED_520_100%' ,
       7 : 'LED_543_50%',
       8 : 'LED_543_100%',
       9 : 'LED_575_50%' , 
       10 : 'LED_575_100%',  
       11 : 'LED_630_50%' , 
       12 : 'LED_630_100%' , 
        } #dictionary for all opsin types numbered according to potential user input 

LED_stim_type = LED_power_setup_dict.get(LED_stim_type_user) #get the dictionary value based on key entered by user   

if LED_stim_type != None:
    print ('Trace ' +str(file_name) + ' was recorded with ' +str(LED_stim_type) + ' light power' ) #print this is choice selected is in the dictionary 
else:
    raise ValueError ('Wrong number entered for LED stimulation type, please run script again. No data was saved') #print this if choice selected in not in the opsin dictionary 

"""
Add any functions used in the script below here
"""
## function to find consecutive elements
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
      
'''
Part 1 - find when a current pulse has been applied
'''

#### find if a current pulse was applied: 

### find indices where current pulse is applied
current_injection_idx = np.asarray(np.where(current_trace_baseline_substracted > 20)) # index of values in current trace where the current injected in more then 10pA
current_injection_idx_cons = consecutive(current_injection_idx[0]) #find consecutive indexes where current over 10pa is applied and split into separate arrays --> each array would be 1 pulse 
current_injection_idx_cons_df = pd.DataFrame(current_injection_idx_cons) #transform data into dataframe for easier processing below 

current_pulse_length = current_injection_idx_cons_df.count(axis = 'columns') #count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
current_pulse_length_final = round(current_pulse_length / sampling_rate) # transform number of elements into actual ms rounded 
current_pulse_length_final = current_pulse_length_final.drop_duplicates ()

current_data_I_pulse = [current_trace_baseline_substracted [i] for i in current_injection_idx_cons ] # use index extracted for each individual pulse to extract current values 
current_data_df = pd.DataFrame(current_data_I_pulse) #transform current_data into a data frame
current_max_I_inj = list(map(max, current_data_I_pulse))
current_max_I_inj = pd.Series(current_max_I_inj)
current_max_I_inj_final = current_max_I_inj[0]
    
### find pulse value 
I_pulse_max_idx = np.argmax(current_data_I_pulse[0])
I_pulse_max = current_data_I_pulse[0] > (10) ## get boolean array with TRUE values where there is a current response smaller then -10pA
I_pulse_max_df = pd.DataFrame(np.where(I_pulse_max == True))
I_pulse_start_idx = I_pulse_max_df[0][0]
stim_type_I_inj = 'I_pulse'

### use current pulse indices to extract corresponding voltage response  
voltage_data_I_injection = [voltage_trace [i] for i in current_injection_idx_cons]
voltage_data_I_injection_df = pd.DataFrame(voltage_data_I_injection)
spike_I_stim = find_peaks(voltage_data_I_injection[0], height=0) ##check if there are any spikes present
            
          
'''
Part 2: find when LED pulse have been applied
'''
###### find index values where LED is ON use this to extract all other info from current trace
LED_idx = (np.where(LED_trace > 0.18)) # index of values in LED trace where V values are over 0
LED_array = np.asarray(LED_idx) #transform into an array for next step

LED_idx_cons = consecutive(LED_array[0]) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim
LED_idx_cons_df = pd.DataFrame(LED_idx_cons) #transform data into dataframe for easier processing below 

### determine lenght of LED Stimq1
LED_time =pd.Series (LED_idx_cons_df.count(axis = 'columns'), name ='LED_time_ON' )#count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
LED_time = round(LED_time / sampling_rate) # transform number of elements into actual ms rounded 


###### use selected LED indices to extract current and voltage data
voltage_data_LED = [voltage_trace [i] for i in LED_idx_cons]
current_data_LED =  [current_trace_baseline_substracted [i] for i in LED_idx_cons] # use index extracted for each individual pulse to extract voltage values 
voltage_data_LED_df = pd.DataFrame (voltage_data_LED)

LED_data = [LED_trace [i] for i in LED_idx_cons] 
LED_data_df = pd.DataFrame(LED_data)

####### determine max current and voltage  response value 
current_max_LED = list(map(max, current_data_LED)) #get list of all current max values per pulse (need to be min since this is VC)
voltage_max_LED = list(map(max, voltage_data_LED)) #get list of all voltage max values per pulse
current_max_LED = list(map(abs, current_max_LED))
voltage_baseline_LED = voltage_data_baseline
voltage_deflection_LED = voltage_max_LED - voltage_baseline_LED
stim_type_LED = 'LED_pulse'

##### counting spikes 
if experimenter == 'Rig 1':
    
    
    mask = np.isin(current_injection_idx,LED_array, invert=True) ## create mask of where only I pulse idx are 

    I_pulse_only = current_injection_idx [mask] ## create array using mask of where only I pulses are 
    I_only_idx_cons = consecutive(I_pulse_only) ## separate them in separate arrays by pulse

    I_only_pre = len(I_only_idx_cons[1])
    I_only_mid = len (I_only_idx_cons[0]) - I_only_pre 
    I_only_post = I_only_pre + I_only_mid

    I_only_pre_mid_post = np.split(I_only_idx_cons[0], [I_only_pre, I_only_mid, I_only_post ])


    voltage_data_I_pulse_only = [voltage_trace [i] for i in I_only_pre_mid_post]

    spike_per_I_only = []
    for pulse in voltage_data_I_pulse_only :
        spike_per_I = find_peaks(pulse, height = -20)
        spike_per_I_only.append(spike_per_I[0])

    control_I_pre_spike = len(spike_per_I_only [0])
    control_I_mid_spike = len(spike_per_I_only [1])
    control_I_post_spike = len(spike_per_I_only [2])

else:
       
    mask = np.isin(current_injection_idx,LED_array, invert=True) ## create mask of where only I pulse idx are 

    I_pulse_only_others = current_injection_idx [mask] ## create array using mask of where only I pulses are 
    I_only_idx_cons_others = consecutive(I_pulse_only_others) ## separate them in separate arrays by pulse

    
    I_only_pre = len(I_only_idx_cons_others[1])
    I_only_mid = len (I_only_idx_cons_others[0]) - I_only_pre 
    I_only_post = I_only_pre + I_only_mid 
    
    I_only_idx_cons_others_long = I_only_idx_cons_others[0::3]
    I_only_pre_mid_post = np.split(I_only_idx_cons_others_long[0], [I_only_pre, I_only_mid, I_only_post ])

    I_only_pre_mid_post = []
    for array in I_only_idx_cons_others_long:
        I_only_pre_mid_post1 = np.split(array, [I_only_pre, I_only_mid, I_only_post ])
        I_only_pre_mid_post.append(I_only_pre_mid_post1)
    
    I_only_pre_pulse = []
    for lst in  I_only_pre_mid_post:
        pre_pulse = lst[0]
        I_only_pre_pulse.append(pre_pulse)

    I_only_mid_pulse = []
    for lst in  I_only_pre_mid_post:
        mid_pulse = lst[1]
        I_only_mid_pulse.append(mid_pulse)
       
    I_only_post_pulse = []
    for lst in  I_only_pre_mid_post:
        post_pulse = lst[2]
        I_only_post_pulse.append(post_pulse)

    voltage_data_I_pulse_pre = [voltage_trace [i] for i in I_only_pre_pulse]
    voltage_data_I_pulse_mid = [voltage_trace [i] for i in I_only_mid_pulse]
    voltage_data_I_pulse_post = [voltage_trace [i] for i in I_only_post_pulse]
   

    spike_per_I_pulse_pre = []
    for pulse in voltage_data_I_pulse_pre :
        spike_per_I = find_peaks(pulse, height = -20)
        spike_per_I_pulse_pre.append(spike_per_I[0])

    spike_per_I_pulse_mid = []
    for pulse in voltage_data_I_pulse_mid :
        spike_per_I = find_peaks(pulse, height = -20)
        spike_per_I_pulse_mid.append(spike_per_I[0])

    spike_per_I_pulse_post = []
    for pulse in voltage_data_I_pulse_post :
        spike_per_I = find_peaks(pulse, height = -20)
        spike_per_I_pulse_post.append(spike_per_I[0])


    control_I_pre_spike = []
    for lst in spike_per_I_pulse_pre:
        spike_count = len(lst)
        control_I_pre_spike.append(spike_count)

    control_I_mid_spike = []
    for lst in spike_per_I_pulse_mid:
        spike_count = len(lst)
        control_I_mid_spike.append(spike_count)
        

    control_I_post_spike = []
    for lst in spike_per_I_pulse_post:
        spike_count = len(lst)
        control_I_post_spike.append(spike_count)
        
        
#####
voltage_data_LED_array = np.vstack(voltage_data_LED)

#### spikes per I pulses paired with coincident opsin activation 
I_plus_LED_index = np.intersect1d(current_injection_idx, LED_array) ## put together indices where both current and light are on 
I_plus_LED_index_cons  = consecutive(I_plus_LED_index) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim

spike_per_I_and_LED_pulse = []
for pulse in voltage_data_LED_array:
    spike_per_LED = find_peaks(pulse, height = -20)
    spike_per_I_and_LED_pulse.append(spike_per_LED[0])

LED_I_spike = []

for lst in spike_per_I_and_LED_pulse:
    spike_per_pulse = len(lst)
    LED_I_spike.append(spike_per_pulse)

## pre post LED spikes
if experimenter == 'Rig 1':
    pre_post_LED_idx= I_only_idx_cons
    del pre_post_LED_idx [0]

    voltage_data_pre_post_LED = [voltage_trace [i] for i in pre_post_LED_idx]

    pre_post_LED_I_spike = []

    for pulse in voltage_data_pre_post_LED:
        spike_per_LED = find_peaks(pulse, height = -20)
        pre_post_LED_I_spike.append(spike_per_LED[0])

    pre_post_LED_I_spike_no = []

    for lst in pre_post_LED_I_spike:
        spike_per_LED = len(lst)
        pre_post_LED_I_spike_no.append(spike_per_LED)

    
    pre_LED_I_spike_no = pre_post_LED_I_spike_no[::2]
    post_LED_I_spike_no = pre_post_LED_I_spike_no[1::2]
   
    post_LED_spike = pre_post_LED_I_spike[1::2]
    
    first_spike_timing = []
    for array in post_LED_spike:
        if array.size: 
            idx_first_spike = array[0]
            spike_timing = idx_first_spike / sampling_rate
            first_spike_timing.append(spike_timing)
        else:
            spike_timing = np.nan
            first_spike_timing.append(spike_timing)


else:
    pre_post_LED_idx = I_only_idx_cons_others
    del pre_post_LED_idx [0::3]
    
    voltage_data_pre_post_LED = [voltage_trace [i] for i in pre_post_LED_idx]

    pre_LED_spike_data = voltage_data_pre_post_LED [0::2]
    post_LED_spike_data = voltage_data_pre_post_LED [1::2]
    
    pre_LED_spike = []
    for array in pre_LED_spike_data:
        spike_per_pulse = find_peaks(array, height = -20)
        pre_LED_spike.append(spike_per_pulse[0])
        
    post_LED_spike = []
    for array in post_LED_spike_data:
        spike_per_LED = find_peaks(array, height = -20)
        post_LED_spike.append(spike_per_LED[0])
        
    pre_LED_I_spike_no = []
    for array in pre_LED_spike:
        spike_count = len(array)
        pre_LED_I_spike_no.append(spike_count)
 
    post_LED_I_spike_no = []
    for array in post_LED_spike:
        spike_count = len(array)
        post_LED_I_spike_no.append(spike_count)
       
    first_spike_timing = []
    for array in post_LED_spike:
        if array.size: 
            idx_first_spike = array[0]
            spike_timing = idx_first_spike / sampling_rate
            first_spike_timing.append(spike_timing)
        else:
            spike_timing = np.nan
            first_spike_timing.append(spike_timing)

####### determine power in mW/mm2 of max LED analog pulse V value
LED_max_V = list(map(max, LED_data)) ## find max LED pulse points 
threshold = len([1 for i in LED_max_V if i > 5])## count if there are any values over 5V in LED_max_V, means that scale was set up wrong and pulse values in V need to be introduced manually by user 

if threshold == 0: ## pulse values are below 5V
    LED_max_V_round = [round (i,1) for i in LED_max_V] ## round number to 2 decimals
    LED_max_V_round_int = [str(i) for i in  LED_max_V_round] ## transform number to int otherwise can't index into LED_stim_power_table
    LED_index_value = LED_max_V_round_int
else:
    print('Wrong scale detected for LED analog input. Please add LED steps in Volts below, and na for no pulse applied:\n')
    LED_max_V_user = [input('pulse_1:  \n'), input('pulse2:  \n'), input('pulse3:  \n'), input('pulse4:  \n'), input('pulse5:  \n'), input('pulse6:  \n'), input('pulse7:  \n')]
    LED_index_value = LED_max_V_user
    while 'na' in LED_max_V_user: LED_max_V_user.remove('na')  ## remove any na values since we don't need them 

##### open up excel sheet with all LED power values
if experimenter == 'Rig 2':
    xlsx_filename = os.path.join(wdir, 'Rig_2_LED_power.xlsx')
    LED_stim_power_table = pd.read_excel(xlsx_filename, index_col = 1) #load dataframe containing LED powers and use V step as index values
    LED_stim_power_table_index = LED_stim_power_table.index.astype(str) #extract index from dataframe as string 
    LED_stim_power_table.index = LED_stim_power_table_index ## add index back to dataframe as string and not float as it was originally 
    LED_stim_power_table.columns = [ 'Voltage_step_V', 'LED_475_50%','LED_475_100%' , 'LED_520_50%', 'LED_520_100%', 'LED_543_50%', 'LED_543_100%', 'LED_630_50%', 'LED_630_100%']

else: 
    xlsx_filename = os.path.join(wdir, 'Rig_1_LED_power.xlsx')
    LED_stim_power_table = pd.read_excel(xlsx_filename, index_col = 1) #load dataframe containing LED powers and use V step as index values
    LED_stim_power_table_index = LED_stim_power_table.index.astype(str) #extract index from dataframe as string 
    LED_stim_power_table.index = LED_stim_power_table_index ## add index back to dataframe as string and not float as it was originally 
    LED_stim_power_table.columns = [ 'Voltage_step_V', 'LED_475_2%', 'LED_475_20%', 'LED_475_50%','LED_475_100%' , 'LED_520_50%', 'LED_520_100%', 'LED_575_50%', 'LED_575_100%']

### use LED max value extracted and the stimulation type to get absolute power value in mW/mm2
LED_power_pulse =  LED_stim_power_table.loc[LED_index_value, LED_stim_type] ## index and extract mW/mm2 value of pulse based on V pulse value and type of stimulation 
LED_power_pulse= round(LED_power_pulse,2 ) ## round up to 2 decimal values, can't do more since I have some 0.xx values 
LED_power_pulse= pd.Series(LED_power_pulse.astype(str)) # transform to str 
LED_power_pulse = LED_power_pulse.reset_index( drop = True)


## create time based on points extracted data 
time_points_plot = (np.arange(len(voltage_data_LED[0]))*abf.dataSecPerPoint) * 1000

###putting all data together to extract response values per trace 

trace_data_LED = pd.DataFrame({'trace_number':file_name ,
                               'date_time' : date_time, 
                               'experimenter': experimenter, 
                               'protocol' : protocol,  
                               'cell_type': cell_type_selected,
                               'V_baseline': voltage_data_baseline, 
                               'LED_wavelenght': LED_wavelength, 
                               'current_pulse_value':current_max_I_inj_final, 
                               'current_pulse_duration_total': current_pulse_length_final, 
                               'LED_time_ms': LED_time, 
                               'LED_power_mWmm': LED_power_pulse, 
                               'control_I_pre_spike': control_I_pre_spike, 
                               'control_I_mid_spike': control_I_mid_spike, 
                               'control_I_post_spike': control_I_post_spike, 
                               'pre_LED_I_spike_no' : pre_LED_I_spike_no, 
                               'LED_I_spike': LED_I_spike, 
                               'post_LED_I_spike_no' : post_LED_I_spike_no, 
                               'time_first_spike_post_LED_ms':first_spike_timing })


trace_data_master = trace_data_LED

### save data status 


### save individual file
data_final_df = trace_data_master ## date data_final array and transform into transposed dataframe
data_final_df.to_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/' + str(file_name) +'.csv', header = True) ## write file as individual csv file 

##### save data in master dataframe

"""
To make am empty dataframe with correctly labelled columns for this particular analysis: 

## make column list   
column_names = list(trace_data_master)     

## make emty dataframe + column list 
CC_inhibitory_Multi_Spike = pd.DataFrame(columns = column_names) #transform into dataframe and use given index 
## save it as .csv
CC_inhibitory_Multi_Spike.to_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_inhibitory_Multi_Spike.csv', header = True)
"""
"""
##open master sheet with data 
CC_inhibitory_Multi_Spike = pd.read_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_inhibitory_Multi_Spike.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
CC_inhibitory_Multi_Spike = CC_inhibitory_Multi_Spike.append(trace_data_master, sort = False) #adds row with new values to main dataframe

## save new version of updated dataframe as csv
CC_inhibitory_Multi_Spike.to_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_inhibitory_Multi_Spike.csv', header = True)
"""

#### plot individual LED stim 
##check for data existance and extract single rows for LED stim + current resp (done for a max of 7 stim per trace)

## data pulse and response 1
if (LED_data_df.index == 0).any() & (voltage_data_LED_df.index == 0).any():
    LED_stim_1 = LED_data_df.iloc[0]
    voltage_resp_1 = voltage_data_LED_df.iloc[0]
    title_1 = str(LED_power_pulse[0]) + 'mW/mm2'
else:
    LED_stim_1 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_1 = np.repeat(np.nan,len(current_data_LED[0]))
    title_1 = 'No LED stim applied'
    
## data pulse and response 2
if (LED_data_df.index == 1).any() & (voltage_data_LED_df.index ==1).any():
    LED_stim_2 = LED_data_df.iloc[1]
    voltage_resp_2 = voltage_data_LED_df.iloc[1]
    title_2 = str(LED_power_pulse[1]) + 'mW/mm2'
else:
    LED_stim_2 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_2 = np.repeat(np.nan,len(current_data_LED[0]))
    title_2 = 'No LED stim applied'
    
## data pulse and response 3
if (LED_data_df.index == 2).any() & (voltage_data_LED_df.index ==2).any():
    LED_stim_3 = LED_data_df.iloc[2]
    voltage_resp_3 = voltage_data_LED_df.iloc[2]
    title_3 = str(LED_power_pulse[2]) + 'mW/mm2'
else:
    LED_stim_3 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_3 = np.repeat(np.nan,len(current_data_LED[0]))
    title_3 = 'No LED stim applied'

## data pulse and response 4
if (LED_data_df.index == 3).any() & (voltage_data_LED_df.index ==3).any():
    LED_stim_4 = LED_data_df.iloc[3]
    voltage_resp_4 = voltage_data_LED_df.iloc[3]
    title_4 = str(LED_power_pulse[3]) + 'mW/mm2'
else:
    LED_stim_4 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_4 = np.repeat(np.nan,len(current_data_LED[0]))
    title_4 = 'No LED stim applied'

## data pulse and response 5
if (LED_data_df.index == 4).any() & (voltage_data_LED_df.index ==4).any():
    LED_stim_5 = LED_data_df.iloc[4]
    voltage_resp_5 = voltage_data_LED_df.iloc[4]
    title_5 = str(LED_power_pulse[4]) + 'mW/mm2'
else:
    LED_stim_5 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_5 = np.repeat(np.nan,len(current_data_LED[0]))
    title_5 = 'No LED stim applied'

## data pulse and response 6
if (LED_data_df.index == 5).any() & (voltage_data_LED_df.index ==5).any():
    LED_stim_6 = LED_data_df.iloc[5]
    voltage_resp_6 = voltage_data_LED_df.iloc[5]
    title_6 = str(LED_power_pulse[5]) + 'mW/mm2'
else:
    LED_stim_6 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_6 = np.repeat(np.nan,len(current_data_LED[0]))
    title_6 = 'No LED stim applied'
     
## data pulse and response 7
if (LED_data_df.index == 6).any() & (voltage_data_LED_df.index == 6).any():
    LED_stim_7 = LED_data_df.iloc[6]
    voltage_resp_7 = voltage_data_LED_df.iloc[6]
    title_7 = str(LED_power_pulse[6]) + 'mW/mm2'
else:
    LED_stim_7 = np.repeat(np.nan,len(LED_data[0]))
    voltage_resp_7 = np.repeat(np.nan,len(current_data_LED[0]))
    title_7 = 'No LED stim applied'
    
#### plot figure of LED stim + response 

fig2 = plt.figure(figsize =(20,5))
sub1 = plt.subplot(2,7,1)
sub1.plot(voltage_resp_1, linewidth=0.3, color = '0.2')
sub1.set_title(title_1, color = '0.2')
sub1.tick_params(axis='x', colors='white')
sub1.spines['bottom'].set_color('white')
plt.setp(sub1.get_xticklabels(), visible = False)
sns.despine()

sub2 = plt.subplot(2,7,2)
sub2.plot(voltage_resp_2, linewidth=0.3, color = '0.2')
sub2.set_title(title_2, color = '0.2')
sub2.tick_params(axis='x', colors='white')
sub2.spines['bottom'].set_color('white')
plt.setp(sub2.get_xticklabels(), visible = False)
sns.despine()

sub3 = plt.subplot(2,7,3)
sub3.plot(voltage_resp_3, linewidth=0.3, color = '0.2')
sub3.set_title(title_3, color = '0.2')
sub3.tick_params(axis='x', colors='white')
sub3.spines['bottom'].set_color('white')
plt.setp(sub3.get_xticklabels(), visible = False)
sns.despine()

sub4 = plt.subplot(2,7,4)
sub4.plot(voltage_resp_4, linewidth=0.3, color = '0.2')
sub4.set_title(title_4, color = '0.2')
sub4.tick_params(axis='x', colors='white')
sub4.spines['bottom'].set_color('white')
plt.setp(sub4.get_xticklabels(), visible = False)
sns.despine()

sub5 = plt.subplot(2,7,5)
sub5.plot(voltage_resp_5, linewidth=0.3, color = '0.2')
sub5.set_title(title_5, color = '0.2')
sub5.tick_params(axis='x', colors='white')
sub5.spines['bottom'].set_color('white')
plt.setp(sub5.get_xticklabels(), visible = False)
sns.despine()

sub6 = plt.subplot(2,7,6)
sub6.plot(voltage_resp_6, linewidth=0.3, color = '0.2')
sub6.set_title(title_6, color = '0.2')
sub6.tick_params(axis='x', colors='white')
sub6.spines['bottom'].set_color('white')
plt.setp(sub6.get_xticklabels(), visible = False)
sns.despine()

sub7 = plt.subplot(2,7,7)
sub7.plot(voltage_resp_7, linewidth=0.3, color = '0.2')
sub7.set_title(title_7, color = '0.2')
sub7.tick_params(axis='x', colors='white')
sub7.spines['bottom'].set_color('white')
plt.setp(sub7.get_xticklabels(), visible = False)
sns.despine()

