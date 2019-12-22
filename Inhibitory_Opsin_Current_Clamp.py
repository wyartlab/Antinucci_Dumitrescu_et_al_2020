#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:32:17 2019

@author: adna.dumitrescu

Script opens abf file with current clamp data aquired during inhibitory opsin stimulation. 

How the script works: 
1. Asks user to provide a file path for the abf file to be analysed
2. Asks user for meta-data that cannot be automatically extracted from abf file (such as stimulation regime and cell type etc)
3. Finds all points where the LED is ON and uses these time-bins to extract data from the trace in which voltage values are recorded. 
4. Puts all extracted data in a single .csv file named after the trace
5. Adds data as a new row to a master .csv file for the whole experiment 
6. Outputs in line graphs showing cell response to current injection (if present) and LED stimulations

Metadata collected: file name, date, experimenter, protocol type, opsin type, LED wavelenght, power and stimulation time.
Data that is calculated by script: 
Baseline current injection level: mean of points across first 100ms of the trace
Resting membrane voltage: mean of points across first 1000ms of the trace
Response type: spike (>-30mV) vs subthreshold (<-30mV)
Maximum voltage deflection attained per each LED stimulation regardless of response type: maximum value within a time period between LED onset plus 1000ms
Maximum voltage deflection attained per each LED stimulation from baseline regardless of response type: maximum value within a time period between LED onset plus 1000ms minus baseline V level
Time to peak V deflection response in ms: time at which max value is found from start of pulse. 
If cell is spiking: extract number of spikes and instantaneous frequency 
If a current pulse is applied during the same trace extract: 
current injection max (pA), stimulation time, response type, max depolarisation level absolute and baseline substracted, and time to peak depolarisation.
Extracts raw voltage trace data points corresponding to LED stimulation
Extracts raw LED trace data points corresponding to LED stimulation
Extract raw current injection data points corresponding to time bin of current pulse

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
       10 : 'LED_575_100%' ,  
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

## function to get spike frequency 
def get_spike_frequency (data):
    index_spike_list = np.array(data[0]) ## feed in data output from find peaks function which is a tuple containing at [0] list of  spikes indices and at [1] peak heights. Only [0] level output is used in this script 
    
    if index_spike_list.size: ### if size = True (i.e values present) spikes were detected proceed to try and calculate frequency
        print('Frequency calculation started')
        index_substract = np.diff(index_spike_list) ## get time between each spike 
        index_convert_time_ms =  index_substract / sampling_rate ## convert time to ms 
    
        if index_convert_time_ms.size == 0: #if index substraction is 0 it means that only 1 spike was present. Not possible to run freq calculation 
            print('Frequency calculation not possible on a single spike')
            freq_Hz = np.nan
            return (freq_Hz)
        
        elif index_convert_time_ms.size == 1:  #if index substraction is a single number extract this and transform in ms as freq result
            print('Frequency calculation done as time between 2 spikes')
            freq_Hz = np.float(1 / index_convert_time_ms) * 1000
            return (freq_Hz)
        
        else: ## if more then 2 inter spike interval detected get average 
            print('Frequency calculation done as average of multiple spike interval times')
            mean_pulse_time_ms = np.mean(index_convert_time_ms)
            freq_Hz = np.float(1 /mean_pulse_time_ms) * 1000
            return (freq_Hz)

    else: ### if size = False it means no spiked were detected so set freq values to nan
        print ('Frequency calculation: no spike detected')
        freq_Hz = np.nan
        return (freq_Hz)        


'''
Part 1: find where LED pulses have been applied
'''
###### find index values where LED is ON use this to extract all other info from current trace
LED_idx = (np.where(LED_trace > 0.2)) # index of values in LED trace where V values are over 0
LED_array = np.asarray(LED_idx) #transform into an array for next step


LED_idx_cons = consecutive(LED_array[0]) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim
LED_idx_cons_df = pd.DataFrame(LED_idx_cons) #transform data into dataframe for easier processing below 
LED_idx_cons_df_T = LED_idx_cons_df.T
LED_idx_cons_df_T_last = LED_idx_cons_df_T.iloc[19899:19999]
LED_idx_cons_df_last = LED_idx_cons_df_T.iloc[19899:19999].T
LED_end_idx = LED_idx_cons_df_last.values

### determine lenght of LED Stim
LED_time =pd.Series (LED_idx_cons_df.count(axis = 'columns'), name ='LED_time_ON' )#count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
LED_time = round(LED_time / sampling_rate) # transform number of elements into actual ms rounded 

### increase array to add 100ms before and 1s after current pulse to collect more current data
LED_expand_idx =[np.concatenate([ [x[0]-i-1 for i in reversed(range(1999))], x, [x[-1]+i+1 for i in range(19999)] ]) for x in LED_idx_cons]
LED_steady_calculation = [np.concatenate([x, [x[-1]+i+1 for i in range(19999)] ]) for x in LED_end_idx]

###### use selected LED indices to extract current and voltage data
voltage_data_LED = [voltage_trace [i] for i in LED_expand_idx]
current_data_LED =  [current_trace_baseline_substracted [i] for i in LED_expand_idx] # use index extracted for each individual pulse to extract voltage values 
voltage_data_LED_df = pd.DataFrame (voltage_data_LED)
voltage_data_steady = [voltage_trace [i] for i in LED_steady_calculation]

LED_data = [LED_trace [i] for i in LED_expand_idx] 
LED_data_df = pd.DataFrame(LED_data)

#### calculate delay between LED ON and peak V response 
opsin_max_resp_idx = voltage_data_LED_df.idxmax (1) ### returns index 
LED_on = 1999 ## since we add 100ms before the start of the pulse 
opsin_resp_max_delay_ms = (opsin_max_resp_idx - LED_on) / sampling_rate

####### determine max current and voltage  response value 
current_max_LED = list(map(max, current_data_LED)) #get list of all current max values per pulse (need to be min since this is VC)

if cell_type_selected == 'GtACR1' or cell_type_selected == 'GtACR2':
    voltage_max_LED = list(map(max, voltage_data_LED)) #get list of all current min values per pulse (need to be min since this is VC) also inward excitatory current for these 2 inhibitory opsins 
    voltage_max_steady_LED = list(map(max, voltage_data_steady))
else:
    voltage_max_LED = list(map(min, voltage_data_LED))
    voltage_max_steady_LED = list(map(max, voltage_data_steady))

current_max_LED = list(map(abs, current_max_LED))
voltage_baseline_LED = voltage_data_baseline
voltage_deflection_LED = voltage_max_LED - voltage_baseline_LED
voltage_deflection_steady_LED = voltage_max_steady_LED - voltage_baseline_LED
stim_type_LED = 'LED_pulse'
 
#### spike count extract number of spikes for each LED stim. Calculated for a max of 7 pulses per trace. 

### LED_stim_1_data
try:
    spike_LED_1 = find_peaks(voltage_data_LED[0], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 

except IndexError:
    spike_LED_1 = 'No LED stim applied' ## if there is no pulse 

if type(spike_LED_1) == str:
    print ('No LED stim applied')
    response_type_LED_1 = np.nan
    spike_count_LED_1 = np.nan
    subthresh_event_LED_1 = np.nan
    voltage_max_LED_1 = np.nan
    V_resp_delay_LED_1 = np.nan
    V_deflection_LED_1 = np.nan
    spike_freq_LED_1 = np.nan
    
else:    

    if spike_LED_1 [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 1 gave rise to subthreshold event')
        response_type_LED_1 = 'sub_thresh'
        spike_count_LED_1 = 0
        subthresh_event_LED_1 = 1
        voltage_max_LED_1 = voltage_max_LED[0]
        voltage_max_LED_1_idx = opsin_max_resp_idx[0]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_1 = ( voltage_max_LED_1_idx -LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_1 = voltage_max_LED_1 - voltage_baseline_LED 
        voltage_data_points_LED_1 = voltage_data_LED[0]
        response_type_LED_1 = 'sub_thresh_event'
        spike_freq_LED_1  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 1 gave rise to Spike(s)')
        response_type_LED_1 = 'Spike'
        spike_count_LED_1 = len(spike_LED_1[0])
        subthresh_event_LED_1 = 0
        voltage_max_LED_1 = spike_LED_1[1]
        voltage_max_LED_1_tup = sum(voltage_max_LED_1.items(), ())
        voltage_max_LED_1_arr = voltage_max_LED_1_tup[1]
        voltage_max_LED_1 = voltage_max_LED_1_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_1_idx = spike_LED_1[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_1 = (voltage_max_LED_1_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_1=  V_resp_delay_LED_1[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_1 = voltage_data_LED[0]
        V_deflection_LED_1 = voltage_max_LED_1 - voltage_baseline_LED 
        response_type_LED_1 = 'Spike'
        spike_freq_LED_1 = get_spike_frequency (spike_LED_1 )
        

### LED_stim_2_data
try:
    spike_LED_2 = find_peaks(voltage_data_LED[1], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 

except IndexError:
    spike_LED_2 = 'No LED stim applied' ## if there is no pulse 

if type(spike_LED_2) == str:
    print ('No LED stim applied')
    response_type_LED_2 = np.nan
    spike_count_LED_2 = np.nan
    subthresh_event_LED_2 = np.nan
    voltage_max_LED_2 = np.nan
    V_resp_delay_LED_2 = np.nan
    V_deflection_LED_2 = np.nan
    spike_freq_LED_2 = np.nan

else:    

    if spike_LED_2 [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 2 gave rise to subthreshold event')
        response_type_LED_2 = 'sub_thresh'
        spike_count_LED_2 = 0
        subthresh_event_LED_2 = 1
        voltage_max_LED_2 = voltage_max_LED[1]
        voltage_max_LED_2_idx = opsin_max_resp_idx[1]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_2 = ( voltage_max_LED_2_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_2 = voltage_max_LED_2 - voltage_baseline_LED 
        voltage_data_points_LED_2 = voltage_data_LED[1]
        response_type_LED_2 = 'sub_thresh_event'
        spike_freq_LED_2  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 2 gave rise to Spike(s)')
        response_type_LED_2 = 'Spike'
        spike_count_LED_2 = len(spike_LED_2[0])
        subthresh_event_LED_2 = 0
        voltage_max_LED_2 = spike_LED_2[1]
        voltage_max_LED_2_tup = sum(voltage_max_LED_2.items(), ())
        voltage_max_LED_2_arr = voltage_max_LED_2_tup[1]
        voltage_max_LED_2 = voltage_max_LED_2_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_2_idx = spike_LED_2[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_2 = (voltage_max_LED_2_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_2=  V_resp_delay_LED_2[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_2 = voltage_data_LED[1]
        V_deflection_LED_2 = voltage_max_LED_2 - voltage_baseline_LED 
        response_type_LED_2 = 'Spike'
        spike_freq_LED_2 = get_spike_frequency (spike_LED_2 )

### LED_stim_3_data
try:
    spike_LED_3 = find_peaks(voltage_data_LED[2], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 

except IndexError:
    spike_LED_3 = 'No LED stim applied' ## if there is no pulse 

if type(spike_LED_3) == str:
    print ('No LED stim applied')
    response_type_LED_3 = np.nan
    spike_count_LED_3 = np.nan
    subthresh_event_LED_3 = np.nan
    voltage_max_LED_3 = np.nan
    V_resp_delay_LED_3 = np.nan
    V_deflection_LED_3 = np.nan
    spike_freq_LED_3 = np.nan

else:    

    if spike_LED_3 [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 3 gave rise to subthreshold event')
        response_type_LED_3 = 'sub_thresh'
        spike_count_LED_3 = 0
        subthresh_event_LED_3 = 1
        voltage_max_LED_3 = voltage_max_LED[2]
        voltage_max_LED_3_idx = opsin_max_resp_idx[2]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_3 = ( voltage_max_LED_3_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_3 = voltage_max_LED_3 - voltage_baseline_LED 
        voltage_data_points_LED_3 = voltage_data_LED[2]
        response_type_LED_3 = 'sub_thresh_event'
        spike_freq_LED_3  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 3 gave rise to Spike(s)')
        response_type_LED_3 = 'Spike'
        spike_count_LED_3 = len(spike_LED_3[0])
        subthresh_event_LED_3 = 0
        voltage_max_LED_3 = spike_LED_3[1]
        voltage_max_LED_3_tup = sum(voltage_max_LED_3.items(), ())
        voltage_max_LED_3_arr = voltage_max_LED_3_tup[1]
        voltage_max_LED_3 = voltage_max_LED_3_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_3_idx = spike_LED_3[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_3 = (voltage_max_LED_3_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_3 =  V_resp_delay_LED_3[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_3 = voltage_data_LED[2]
        V_deflection_LED_3 = voltage_max_LED_3 - voltage_baseline_LED 
        response_type_LED_3 = 'Spike'
        spike_freq_LED_3 = get_spike_frequency (spike_LED_3 )

### LED_stim_4_data
try:
    spike_LED_4 = find_peaks(voltage_data_LED[3], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 

except IndexError:
    spike_LED_4 = 'No LED stim applied' ## if there is no pulse 

if type(spike_LED_4) == str:
    print ('No LED stim applied')
    response_type_LED_4 = np.nan
    spike_count_LED_4 = np.nan
    subthresh_event_LED_4 = np.nan
    voltage_max_LED_4 = np.nan
    V_resp_delay_LED_4 = np.nan
    V_deflection_LED_4 = np.nan
    spike_freq_LED_4 = np.nan

else:    

    if spike_LED_4 [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 4 gave rise to subthreshold event')
        response_type_LED_4 = 'sub_thresh'
        spike_count_LED_4 = 0
        subthresh_event_LED_4 = 1
        voltage_max_LED_4 = voltage_max_LED[3]
        voltage_max_LED_4_idx = opsin_max_resp_idx[3]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_4 = ( voltage_max_LED_4_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_4 = voltage_max_LED_4 - voltage_baseline_LED 
        voltage_data_points_LED_4 = voltage_data_LED[3]
        response_type_LED_4 = 'sub_thresh_event'
        spike_freq_LED_4  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 4 gave rise to Spike(s)')
        response_type_LED_4 = 'Spike'
        spike_count_LED_4 = len(spike_LED_4[0])
        subthresh_event_LED_4 = 0
        voltage_max_LED_4 = spike_LED_4[1]
        voltage_max_LED_4_tup = sum(voltage_max_LED_4.items(), ())
        voltage_max_LED_4_arr = voltage_max_LED_4_tup[1]
        voltage_max_LED_4 = voltage_max_LED_4_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_4_idx = spike_LED_4[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_4 = (voltage_max_LED_4_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_4 =  V_resp_delay_LED_4[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_4 = voltage_data_LED[3]
        V_deflection_LED_4 = voltage_max_LED_4 - voltage_baseline_LED 
        response_type_LED_4 = 'Spike'
        spike_freq_LED_4 = get_spike_frequency (spike_LED_4 )


### LED_stim_5_data        
try:
    spike_LED_5 = find_peaks(voltage_data_LED[4], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 
except IndexError:
    spike_LED_5 = 'No LED stim applied' ## if there is no pulse 
   
if type(spike_LED_5) == str:
    print ('No LED stim applied')
    response_type_LED_5 = np.nan
    spike_count_LED_5 = np.nan
    subthresh_event_LED_5 = np.nan
    voltage_max_LED_5 = np.nan
    V_resp_delay_LED_5 = np.nan
    V_deflection_LED_5 = np.nan
    spike_freq_LED_5 = np.nan

else:   
    
    if spike_LED_5[0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 5 gave rise to subthreshold event')
        response_type_LED_5 = 'sub_thresh'
        spike_count_LED_5 = 0
        subthresh_event_LED_5 = 1
        voltage_max_LED_5 = voltage_max_LED[4]
        voltage_max_LED_5_idx = opsin_max_resp_idx[4]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_5 = ( voltage_max_LED_5_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_5 = voltage_max_LED_5 - voltage_baseline_LED 
        voltage_data_points_LED_5 = voltage_data_LED[4]
        response_type_LED_5 = 'sub_thresh_event'
        spike_freq_LED_5  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 5 gave rise to Spike(s)')
        response_type_LED_5 = 'Spike'
        spike_count_LED_5 = len(spike_LED_5[0])
        subthresh_event_LED_5 = 0
        voltage_max_LED_5 = spike_LED_5[1]
        voltage_max_LED_5_tup = sum(voltage_max_LED_5.items(), ())
        voltage_max_LED_5_arr = voltage_max_LED_5_tup[1]
        voltage_max_LED_5 = voltage_max_LED_5_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_5_idx = spike_LED_5[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_5 = (voltage_max_LED_5_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_5 =  V_resp_delay_LED_5[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_5 = voltage_data_LED[4]
        V_deflection_LED_5 = voltage_max_LED_5 - voltage_baseline_LED 
        response_type_LED_5 = 'Spike'
        spike_freq_LED_5 = get_spike_frequency (spike_LED_5 )
        

### LED_stim_6_data        
try:
    spike_LED_6 = find_peaks(voltage_data_LED[5], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 
except IndexError:
    spike_LED_6 = 'No LED stim applied' ## if there is no pulse 
   
if type(spike_LED_6) == str:
    print ('No LED stim applied')
    response_type_LED_6 = np.nan
    spike_count_LED_6 = np.nan
    subthresh_event_LED_6 = np.nan
    voltage_max_LED_6 = np.nan
    V_resp_delay_LED_6 = np.nan
    V_deflection_LED_6 = np.nan
    spike_freq_LED_6 = np.nan

else:   
    
    if spike_LED_6[0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 6 gave rise to subthreshold event')
        response_type_LED_6 = 'sub_thresh'
        spike_count_LED_6 = 0
        subthresh_event_LED_6 = 1
        voltage_max_LED_6 = voltage_max_LED[5]
        voltage_max_LED_6_idx = opsin_max_resp_idx[5]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_6 = ( voltage_max_LED_6_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_6 = voltage_max_LED_6 - voltage_baseline_LED 
        voltage_data_points_LED_6 = voltage_data_LED[5]
        response_type_LED_6 = 'sub_thresh_event'
        spike_freq_LED_6  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 6 gave rise to Spike(s)')
        response_type_LED_6 = 'Spike'
        spike_count_LED_6 = len(spike_LED_6[0])
        subthresh_event_LED_6 = 0
        voltage_max_LED_6 = spike_LED_6[1]
        voltage_max_LED_6_tup = sum(voltage_max_LED_6.items(), ())
        voltage_max_LED_6_arr = voltage_max_LED_6_tup[1]
        voltage_max_LED_6 = voltage_max_LED_6_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_6_idx = spike_LED_6[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_6 = (voltage_max_LED_6_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_6 =  V_resp_delay_LED_6[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_6 = voltage_data_LED[5]
        V_deflection_LED_6 = voltage_max_LED_6 - voltage_baseline_LED 
        response_type_LED_6 = 'Spike'
        spike_freq_LED_6 = get_spike_frequency (spike_LED_6 )

### LED_stim_7_data        
try:
    spike_LED_7 = find_peaks(voltage_data_LED[6], height=-30) #find peaks with height over 0mV in array 0 from voltage_data 
except IndexError:
    spike_LED_7 = 'No LED stim applied' ## if there is no pulse 
   
if type(spike_LED_7) == str:
    print ('No LED stim applied')
    response_type_LED_7 = np.nan
    spike_count_LED_7 = np.nan
    subthresh_event_LED_7 = np.nan
    voltage_max_LED_7 = np.nan
    V_resp_delay_LED_7 = np.nan
    V_deflection_LED_7 = np.nan
    spike_freq_LED_7 = np.nan


else:   
    
    if spike_LED_7[0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim 7 gave rise to subthreshold event')
        response_type_LED_7 = 'sub_thresh'
        spike_count_LED_7 = 0
        subthresh_event_LED_7 = 1
        voltage_max_LED_7 = voltage_max_LED[6]
        voltage_max_LED_7_idx = opsin_max_resp_idx[6]
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_7 = ( voltage_max_LED_7_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED_7 = voltage_max_LED_7 - voltage_baseline_LED 
        voltage_data_points_LED_7 = voltage_data_LED[6]
        response_type_LED_7 = 'sub_thresh_event'
        spike_freq_LED_7  = np.nan
    
    else: ## is spike detected extract values 
        print('LED stim 7 gave rise to Spike(s)')
        response_type_LED_7 = 'Spike'
        spike_count_LED_7 = len(spike_LED_7[0])
        subthresh_event_LED_7 = 0
        voltage_max_LED_7 = spike_LED_7[1]
        voltage_max_LED_7_tup = sum(voltage_max_LED_7.items(), ())
        voltage_max_LED_7_arr = voltage_max_LED_7_tup[1]
        voltage_max_LED_7 = voltage_max_LED_7_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_7_idx = spike_LED_7[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED_7 = (voltage_max_LED_7_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED_7 =  V_resp_delay_LED_7[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_points_LED_7 = voltage_data_LED[6]
        V_deflection_LED_7 = voltage_max_LED_7 - voltage_baseline_LED 
        response_type_LED_7 = 'Spike'
        spike_freq_LED_7 = get_spike_frequency (spike_LED_7 )



max_list_index = len(LED_data)

response_type_LED_all = [response_type_LED_1, response_type_LED_2, response_type_LED_3, response_type_LED_4, response_type_LED_5, response_type_LED_6, response_type_LED_7]
del response_type_LED_all [max_list_index:7]

spike_count_LED_all = [spike_count_LED_1, spike_count_LED_2, spike_count_LED_3, spike_count_LED_4, spike_count_LED_5, spike_count_LED_6, spike_count_LED_7]
del spike_count_LED_all [max_list_index:7]

spike_count_total_LED_trace = sum(spike_count_LED_all)

subthresh_event_LED_all = [subthresh_event_LED_1, subthresh_event_LED_2, subthresh_event_LED_3, subthresh_event_LED_4, subthresh_event_LED_5, subthresh_event_LED_6, subthresh_event_LED_7]
del subthresh_event_LED_all [max_list_index:7]

spike_freq_LED_all = [spike_freq_LED_1, spike_freq_LED_2, spike_freq_LED_3, spike_freq_LED_4, spike_freq_LED_5, spike_freq_LED_6, spike_freq_LED_7 ]
del spike_freq_LED_all [max_list_index:7]  

V_resp_delay_LED_all = [ V_resp_delay_LED_1,  V_resp_delay_LED_2,  V_resp_delay_LED_3,  V_resp_delay_LED_4,  V_resp_delay_LED_5,  V_resp_delay_LED_6,  V_resp_delay_LED_7]
del V_resp_delay_LED_all [max_list_index:7]  


if cell_type_selected == 'GtACR1' or cell_type_selected == 'GtACR2':
    voltage_max_LED_all = list(map(max, voltage_data_LED)) #get list of all current min values per pulse (need to be min since this is VC) also inward excitatory current for these 2 inhibitory opsins 
else:
    voltage_max_LED_all = list(map(min, voltage_data_LED))

V_deflection_LED_all = voltage_max_LED_all - voltage_baseline_LED 


### extract tau off value from steady to baseline


if cell_type_selected == 'GtACR1' or cell_type_selected == 'GtACR2':
    from exponentialFitGetTauInhibitory import exponentialFitGetTau # fits to depolarising inward current 
else:
    from exponentialFitGetTau  import exponentialFitGetTau # fits to hyperpolarising outward current 

deactivation_tau = []
for data_row in voltage_data_steady:
    y = data_row
    x = np.linspace(1, len(y), len(y))
    tau = exponentialFitGetTau(x, y, 1, 19000)
    tau_LED_stim = tau / sampling_rate
    deactivation_tau.append(tau_LED_stim)
    print('Deactivation time constant for this photocurrent response is ' +str(round(tau_LED_stim,2)) + ' ms')


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
                               'stim_type': stim_type_LED, 
                               'V_baseline': voltage_data_baseline, 
                               'LED_wavelenght': LED_wavelength,
                               'LED_time_ms': LED_time, 
                               'LED_power_mWmm': LED_power_pulse, 
                               'response_type_LED':response_type_LED_all, 
                               'spike_per_LED_stim': spike_count_LED_all, 
                               'spike_freq_LED': spike_freq_LED_all, 
                               'subthresh_per_LED_stim': subthresh_event_LED_all, 
                               'Max_V_response_mV' : voltage_max_LED_all, 
                               'total_V_deflection_from_baseline_mV': V_deflection_LED_all , 
                               'Steady_state_V_deflection_mV':voltage_deflection_steady_LED ,
                               'Time_to_peak_response_ms': V_resp_delay_LED_all, 
                               'deactivation_time_ms': deactivation_tau, 
                               'voltage_points_plot': voltage_data_LED, 
                               'LED_points_plot': LED_data  })


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
CC_opsin_inhibitory_master = pd.DataFrame(columns = column_names) #transform into dataframe and use given index 
## save it as .csv
CC_opsin_inhibitory_master.to_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_opsin_inhibitory_master.csv', header = True)
"""

##open master sheet with data 
CC_opsin_inhibitory_master = pd.read_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_opsin_inhibitory_master.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
CC_opsin_inhibitory_master = CC_opsin_inhibitory_master.append(trace_data_master, sort = False) #adds row with new values to main dataframe

## save new version of updated dataframe as csv
CC_opsin_inhibitory_master.to_csv('/Users/adna.dumitrescu/Documents/Wyart_Postdoc/Data/OPSIN_testing_project/Opsin_Ephys_Analysis/CC_analysis/CC_opsin_inhibitory_master.csv', header = True)

#### plot figure of LED stim + response 

## create arrays for sample data to be plotted  
LED_expand_idx_plot =[np.concatenate([ [x[0]-i-1 for i in reversed(range(1999))], x, [x[-1]+i+1 for i in range(19999)] ]) for x in LED_idx_cons] ### add 5ms pre LED start and 100ms after .  
voltage_data_plot = [voltage_trace [i] for i in LED_expand_idx_plot]

time_points_plot = (np.arange(len(voltage_data_plot[0]))*abf.dataSecPerPoint) * 1000
time_points_plot = [time_points_plot] * len(voltage_data_LED)



fig1 = plt.figure(figsize =(30,5))
fig1.subplots_adjust(wspace=0.5)

for counter, (voltage, time, power) in enumerate (zip (voltage_data_plot, time_points_plot, LED_power_pulse), start = 1): 
    sub = plt.subplot(2,len(voltage_data_LED),counter)
    markers_on = [1999, 22000]
    sub.plot (time, voltage, '-om', markevery = markers_on, markerfacecolor="m", markeredgecolor = 'w', linewidth=0.3, color = '0.2')
    sub.tick_params (axis = 'x', colors='black') 
    sub.spines['bottom'].set_color('black')
    sub.set_title(power + ' mW/mm2', color = '0.2')
    plt.setp(sub.get_xticklabels(), visible = True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Photocurrent (pA)')
    plt.text (1500, -70, 'LED ON/OFF', ha='center', color='magenta')
    plt.suptitle('Example opsin induced voltage deflections from this trace', fontsize=16)

    sns.despine()

print ('Total number of spikes detected in this trace: N = ' +str(spike_count_total_LED_trace ))
