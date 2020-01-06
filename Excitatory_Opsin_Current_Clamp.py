#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:10:37 2019
@author: adna.dumitrescu

Script opens abf file with current clamp data aquired during excitatory opsin stimulation. 

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
            print('Frequency calculation not possible on a single spike\n')
            freq_Hz = np.nan
            return (freq_Hz)
        
        elif index_convert_time_ms.size == 1:  #if index substraction is a single number extract this and transform in ms as freq result
            print('Frequency calculation done as time between 2 spikes\n')
            freq_Hz = np.float(1 / index_convert_time_ms) * 1000
            return (freq_Hz)
        
        else: ## if more then 2 inter spike interval detected get average 
            print('Frequency calculation done as average of multiple spike interval times\n')
            mean_pulse_time_ms = np.mean(index_convert_time_ms)
            freq_Hz = np.float(1 /mean_pulse_time_ms) * 1000
            return (freq_Hz)

    else: ### if size = False it means no spiked were detected so set freq values to nan
        print ('Frequency calculation: no spike detected\n')
        freq_Hz = np.nan
        return (freq_Hz)        

'''
Part 1 - find if a current pulse has been applied
'''
#### find if a current pulse was applied: 
if np.max(current_trace) > 40: ## if pulse detected in this trace
   
    print('Current pulse applied in this trace')   
    
    ### find indices where current pulse is applied
    current_injection_idx = np.asarray(np.where(current_trace_baseline_substracted > 10)) # index of values in current trace where the current injected in more then 10pA
    current_injection_idx_cons = consecutive(current_injection_idx[0]) #find consecutive indexes where current over 10pa is applied and split into separate arrays --> each array would be 1 pulse 
    current_injection_idx_cons_df = pd.DataFrame(current_injection_idx_cons) #transform data into dataframe for easier processing below 

    current_pulse_length = current_injection_idx_cons_df.count(axis = 'columns') #count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
    current_pulse_length_final = round(current_pulse_length / sampling_rate) # transform number of elements into actual ms rounded 
    current_pulse_length_final = current_pulse_length_final.drop_duplicates ()

    ### increase array to add 2ms before and after current pulse to collect more voltage data
    current_pulses_expanded =[np.concatenate([ [x[0]-i-1 for i in reversed(range(39))], x, [x[-1]+i+1 for i in range(39)] ]) for x in current_injection_idx_cons]
    current_data_I_pulse = [current_trace_baseline_substracted [i] for i in current_pulses_expanded] # use index extracted for each individual pulse to extract current values 
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
    voltage_data_I_injection = [voltage_trace [i] for i in current_pulses_expanded]
    voltage_data_I_injection_df = pd.DataFrame(voltage_data_I_injection)
    spike_I_stim = find_peaks(voltage_data_I_injection[0], height=0) ##check if there are any spikes present
    
    if spike_I_stim [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('Curent pulses gave rise to subthreshold event\n')
        spike_count_I_stim = 0
        subthresh_event = 1
        voltage_max_I_inj = list(map(max, voltage_data_I_injection))
        voltage_max_I_inj_idx = voltage_data_I_injection_df.idxmax(1)
        I_pulse_start_idx = 39  ## because I added 2ms worth of data before each pulse 
        V_resp_delay_I_inj= (voltage_max_I_inj_idx - I_pulse_start_idx) / sampling_rate
        I_inj_V_deflection = voltage_max_I_inj - voltage_data_baseline 
        voltage_data_I_injection_points = voltage_data_I_injection
        I_pulse_response_type = 'sub_thresh_event'
        I_pulse_spike_freq  = np.nan
    
    else: ## is spike detected extract values 
        print('Curent pulses gave rise to Spike(s)\n')
        spike_count_I_stim = len(spike_I_stim[0])
        voltage_max_I_inj = spike_I_stim[1]
        voltage_max_I_inj_tup = sum(voltage_max_I_inj.items(), ())
        voltage_max_I_inj_arr = voltage_max_I_inj_tup[1]
        voltage_max_I_inj = voltage_max_I_inj_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_I_inj_idx = spike_I_stim[0] ###
        V_resp_delay_I_inj = (voltage_max_I_inj_idx - I_pulse_start_idx) / sampling_rate
        V_resp_delay_I_inj =  V_resp_delay_I_inj[0].tolist() ## extract delay just to the 1st encoutered spike 
        voltage_data_I_injection_points = [voltage_data_I_injection[0]]
        I_inj_V_deflection =  voltage_max_I_inj - voltage_data_baseline 
        I_pulse_spike_freq = get_spike_frequency (spike_I_stim)
        I_pulse_response_type = 'Spike'
        
else:    
    ## if no pulse detected make empty lists 
    print('No current pulse applied in this trace\n')
    current_pulse_length_final = np.nan 
    current_max_I_inj_final = np.nan
    spike_count_I_stim = np.nan 
    voltage_max_I_inj = np.nan
    I_inj_V_deflection = np.nan
    V_resp_delay_I_inj = np.nan
    voltage_data_I_injection_points = []
    stim_type_I_inj = 'no_I_pulse_present'
    I_pulse_response_type = np.nan   
    I_pulse_spike_freq  = np.nan
    I_pulse_response_type = np.nan
    voltage_data_I_injection = np.nan
    
        
   
### put all data extracted together     
trace_data_I_inj = pd.DataFrame ({'trace_number':file_name ,'date_time' : date_time, 'experimenter': experimenter,'protocol' : protocol,  'cell_type': cell_type_selected, 'stim_type': stim_type_I_inj, 'V_baseline': voltage_data_baseline, 'I_inj_duration_ms': current_pulse_length_final, 'I_inj_max_value_pA':current_max_I_inj_final, 'I_inj_response_type': I_pulse_response_type, 'spike_per_I_inj': spike_count_I_stim, 'I_pulse_spike_freq': I_pulse_spike_freq,  'I_inj_voltage_max_resp_mV': voltage_max_I_inj,  'I_inj_V_deflection_total_from_base_mV' : I_inj_V_deflection, 'V_resp_max_delay_1st_resp_ms': V_resp_delay_I_inj, 'V_data_points': voltage_data_I_injection_points })

'''
Part 2: find if LED pulses have been applied
'''
###### find index values where LED is ON use this to extract all other info from current trace
LED_idx = (np.where(LED_trace > 0.1)) # index of values in LED trace where V values are over 0
LED_array = np.asarray(LED_idx) #transform into an array for next step


LED_idx_cons = consecutive(LED_array[0]) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim
LED_idx_cons_df = pd.DataFrame(LED_idx_cons) #transform data into dataframe for easier processing below 

### determine lenght of LED Stim
LED_time =pd.Series (LED_idx_cons_df.count(axis = 'columns'), name ='LED_time_ON' )#count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
LED_time = round(LED_time / sampling_rate) # transform number of elements into actual ms rounded 

### increase array to add 100ms before and 1s after current pulse to collect more current data
LED_expand_idx =[np.concatenate([ [x[0]-i-1 for i in reversed(range(1999))], x, [x[-1]+i+1 for i in range(19999)] ]) for x in LED_idx_cons]

###### use selected LED indices to extract current and voltage data
voltage_data_LED = [voltage_trace [i] for i in LED_expand_idx]
current_data_LED =  [current_trace_baseline_substracted [i] for i in LED_expand_idx] # use index extracted for each individual pulse to extract voltage values 
voltage_data_LED_df = pd.DataFrame (voltage_data_LED)
LED_data = [LED_trace [i] for i in LED_expand_idx] 
LED_data_df = pd.DataFrame(LED_data)

#### calculate delay between LED ON and peak V response 
opsin_max_resp_idx = voltage_data_LED_df.idxmax (1) ### returns index 
LED_on = 1999 ## since we add 100ms before the start of the pulse 
opsin_resp_max_delay_ms = (opsin_max_resp_idx - LED_on) / sampling_rate

####### determine max current and voltage  response value 
current_max_LED = list(map(max, current_data_LED)) #get list of all current max values per pulse (need to be min since this is VC)
voltage_max_LED = list(map(max, voltage_data_LED)) #get list of all voltage max values per pulse
current_max_LED = list(map(abs, current_max_LED))
voltage_baseline_LED = voltage_data_baseline
voltage_deflection_LED = voltage_max_LED - voltage_baseline_LED
stim_type_LED = 'LED_pulse'
 

####  extract number and max voltage deflection of subthreshold event or spikes for each LED stim.

## set up empty arrays that will be populated with for loop below 
response_type_LED_all = []
spike_count_LED_all = []
subthresh_event_LED_all = []
voltage_max_LED_all = []
V_resp_delay_LED_all = []
V_deflection_LED_all = []
spike_freq_LED_all = []    


## loop that 1st counts number of pulses applied, then iterates and extracts information about peak of response and based on this categorises
## subthreshold or spikes and extracts their peak responses and timing etc 

for counter, (voltage, V_max, resp_time) in enumerate ( zip (voltage_data_LED, voltage_max_LED, opsin_max_resp_idx), start = 1): 
    
    spike_LED = find_peaks(voltage, height=-30) #find peaks with height over 0mV in array 0 from voltage_data 

    if spike_LED [0].size ==0: ## no spike is detected for a given current pulse create subthreshold response arrays 
        print('LED stim ' +str (counter) + ' gave rise to subthreshold event\n')
        response_type_LED = 'sub_thresh'
        spike_count_LED = 0
        subthresh_event_LED = 1
        voltage_max_LED = V_max
        voltage_max_LED_idx = resp_time
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED = (voltage_max_LED_idx - LED_pulse_start_idx) / sampling_rate
        V_deflection_LED = voltage_max_LED - voltage_baseline_LED 
        response_type_LED = 'sub_thresh_event'
        spike_freq_LED  = np.nan
        
        response_type_LED_all.append(response_type_LED )
        spike_count_LED_all.append(spike_count_LED)
        subthresh_event_LED_all.append(subthresh_event_LED)
        voltage_max_LED_all.append(voltage_max_LED)
        V_resp_delay_LED_all.append(V_resp_delay_LED)
        V_deflection_LED_all.append(V_deflection_LED)
        spike_freq_LED_all.append(spike_freq_LED)
    
    else: ## is spike detected extract values 
        print('LED stim ' +str (counter) + ' gave rise to Spike(s)\n')
        response_type_LED_1 = 'Spike'
        spike_count_LED = len(spike_LED[0])
        subthresh_event_LED = 0
        voltage_max_LED = spike_LED[1]
        voltage_max_LED_tup = sum(voltage_max_LED.items(), ())
        voltage_max_LED_arr = voltage_max_LED_tup[1]
        voltage_max_LED = voltage_max_LED_arr[0].tolist() ## extract peak spike value for 1st spike
        voltage_max_LED_idx = spike_LED[0] ###
        LED_pulse_start_idx = 1999  ## because I added 100ms worth of data before each pulse 
        V_resp_delay_LED = (voltage_max_LED_idx - LED_pulse_start_idx) / sampling_rate
        V_resp_delay_LED =  V_resp_delay_LED[0].tolist() ## extract delay just to the 1st encoutered spike 
        V_deflection_LED = voltage_max_LED - voltage_baseline_LED 
        response_type_LED = 'Spike'
        spike_freq_LED = get_spike_frequency (spike_LED)

        response_type_LED_all.append(response_type_LED )
        spike_count_LED_all.append(spike_count_LED)
        subthresh_event_LED_all.append(subthresh_event_LED)
        voltage_max_LED_all.append(voltage_max_LED)
        V_resp_delay_LED_all.append(V_resp_delay_LED)
        V_deflection_LED_all.append(V_deflection_LED)
        spike_freq_LED_all.append(spike_freq_LED)


spike_count_total_LED_trace = sum(spike_count_LED_all)


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
                               'max_V_deflection_level_mV' : voltage_max_LED_all, 
                               'total_V_deflection_from_baseline_mV': V_deflection_LED_all ,
                               'time_to_peak_v_deflection_ms': V_resp_delay_LED_all, 
                               'voltage_points_plot': voltage_data_LED, 
                               'LED_points_plot': LED_data  })


trace_data_master = trace_data_LED.append(trace_data_I_inj, sort = False)

### save data 

### save individual file
data_final_df = trace_data_master ## date data_final array and transform into transposed dataframe
data_final_df.to_csv('Analysis_output/Single_trace_data/CC_excitatory/' + str(file_name) +'.csv', header = True) ## write file as individual csv file 

##### save data in master dataframe

"""
To make am empty dataframe with correctly labelled columns for this particular analysis: 

## make column list   
column_names = list(trace_data_master)     

## make emty dataframe + column list 
CC_excitatory_opsin_master = pd.DataFrame(columns = column_names) #transform into dataframe and use given index 

## save it as .csv
CC_excitatory_opsin_master.to_csv('Analysis_output/CC_excitatory_opsin_master.csv', header = True)
"""

##open master sheet with data 
CC_excitatory_opsin_master = pd.read_csv('Analysis_output/CC_excitatory_opsin_master.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
CC_excitatory_opsin_master = CC_excitatory_opsin_master.append(trace_data_master, sort = False) #adds row with new values to main dataframe

## save new version of updated dataframe as csv
CC_excitatory_opsin_master.to_csv('Analysis_output/CC_excitatory_opsin_master.csv', header = True)

    
#### plot figure of current injection + response 
    
if not voltage_data_I_injection_points:
    print('No I pulse to graph')

else:
    time_plot_I_injection = (np.arange(len(voltage_data_I_injection[0]))*abf.dataSecPerPoint) * 1000
    
    fig1 = plt.figure(figsize =(2,4))
    sub1 = plt.subplot(2,1,1)
    sub1.plot(time_plot_I_injection, voltage_data_I_injection[0], linewidth=1, color = '0.2')
    sub1.set_title('Current Injection Response', color = '0.2')
    sub1.tick_params(axis='x', colors='white')
    sub1.spines['bottom'].set_color('white')
    plt.setp(sub1.get_xticklabels(), visible = False)
    sns.despine()    
    
    sub2 = plt.subplot(2,1,2)
    sub2.plot(time_plot_I_injection, current_data_I_pulse[0], linewidth=0.5, color = '0.2')
    plt.xlabel('Time (ms)')
    sns.despine()
        

#### plot figure of LED stim + response 

## create arrays for sample data to be plotted  
LED_expand_idx_plot =[np.concatenate([ [x[0]-i-1 for i in reversed(range(199))], x, [x[-1]+i+1 for i in range(1999)] ]) for x in LED_idx_cons] ### add 5ms pre LED start and 100ms after .  
voltage_data_plot = [voltage_trace [i] for i in LED_expand_idx_plot]

time_points_plot = (np.arange(len(voltage_data_plot[0]))*abf.dataSecPerPoint) * 1000
time_points_plot = [time_points_plot] * len(voltage_data_LED)



fig1 = plt.figure(figsize =(30,5))
fig1.subplots_adjust(wspace=0.2)

for counter, (voltage, time, power) in enumerate (zip (voltage_data_plot, time_points_plot, LED_power_pulse), start = 1): 
    sub = plt.subplot(2,len(voltage_data_LED),counter)
    markers_on = [199]
    sub.plot (time, voltage, '-om', markevery = markers_on, markerfacecolor="m", markeredgecolor = 'w', linewidth=0.3, color = '0.2')
    sub.tick_params (axis = 'x', colors='black') 
    sub.spines['bottom'].set_color('black')
    sub.set_title(power + ' mW/mm2', color = '0.2')
    plt.setp(sub.get_xticklabels(), visible = True)
    plt.xlabel('Time (ms)\n Magenta dot = LED ON')
    plt.ylabel('Photocurrent (pA)')
    plt.suptitle('Example opsin induced voltage deflections from this trace', fontsize=16)

    sns.despine()


print ('Total number of spikes detected in this trace: N = ' +str(spike_count_total_LED_trace ))
