#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:10:54 2019
@author: adna.dumitrescu

Script opens abf file with voltage clamp data aquired during excitatory opsin stimulation. 

How the script works: 
1. Asks user to provide a file path for the abf file to be analysed
2. Asks user for meta-data that cannot be automatically extracted from abf file (such as stimulation regime and cell type etc)
3. Finds all points where the LED is ON and uses these time-bins to extract data from the trace in which current values are recorded. 
4. Puts all extracted data in a single .csv file named after the trace
5. Adds data as a new row to a master .csv file for the whole experiment 
6. Outputs in line graphs showing cell response to LED stimulation

Metadata collected: file name, date, experimenter, protocol type, opsin type, LED wavelenght, power and stimulation time.
Data that is calculated by script: 
Baseline current injection level: mean of points across first 100ms of the trace 
Resting membrane voltage: mean of points across first 1000ms of the trace
Maximum photocurrent response per each LED stimulation: maximum value within a time period between LED onset plus 1000ms.
Photocurrent response activation time as time to peak photocurrent response from light onset in ms. 
Photocurrent response deactivation time in ms: monoexponential fit from max value found during last 5ms of light stimulation + 500 ms. This time-window was empirically determined to match the same calculation done in Clampfit.
Extracts raw current trace data points corresponding to LED stimulation
Extracts raw LED trace data points corresponding to LED stimulation

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyabf
import pandas as pd
from exponentialFitGetTau import exponentialFitGetTau

import os

wdir=os.getcwd() 


#from scipy.signal import find_peaks

#### open file 
file_path = input('Please give me the complete file path of the trace you want to analyse below:\n')
abf = pyabf.ABF(file_path)

### extract main data
data = abf.data
current_trace = data[0,:] # extracts primary channel recording in VC which is voltage measurement 
current_data_baseline = np.mean(current_trace [0:1999])### calculate baseline pA response by averaging the first 100ms
current_trace_baseline_substracted = current_trace - current_data_baseline
voltage_trace = data[1,:] # extracts 2ndary channel recording in VC which is current measurement 
voltage_data_baseline = np.mean(voltage_trace [0:1999])
LED_trace = data[-1,:] # extracts last channel recording in VC which is LED analog signal (channel 3 = LED TTL, not needed)
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

###### find index values where LED is ON use this to extract all other info from current trace
LED_idx = (np.where(LED_trace > 0)) # index of values in LED trace where V values are over 0
LED_array = np.asarray(LED_idx) #transform into an array for next step

## function to find consecutive elements
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

LED_idx_cons = consecutive(LED_array[0]) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim
LED_idx_cons_df = pd.DataFrame(LED_idx_cons) #transform data into dataframe for easier processing below 

### determine lenght of LED Stim
LED_time =pd.Series (LED_idx_cons_df.count(axis = 'columns'), name ='LED_time_ON' )#count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
LED_time = round(LED_time / sampling_rate) # transform number of elements into actual ms rounded 

### increase array to add 100ms before and 1s after current pulse to collect more current data
LED_expand_idx =[np.concatenate([ [x[0]-i-1 for i in reversed(range(1999))], x, [x[-1]+i+1 for i in range(19999)] ]) for x in LED_idx_cons]

###### use selected indices to extract current and voltage data
current_data = [ current_trace_baseline_substracted [i] for i in LED_expand_idx] # use index extracted for each individual pulse to extract current values 
current_data_df = pd.DataFrame(current_data) #transform current_data into a data frame
voltage_data = [voltage_trace [i] for i in LED_expand_idx]  # use index extracted for each individual pulse to extract voltage values 
LED_data = [LED_trace [i] for i in LED_expand_idx] 
LED_data_df = pd.DataFrame(LED_data)

####### determine max current response value 
current_max = list(map(min, current_data)) #get list of all current min values per pulse (need to be min since this is VC)
current_max = list(map(abs, current_max))

####### determine max LED analog pulse value
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


#### calculate delay between light on and peak of response

## find where current response starts 
current_data_arr = np.asarray(current_data).T ## transpose array of current data 
opsin_response = current_data_arr < (-10) ## get boolean array with TRUE values where there is a current response smaller then -10pA
opsin_response_df = pd.DataFrame(opsin_response) ## transform array into dataframe

opsin_start_all = pd.DataFrame (np.where(opsin_response_df ==True)).T ## find the start of opsin response by finding and ranking all TRUE values
opsin_start_single = opsin_start_all.drop_duplicates(1) ## drop duplicates so that you are left with the first value for every pulse 
opsin_start_single = opsin_start_single.set_index(1) ## index using column 1 which is the index used to count pulse number and which matches the rest of the document. 

LED_on = 1999 ## since we add 100ms before the start of the pulse 

opsin_resp_start_delay_ms = (opsin_start_single - LED_on ) / sampling_rate ## calculate how long it takes between LED on time and the opsin response which is calculated as the first value which is smaller then -10pA
opsin_max_resp_idx = current_data_df.idxmin (1) ### returns index 
current_data_df_temp = current_data_df.T
LED_max_idx = LED_data_df.idxmax(1)

opsin_resp_max_delay_ms = (opsin_max_resp_idx - LED_on) / sampling_rate


### extracting tau value for opsin off response 


deactivation_tau = []
for data_row in current_data:
    y = data_row
    x = np.linspace(1, len(y), len(y))
    tau = exponentialFitGetTau(x, y, 1, 2000)
    tau_LED_stim = tau / sampling_rate
    deactivation_tau.append(tau_LED_stim)
    print('Deactivation time constant for this photocurrent response is ' +str(round(tau_LED_stim,2)) + ' ms\n\n')


###putting all data together to extract response values per trace 
trace_data = pd.DataFrame({'trace_number':file_name ,
                           'date_time' : date_time,
                           'Experimenter': experimenter,
                           'protocol' : protocol,  
                           'cell_type': cell_type_selected, 
                           'I_level_baseline_pA': abs(current_data_baseline), 
                           'V_data_baseline':voltage_data_baseline,  
                           'LED_stim_wavelenght': LED_wavelength, 
                           'LED_time_ms': LED_time, 
                           'LED_power_mWmm': LED_power_pulse, 
                           'Max_photocurrent_pA' : current_max, 
                           'Activation_time_ms':opsin_resp_max_delay_ms, 
                           'Deactivation_time_ms': deactivation_tau, 
                           'Current_points_plot': current_data, 
                           'LED_points_plot': LED_data })


### save individual file
data_final_df = trace_data ## date data_final array and transform into transposed dataframe
data_final_df.to_csv('Analysis_output/Single_Trace_data/VC_excitatory/' + str(file_name) +'.csv', header = True) ## write file as individual csv file 


##### save data in master dataframe

"""
To make am empty dataframe with correctly labelled columns for this particular analysis: 

## make column list     
column_names = list(trace_data)     

## make emty dataframe + column list 
VC_excitatory_opsin_master = pd.DataFrame(columns = column_names) #transform into Series and use given index 
## save it as .csv
VC_excitatory_opsin_master.to_csv('Analysis_output/VC_excitatory_opsin_master.csv', header = True)
"""

##open master sheet with data 
VC_excitatory_opsin_master = pd.read_csv('Analysis_output/VC_excitatory_opsin_master.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
VC_excitatory_opsin_master = VC_excitatory_opsin_master.append(trace_data, sort = False) #adds row with new values to main dataframe

## save new version of updated dataframe as csv
VC_excitatory_opsin_master.to_csv('Analysis_output/VC_excitatory_opsin_master.csv', header = True)


#### plotting data

## create arrays for sample data to be plotted  
LED_expand_idx_plot =[np.concatenate([ [x[0]-i-1 for i in reversed(range(199))], x, [x[-1]+i+1 for i in range(1999)] ]) for x in LED_idx_cons] ### add 5ms pre LED start and 100ms after .  
current_data_plot = [ current_trace_baseline_substracted [i] for i in LED_expand_idx_plot]

time_points_plot = (np.arange(len(current_data_plot[0]))*abf.dataSecPerPoint) * 1000
time_points_plot = [time_points_plot] * len(current_data)

    
#### make figure with sample data 
    
fig1 = plt.figure(figsize =(30,5))
fig1.subplots_adjust(wspace=0.5)

for counter, (current, time, power) in enumerate (zip (current_data_plot, time_points_plot, LED_power_pulse), start = 1): 
    sub = plt.subplot(2,len(current_data),counter)
    markers_on = [199]
    sub.plot (time, current, '-om', markevery = markers_on, markerfacecolor="m", markeredgecolor = 'w', linewidth=0.3, color = '0.2')
    sub.tick_params (axis = 'x', colors='black') 
    sub.spines['bottom'].set_color('black')
    sub.set_title(power + ' mW/mm2', color = '0.2')
    plt.setp(sub.get_xticklabels(), visible = True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Photocurrent (pA)')
    plt.text (30, -10, 'Dot = LED ON', ha='center', color='magenta')
    plt.suptitle('Example opsin photocurrent responses from this trace', fontsize=16)

    sns.despine()
