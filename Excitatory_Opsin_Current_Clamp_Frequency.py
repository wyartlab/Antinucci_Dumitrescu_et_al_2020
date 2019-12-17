#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:02:36 2019
@author: adna.dumitrescu

Script opens abf file with current clamp data aquired during excitatory opsin stimulation performed at different frequencies (1-100Hz). 

How the script works: 
1. Asks user to provide a file path for the abf file to be analysed
2. Asks user for meta-data that cannot be automatically extracted from abf file (such as stimulation regime and cell type etc)
3. Finds all points where the LED is ON and uses these time-bins to extract data from the trace in which voltage values are recorded. 
4. Puts all extracted data in a single .csv file named after the trace
5. Adds data as a new row to a master .csv file for the whole experiment 
6. Outputs in line graph showing cell response to LED stimulation

Metadata collected: file name, date, experimenter, protocol type, opsin type, LED wavelenght, power, stimulation time and frequency.
Data that is calculated by script: 
Baseline membrane voltage: mean of points across first 1000ms of the trace
LED spike number target which is 1x number of LED pulses present in a train 
Cell spiking frequency response as instantaneous frequency of all spikes present during the train
Total spike number per trace
Spike amplitude in mV for all spikes 
Number of spikes per each LED stimulation in a train 
Time to first spike (in ms) per each LED stimulation in a train 
Spike jitter which is calculated as the standard deviation of the mean for all times to 1st spike across the stimulation train. 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyabf
import pandas as pd
from scipy.signal import find_peaks
import statistics

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
LED_data = data [-1,:]
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

'''
Part 1: find frequency of LED pulses and analog input value
'''
###### find index values where LED is ON use this to extract all other info from current trace
LED_idx = (np.where(LED_trace > 0.1)) # index of values in LED trace where V values are over 0
LED_idx_flat = [item for sublist in LED_idx for item in sublist]
LED_array = np.asarray(LED_idx) #transform into an array for next step
LED_idx_cons = consecutive(LED_array[0]) #find consecutive indexes where LED is ON and split into separate arrays --> each array would be 1 LED stim
LED_idx_cons_df = pd.DataFrame(LED_idx_cons) #transform data into dataframe for easier processing below 
LED_stim_no = len(LED_idx_cons)


### determine lenght of LED Stim
LED_time =pd.Series (LED_idx_cons_df.count(axis = 'columns'), name ='LED_time_ON' )#count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
LED_time = round(LED_time / sampling_rate) # transform number of elements into actual ms rounded 
LED_time = LED_time[0]

 
###### use selected LED indices to extract current and voltage data
LED_data = [LED_trace [i] for i in LED_idx_cons] 
LED_data_df = pd.DataFrame(LED_data)


####### determine power in mW/mm2 of max LED analog pulse V value
LED_max_V = list(map(max, LED_data)) ## find max LED pulse points 
threshold = len([1 for i in LED_max_V if i > 5])## count if there are any values over 5V in LED_max_V, means that scale was set up wrong and pulse values in V need to be introduced manually by user 

if threshold == 0: ## pulse values are below 5V
    LED_max_V_round = [round (i,1) for i in LED_max_V] ## round number to 2 decimals
    LED_max_V_round_int = [str(i) for i in  LED_max_V_round] ## transform number to int otherwise can't index into LED_stim_power_table
    LED_index_value = LED_max_V_round_int
else:
    print('Wrong scale detected for LED analog input. Please add LED steps in Volts below, and na for no pulse applied:\n')
    LED_max_V_user = [input('pulse_1:  \n'), input]
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


### calculate frequency 

##extract frequency tested
LED_frequency = float(input('What LED frequency did you test? enter number as float 1 = 1.0 \n'))


if isinstance(LED_frequency, float):
    print ('LED frequency challenge at ' +str(LED_frequency) + 'Hz' ) #print this is choice selected is in the dictionary 
else:
    raise ValueError ('Wrong number entered for LED_frequency, please run script again. No data was saved') #print this if choice selected in not in the opsin dictionary 


LED_spike_target = LED_stim_no


#### extract frequency response data

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


#### find all spikes
spikes = find_peaks(voltage_trace, height=-20) #find peaks with height over -20mV in array 0 from voltage_data 

spikes_total = len(spikes[0])
spikes_amplitude = spikes [1]

spikes_amplitude_tup = sum(spikes_amplitude.items(), ())
spikes_amplitude_arr = spikes_amplitude_tup[1]
spikes_amplitude = spikes_amplitude_arr.tolist() ## extract peak spike value for 1st spike


spike_freq = get_spike_frequency (spikes)

####

LED_expand_idx =[np.concatenate([x, [x[-1]+i+1 for i in range(500)] ]) for x in LED_idx_cons] ## add 10ms after light pulse to count spikes. This is the max time we can add since some traces are done with 100Hz stim. 


voltage_data_LED = [voltage_trace [i] for i in LED_expand_idx]
LED_pulses = range (len(LED_data))

spike_per_LED = []
spikes_extracted = []
spike_time_extracted = []
for i in range (len(LED_data)):
    peaks = find_peaks(voltage_data_LED[i], height=-30)
    spike_per_LED.append(peaks)
    for j in peaks:
        spike = len(j)
        spikes_extracted.append(spike)

spikes_extracted_clean = spikes_extracted 
del spikes_extracted_clean[1::2]

spike_per_LED_stim = spikes_extracted_clean

spike_time = [item [0] for item in spike_per_LED]
spike_time2 = []
for i in spike_time:
    if len(i)>0:
        spike_time2.append(i)
spike_time_fin = [item [0] for item in spike_time2]

spike_time_ms = [i / sampling_rate for i in spike_time_fin ]

try:
    spike_jitter = statistics.stdev(spike_time_ms)
except statistics.StatisticsError:
    spike_jitter = np.nan
    print("spike jitter calculation not possible since cell responded only to a single stimulation")
 

###putting all data together to extract response values per trace 

trace_data_LED = pd.DataFrame({'trace_number':file_name ,'date_time' : date_time, 'experimenter': experimenter, 'protocol' : protocol,  'cell_type': cell_type_selected, 'V_baseline': voltage_data_baseline, 'LED_wavelenght': LED_wavelength,   
'LED_time_ms': LED_time, 'LED_power_mWmm': LED_power_pulse, 'LED_freq_target': LED_frequency, 'LED_spike_no_target': LED_spike_target, 'LED_freq_response': spike_freq, 'Spike_no_total': spikes_total, 'Spikes_amplitude': [spikes_amplitude], 'Spike_per_LED_stim': [spike_per_LED_stim], '1st_spike_time_ms': [spike_time_ms], 'Spike_jitter': spike_jitter  })


### save data status 


### save individual file
data_final_df = trace_data_LED ## date data_final array and transform into transposed dataframe
data_final_df.to_csv('Analysis_output/Single_Trace_data/CC_excitatory_frequency/' + str(file_name) +'.csv', header = True) ## write file as individual csv file 

##### save data in master dataframe

"""
#To make am empty dataframe with correctly labelled columns for this particular analysis: 

## make column list   
column_names = list(trace_data_LED)     

## make emty dataframe + column list 
CC_excitatory_opsin_frequency = pd.DataFrame(columns = column_names) #transform into dataframe and use given index 
## save it as .csv
CC_excitatory_opsin_frequency.to_csv('Analysis_output/CC_excitatory_opsin_frequency.csv', header = True)
"""

##open master sheet with data 
CC_excitatory_opsin_frequency = pd.read_csv('Analysis_output/CC_excitatory_opsin_frequency.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
CC_excitatory_opsin_frequency = CC_excitatory_opsin_frequency.append(trace_data_LED, sort = False) #adds row with new values to main dataframe

## save new version of updated dataframe as csv
CC_excitatory_opsin_frequency.to_csv('Analysis_output/CC_excitatory_opsin_frequency.csv', header = True)


    

#### plot figure of LED stim + response 

##full trace 
fig1 = plt.figure(figsize =(15,5))## plot all raw data 
sub1 = plt.subplot(211, )
sub1.plot(time, voltage_trace, linewidth=0.5, color = '0.2')
plt.ylim(-80,50) #for y axis
plt.xlim(0.5,4) #for x axiss
plt.ylabel('pA')
sub1.spines['left'].set_color('0.2')
sub1.spines['bottom'].set_color('white')
sub1.tick_params(axis='y', colors='0.2')
sub1.tick_params(axis='x', colors='white')
plt.setp(sub1.get_xticklabels(), visible = False)
sns.despine()

sub2 = plt.subplot(212, sharex=sub1)
plt.plot(time, LED_trace, linewidth=0.5, color = '0.2')
plt.ylim(-0.5,5) #for y axis
plt.xlim(0.5,4) #for x axis
plt.xlabel('time (s)')
plt.ylabel('LED_V_input')
sub2.spines['left'].set_color('0.2')
sub2.spines['bottom'].set_color('white')
sub2.tick_params(axis='y', colors='0.2')
sub2.tick_params(axis='x', colors='0.2')
sns.despine()
plt.show()

