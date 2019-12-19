"""
Created on Thu Mar  7 15:20:04 2019
@author: adna.dumitrescu

Script opens abf file with current clamp data acquired during a Gap Free recording 
during which pulses of 1, 2, 5 ms were applied at different steps in order to find 
minimum current neccesary to trigger spiking with each duration.  


How the script works: 
1. Asks user to provide a file path for the abf file to be analysed
2. Asks user for meta-data that cannot be automatically extracted from abf file (such as rig ID and opsin type)
3. Finds all points where a current pulse is applied and uses these time-bins to extract data from the trace in which voltage values are recorded.
4. Puts all extracted data in a single .csv file named after the trace
5. Adds data as a new row to a master .csv file for the whole experiment 
6. Outputs in line graphs showing the full trace and the selected minimum pulses and the spikes their produce for each duration tested. 

Metadata collected: file name, experimenter, opsin type, date
Resting membrane potential 


Metadata  collected:: file name, date, opsin type 
Data that is calculated by script: 
Resting membrane potential as average of all values during the 1st second of the trace. 
Script looks for every current pulse applied and categorises them based on pulse duration as 1, 2, 5ms pulses
For each category the script will find the minimum current step at which a spike is detected and will output:
Spike number for this pulse
Current pulse value in pA
Max depolarisation level attained by AP in mV
Time to spike from begining of pulse. 
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

### select experimenter 
user_input = int(input('Which rig was used for this recording: Rig 1 = 1   Rig 2 = 2\n'))

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


### extract main data
data = abf.data
voltage_trace = data[0,:] # extracts primary channel recording in CC which is voltage measurement 
current_trace = data[1,:] # extracts 2ndary channel recording in CC which is current measurement 
date_time = abf.abfDateTime
protocol = abf.protocol
time = abf.sweepX
resting_potential = np.mean(voltage_trace [0:20000]) # takes mean of all values aquired in the 1st second which is used as baseline membrane resting potential 
sampling_rate = abf.dataPointsPerMs
sampling_rate_2 = abf.dataSecPerPoint
file_name = abf.abfID ### extract filename 
protocol = abf.protocol


#### baseline substraction of current 
current_baseline =  np.mean(current_trace [0:20000])
current_base_substract = current_trace - current_baseline

############### first processing of values

### find index values where current injection is applied 
current_injection_idx = (np.where(current_trace > 10)) # index of values in current trace where the current injected in more then 10pA
current_injection_idx_array = np.asarray(current_injection_idx) #transform into an array for next step

## function to find consecutive elements
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

current_injection_idx_cons = consecutive(current_injection_idx_array[0]) #find consecutive indexes where current over 10pa is applied and split into separate arrays --> each array would be 1 pulse 
current_injection_idx_cons_df = pd.DataFrame(current_injection_idx_cons) #transform data into dataframe for easier processing below 

### determine lenght of current pulse
current_pulse_length = current_injection_idx_cons_df.count(axis = 'columns') #count the number of elements in each row so that you can extract the length of each pulse (e.g 20 elements = 1ms)
current_pulse_length_final = round(current_pulse_length / sampling_rate) # transform number of elements into actual ms rounded 

### increase array to add 2ms before and after current pulse to collect more voltage data
current_pulses_expanded =[np.concatenate([ [x[0]-i-1 for i in reversed(range(39))], x, [x[-1]+i+1 for i in range(39)] ]) for x in current_injection_idx_cons]

### use selected indices to extract current and voltage data
current_data = [current_base_substract [i] for i in current_pulses_expanded] # use index extracted for each individual pulse to extract current values 
current_data_df = pd.DataFrame(current_data) #transform current_data into a data frame
voltage_data = [voltage_trace [i] for i in current_pulses_expanded]  # use index extracted for each individual pulse to extract voltage values 

### determine lenght of current pulse, pA value and max voltage response 
current_max = list(map(max, current_data)) #get list of all current max values per pulse
voltage_max = list(map(max, voltage_data))  #get list of all voltage max values per pulse

###putting all data together to extract final response values
pulse_data = pd.DataFrame({'Pulse_Length': current_pulse_length_final, 'Current_Value' : current_max, 'Voltage_Value' : voltage_max})

###extracting mim current injection for each duration plus corresponding AP peak V value
pulse_data_1ms_all = pulse_data.query('Pulse_Length ==1 & Voltage_Value > 0') # get all values where a 1ms pulse results in a V value over 0 (spike!)
pulse_data_2ms_all = pulse_data.query('Pulse_Length ==2 & Voltage_Value > 0') # get all values where a 2ms pulse results in a V value over 0 (spike!)
pulse_data_5ms_all = pulse_data.query('Pulse_Length ==5 & Voltage_Value > 0') # get all values where a 5ms pulse results in a V value over 0 (spike!)

########## series of if else statements to determine if a pulse of a certain length has been given to then extract the minimum value for which a spike was produced, if not then to create an empty series 

if pulse_data_1ms_all.empty:
    pulse_data_1ms_min = pd.Series([1, np.nan, np.nan])
    pulse_data_1ms_min.index = ['Pulse_Length_1ms', 'Current_Value_1ms', 'Voltage_Value_1ms']
    pulse_1ms_voltage_points = np.repeat(0,100) #make empy list of values otherwise you cannot add all data together in a dataframe
    pulse_1ms_current_points = np.repeat(0,100) #make empy list of values 
    spike_delay_start_I_1ms = np.nan
    spike_delay_end_I_1ms= np.nan
    spikes_total_1ms = 0
    print ("\nNo 1ms pulses applied in this trace\n\n")

else:
    pulse_data_1ms_min = pulse_data_1ms_all.loc [pulse_data_1ms_all['Current_Value'].idxmin()]
    pulse_data_1ms_min.index = ['Pulse_Length_1ms', 'Current_Value_1ms', 'Voltage_Value_1ms']
    pulse_1ms_idx = pulse_data_1ms_min.name #use name of series to extract index of data chosen from the large voltage or current_data dataframes
    pulse_1ms_voltage_points = voltage_data[pulse_1ms_idx] #extract all corresponding voltage values from which the pulse 1ms was chosen 
    pulse_1ms_current_points = current_data[pulse_1ms_idx ]#extract all corresponding current values from which the pulse 1ms was chosen based on index
    delay_beg_I_pulse_2ms_spike = np.nan
    delay_end_I_pulse_2ms_spike = np.nan
    
    ##1ms pulse data delay between start and end of current step and AP spike
    I_pulse_max_idx_1ms = np.where(pulse_1ms_current_points > 10) #get all indexes with values over 10pa
    I_pulse_max_idx_1ms_start =  I_pulse_max_idx_1ms[0][0] #get index where the first large current value occurs i.e. start of pulse 
    I_pulse_max_idx_1ms_end = I_pulse_max_idx_1ms[0][-1]#get index where the last large current value max value occurs i.e. end of pulse
    V_pulse_max_idx_1ms = np.where(pulse_1ms_voltage_points == np.amax(pulse_1ms_voltage_points))[0][0] #extract index of max voltage value
    spike_delay_start_I_1ms = (V_pulse_max_idx_1ms - I_pulse_max_idx_1ms_start) / sampling_rate #value in ms of delay between start of current pulse and spike max 
    spike_delay_end_I_1ms = (V_pulse_max_idx_1ms - I_pulse_max_idx_1ms_end) / sampling_rate #value in ms of delay between end of current pulse and spike max 

    peaks1ms, _ = find_peaks(pulse_1ms_voltage_points, height=0) #find peaks with height over 0mV
    spikes_total_1ms = len(peaks1ms)### total number of spikes counted - remember to add to final data list 
 
    print ('1ms current pulse data\nAP current input threshold = ' + str(int(pulse_data_1ms_min.loc['Current_Value_1ms'])) + 'pA.\nAP max height =  ' + str(int(pulse_data_1ms_min.loc['Voltage_Value_1ms'])) + 'mV\nSpike delay of: ' + str(int(spike_delay_start_I_1ms)) + 'ms between the begining, and ' + str(int(spike_delay_end_I_1ms))+ 'ms between the end of the current pulse\n\n')

if pulse_data_2ms_all.empty:
    pulse_data_2ms_min = pd.Series([2, np.nan, np.nan])
    pulse_data_2ms_min.index = ['Pulse_Length_2ms', 'Current_Value_2ms', 'Voltage_Value_2ms']
    pulse_2ms_voltage_points = np.repeat(0,100)
    pulse_2ms_current_points = np.repeat(0,100)
    spike_delay_start_I_2ms = np.nan
    spike_delay_end_I_2ms = np.nan
    spikes_total_2ms = 0
    print ("No 2ms pulses applied in this trace\n\n")

else:
   pulse_data_2ms_min = pulse_data_2ms_all.loc [pulse_data_2ms_all['Current_Value'].idxmin()]
   pulse_data_2ms_min.index = ['Pulse_Length_2ms', 'Current_Value_2ms', 'Voltage_Value_2ms']
   pulse_2ms_idx = pulse_data_2ms_min.name
   pulse_2ms_voltage_points = voltage_data[pulse_2ms_idx]
   pulse_2ms_current_points = current_data[pulse_2ms_idx]
   
   ##2ms pulse data delay 
   I_pulse_max_idx_2ms = np.where(pulse_2ms_current_points > 10) #get all indexes with values over 10pa
   I_pulse_max_idx_2ms_start =  I_pulse_max_idx_2ms[0][0] #get index where the first large current value occurs i.e. start of pulse 
   I_pulse_max_idx_2ms_end = I_pulse_max_idx_2ms[0][-1]#get index where the last large current value max value occurs i.e. end of pulse
   V_pulse_max_idx_2ms = np.where(pulse_2ms_voltage_points == np.amax(pulse_2ms_voltage_points))[0][0] #extract index of max voltage value
   spike_delay_start_I_2ms = (V_pulse_max_idx_2ms - I_pulse_max_idx_2ms_start) / sampling_rate #value in ms of delay between start of current pulse and spike max 
   spike_delay_end_I_2ms = (V_pulse_max_idx_2ms - I_pulse_max_idx_2ms_end) / sampling_rate #value in ms of delay between end of current pulse and spike max 

   peaks2ms, _ = find_peaks(pulse_2ms_voltage_points, height=0) #find peaks with height over 0mV
   spikes_total_2ms = len(peaks2ms)### total number of spikes counted - remember to add to final data list 

   print ('2ms current pulse data\nAP current input threshold = ' + str(int(pulse_data_2ms_min.loc['Current_Value_2ms'])) + 'pA.\nAP max height =  ' + str(int(pulse_data_2ms_min.loc['Voltage_Value_2ms'])) + 'mV\nSpike delay of: ' + str(int(spike_delay_start_I_2ms)) + 'ms between the begining, and ' + str(int(spike_delay_end_I_2ms))+ 'ms between the end of the current pulse\n\n')

if pulse_data_5ms_all.empty:
    pulse_data_5ms_min = pd.Series([5, np.nan, np.nan])
    pulse_data_5ms_min.index = ['Pulse_Length_5ms', 'Current_Value_5ms', 'Voltage_Value_5ms']
    pulse_5ms_voltage_points = np.repeat(0,100)
    pulse_5ms_current_points = np.repeat(0,100)
    spike_delay_start_I_5ms = np.nan
    spike_delay_end_I_5ms= np.nan
    spikes_total_5ms = 0
    print ("No 5ms pulses applied in this trace\n\n")

else:
   pulse_data_5ms_min = pulse_data_5ms_all.loc [pulse_data_5ms_all['Current_Value'].idxmin()]
   pulse_data_5ms_min.index = ['Pulse_Length_5ms', 'Current_Value_5ms', 'Voltage_Value_5ms']
   pulse_5ms_idx = pulse_data_5ms_min.name
   pulse_5ms_voltage_points = voltage_data[pulse_5ms_idx]
   pulse_5ms_current_points = current_data[pulse_5ms_idx]
   
   ###5ms pulse  data delay ### fix this so that the extracted delay values are simple floats
   I_pulse_max_idx_5ms = np.where(pulse_5ms_current_points > 10) #get all indexes with values over 10pa
   I_pulse_max_idx_5ms_start =  I_pulse_max_idx_5ms[0][0] #get index where the first large current value occurs i.e. start of pulse 
   I_pulse_max_idx_5ms_end = I_pulse_max_idx_5ms[0][-1]#get index where the last large current value max value occurs i.e. end of pulse
   V_pulse_max_idx_5ms = np.where(pulse_5ms_voltage_points == np.amax(pulse_5ms_voltage_points))[0][0] #extract index of max voltage value
   spike_delay_start_I_5ms_test = (V_pulse_max_idx_5ms - I_pulse_max_idx_5ms_start) / sampling_rate #value in ms of delay between start of current pulse and spike max 
   spike_delay_start_I_5ms = (V_pulse_max_idx_5ms - I_pulse_max_idx_5ms_start) / sampling_rate #value in ms of delay between start of current pulse and spike max 
   spike_delay_end_I_5ms = (V_pulse_max_idx_5ms - I_pulse_max_idx_5ms_end) / sampling_rate #value in ms of delay between end of current pulse and spike max 

   peaks5ms, _ = find_peaks(pulse_5ms_voltage_points, height=0) #find peaks with height over 0mV
   spikes_total_5ms = len(peaks5ms)### total number of spikes counted - remember to add to final data list 

   print ('5ms current pulse data\nAP current input threshold = ' + str(pulse_data_5ms_min.loc['Current_Value_5ms']) + 'pA.\nAP max height =  ' + str(pulse_data_5ms_min.loc['Voltage_Value_5ms']) + 'mV\nSpike delay of: ' + str(int(spike_delay_start_I_5ms)) + 'ms between the begining, and ' + str(int(spike_delay_end_I_5ms))+ 'ms between the end of the current pulse')


####### plot data for visual check up 
    
### plot full current and voltage trace
fig1 = plt.figure(figsize =(15,5))

sub1 = plt.subplot(211, )
sub1.plot(time, voltage_trace, linewidth=0.5, color = '0.2')
plt.ylim(-80,60) #for y axis
plt.xlim(0,) #for x axiss
plt.ylabel('mV')
sub1.spines['left'].set_color('0.2')
sub1.spines['bottom'].set_color('white')
sub1.tick_params(axis='y', colors='0.2')
sub1.tick_params(axis='x', colors='white')
plt.setp(sub1.get_xticklabels(), visible = False)
sns.despine()

sub2 = plt.subplot(212, sharex=sub1)
plt.plot(time, current_base_substract, linewidth=0.5, color = '0.2')
plt.ylim(-10,400) #for y axis
plt.xlim(0,) #for x axis
plt.xlabel('time (s)')
plt.ylabel('pA')
sub2.spines['left'].set_color('0.2')
sub2.spines['bottom'].set_color('white')
sub2.tick_params(axis='y', colors='0.2')
sub2.tick_params(axis='x', colors='0.2')
sns.despine()
plt.show()

### plot individual chosen spikes + corresponding current injection trace 
time1ms = (np.arange(len(pulse_1ms_voltage_points))*abf.dataSecPerPoint) * 1000
time2ms = (np.arange(len(pulse_2ms_voltage_points))*abf.dataSecPerPoint) * 1000
time5ms = (np.arange(len(pulse_5ms_voltage_points))*abf.dataSecPerPoint) * 1000

fig2 = plt.figure(figsize =(10,5))
sub1 = plt.subplot(231)
sub1.plot(time1ms, pulse_1ms_voltage_points, color = '0.8')
sub1.set_title('1ms Pulse', color = '0.8')
sub2.tick_params(axis='x', colors='white')
sub1.spines['bottom'].set_color('white')
plt.setp(sub1.get_xticklabels(), visible = False)
sns.despine()

sub2 = plt.subplot(232)
sub2.plot(time2ms, pulse_2ms_voltage_points, color = '0.6')
sub2.set_title('2ms Pulse', color = '0.4')
sub2.tick_params(axis='x', colors='white')
sub2.spines['bottom'].set_color('white')
plt.setp(sub2.get_xticklabels(), visible = False)
sns.despine()

sub3 = plt.subplot(233)
sub3.plot(time5ms, pulse_5ms_voltage_points, color = '0.4')
sub3.set_title('5ms Pulse', color = '0.4')
sub3.tick_params(axis='x', colors='white')
sub3.spines['bottom'].set_color('white')
plt.setp(sub3.get_xticklabels(), visible = False)
sns.despine()

sub4 = plt.subplot(234)
sub4.plot(time1ms,pulse_1ms_current_points, color = '0.8')
plt.xlabel('Time (ms)')
sns.despine()

sub5 = plt.subplot(235)
sub5.plot(time2ms, pulse_2ms_current_points, color = '0.6')
plt.xlabel('Time (ms)')

sub6 = plt.subplot(236)
sub6.plot(time5ms,pulse_5ms_current_points, color = '0.4')
plt.xlabel('Time (ms)')
sns.despine()


########## putting all data that needs to be extracted together 
data_final =[file_name, experimenter, cell_type_selected, date_time, resting_potential, spikes_total_5ms, pulse_data_5ms_min.loc['Current_Value_5ms'], pulse_data_5ms_min.loc['Voltage_Value_5ms'], spike_delay_start_I_5ms, spikes_total_2ms, pulse_data_2ms_min.loc['Current_Value_2ms'], pulse_data_2ms_min.loc['Voltage_Value_2ms'], spike_delay_start_I_2ms, spikes_total_1ms, pulse_data_1ms_min.loc['Current_Value_1ms'], pulse_data_1ms_min.loc['Voltage_Value_1ms'], spike_delay_start_I_1ms]#make list with all values

##### saving data 

#######save data as individual csv file 
data_final_df = pd.DataFrame(data_final).T ## date data_final array and transform into transposed dataframe
data_final_df.to_csv('Analysis_output/Single_Trace_data/Gapfree_AP_stim/' + str(file_name) +'.csv', header = True) ## write file as individual csv file 

"""
To make am empty dataframe with correctly labelled columns for this particular analysis: 

### name colums for data     
Gapfree_AP_stim_columns = ['Trace_Number',
                           'Experimenter',
                           'Cell_Type', 
                           'Date_Time',  
                           'Resting_Potential_mV', 
                           'Spike_count_5ms', 
                           'AP_thresh_pA_5ms', 
                           'AP_max_mV_5ms', 
                           'Spike_delay_start_I_5ms', 
                           'Spike_count_2ms', 
                           'AP_thresh_pA_2ms', 
                           'AP_max_mV_2ms', 
                           'Spike_delay_start_I_2ms', 
                           'Spike_count_1ms', 
                           'AP_thresh_pA_1ms', 
                           'AP_max_mV_1ms', 
                           'Spike_delay_start_I_1ms'] 

## make emty dataframe + column list 
Gapfree_AP_stim = pd.DataFrame(columns = Gapfree_AP_stim_columns) #transform into Series and use given index 

## save it as .csv
Gapfree_AP_stim.to_csv('Analysis_output/Gapfree_AP_stim.csv', header = True)
"""

##open master sheet with data 
Gapfree_AP_stim = pd.read_csv('Analysis_output/Gapfree_AP_stim.csv', index_col = 0) 

### add data extracted here as a new row in opened dataframe
Gapfree_AP_stim.loc[len(Gapfree_AP_stim)] = data_final #adds row with new values to main dataframe

## save new version of updated dataframe as csv
Gapfree_AP_stim.to_csv('Analysis_output/Gapfree_AP_stim.csv', header = True)



