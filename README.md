# Analysis Scripts for Electrophysiology Data

<p>Here you will find a series of scripts and functions used to analyse the bulk of the electrophysiology data present in Antinucci, Dumitrescu et al 2020. Each script has an indepth README at the begining comprising a full description of the process of data extraction and analysis.

As an overview the following figures (both main and supplemental) from the paper contain data extracted with:\
**Fig 4:** Gapfree_AP_stim & Excitatory_Opsin_Voltage_Clamp\
**Fig 5:** Excitatory_Opsin_Current_Clamp and Excitatory_Opsin_Current_Clamp_Frequency\
**Fig 8:** Inhibitory_Opsin_Voltage_Clamp\
**Fig 9:** Inhibitory_Opsin_Current_Clamp, Inhibitory_Opsin_CC_Short_AP_Inhibit, Inhibitory_Opsin_CC_Long_AP_Inhibit
add some random text 
### Folder organisation:
All the scripts except Gapfree_AP_stim are dependant on the functions: *exponentialFitGetTau* & *exponentialFitGetTauInhibitory* (details within scripts) and the 2 excel files *Rig_1_LED_power* & *Rig_2_LED_power*. For this reason these 4 files need to be in same location as the python analysis scripts.\
The *Sample_data* folder contains several .abf files that can be used to test every script. The details of each .abf file and which script it can be used with are in the *Sample_data_info* excel sheet.\
The *Analysis_output* folder contains the data extracted organised per script type. For the analysis to run correctly, this folder also needs to be present in the same location as the .py files

### Data Analysis:

#### 1 Prepare python enviroment 
The scripts are using additional packages which need to be present. Please install them using your favourite method (pip, anaconda etc) if they are not already part of your instalation:\
numpy  https://numpy.org \
matplotlib https://matplotlib.org \
seaborn  https://seaborn.pydata.org \
pyabf  https://pypi.org/project/pyabf/ \
pandas https://pandas.pydata.org \
scipy https://www.scipy.org

#### 2 Select trace to analyse 
Open the *Sample_data_info* excel sheet which is in the *Sample_data* folder and select one trace for analysis. 
For example let's assume we want to analyse the trace *18n270027_1*. Reading the info sheet we can see that it should be analysed with the script: *Excitatory_Opsin_Voltage_Clamp.py* 

#### 3 Run pyton script 
If you run *Excitatory_Opsin_Voltage_Clamp.py* you will get some prompts that you need to respond to as part of the analysis. The information that matches each .abf file is in the *Sample_data_info* excel sheet. 

For trace *18n270027_1* analysed with *Excitatory_Opsin_Voltage_Clamp.py* you will first have to provide the full file path where the data is located. Paste in the python console the full path where trace *18n270027_1* is located. 

Next you will be asked to provide the ID of the rig used to collect data. For this trace enter 2.\
Next you will be able to select which opsin was tested. Enter the number associated with Chrimson.\
Next you will be able to select which wavelenght was used to stimulate cell. Enter the number associated with 630nm. \
Finally you will be asked to select the irradiance max range used to stimulate cell. Enter the number associated with option LED_630_100%. 

The data will be extracted, colated and you will also see a figure produced with key data once the script has finished running. 

#### 4 Check data analysis
Open Analysis_output folder and check that (1) the *VC_excitatory_opsin_master.csv* file has one new row containing your newly extracted data and (2) open *Single_Trace_data/VC_excitatory* folder and check that a .csv file named after your trace is present. 

To check that the analysis was run correctly you can open either the single or the master file containing the extracted data and check that the value of the 1st LED stimulation performed in trace *18n270027_1* is **7.44 mW/mm2** and that it resulted in a photocurrent response with a max amplitude of **223.234 pA** etc. 

#### 5 Problem reporting 
If you have problems running the example scripts with the example .abf files provided or you spot any mistakes please get in touch at *adna.siana@gmail.com*
