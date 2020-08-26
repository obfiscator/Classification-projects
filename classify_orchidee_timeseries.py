# The goal of this script is to take a map of output from ORCHIDEE and
# classify timeseries on every pixel, producing a single map that
# shows how different pixels are classified as well as example timeseries
# for each classification.
#
# 1) Extract the variables of interest (can be multiple)
# 2) Extract the timeseries of interest (only a single one..selected based on all variables extracted)
# 3) Establish classification criteria for the timeseries of interest
# 4) Classify the timeseries of interest for all pixels
# 5) Create sample plots of various classifications
# 6) Create the map showing the classification of each pixel

# I try to do all the above in different subroutines

###############################################
# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys
import argparse
import re

# Import from local routines
from grid import Grid
from extraction_subroutines import extract_variables_from_file,extract_timeseries,read_in_extracted_timeseries
from classification_subroutines import create_classification_criteria,classify_observations,plot_classified_observations,map_classified_pixels,extract_and_calculate_classifiction_metrics,simulation_parameters
from netcdf_subroutines import find_variable
###############################################


###############################################
# Some possible flags to control the simulation
parser = argparse.ArgumentParser(description='Extract and classify timeseries from a series of ORCHIDEE history files.')
parser.add_argument('--pft', dest='pft_selected', action='store',
                   default=6,type=int,
                   help='the PFT for which the timeseries are extracted (default: 6)')
parser.add_argument('--vmt', dest='veget_max_threshold', action='store',
                   default=0.01, type=float,
                   help='below this vegetation fraction, ignore the pixel, assuming it\'s on the edge of its range.  0.01 removed about 9% of the PFT 6, 0.02 removed less than 20%.')
parser.add_argument('--classification', dest='timeseries_flag', action='store',
                   default="LAI_MEAN1", 
                   help='dictates how we classify the timeseries.  LAI_MEAN1 makes judgements just based on LAI_MEAN.  LAI_MEAN2 makes judgements based on LAI_MEAN and IND.')
parser.add_argument('--input_file_string', dest='input_file_string', action='store',required=True,
                   help='the ORCHIDEE history files where we get the data from, in a format like \"FG1.PFTCHECK.r6811_,_1Y_stomate_history.nc\".  I assume that a string like YYYYMMDD_YYYYMMDD appears where the comma is.')

parser.add_argument('--do_test', dest='do_test', action='store',default=False,
                   help='if TRUE, we will limit the number of timeseries we extract to ntest_points.  This makes the process faster, and is good for checking the format of output plots.')
parser.add_argument('--data_years', dest='data_years', action='store',
                   default="1901,2241",
                   help='the years of data to analyze. If a file with these data years is not found, we attempt to create one (which takes longer).')
parser.add_argument('--plot_years', dest='plot_years', action='store',
                   default="101,150",
                   help='the years of the timeseries to plot in the example timeseries.  101,150 is good for forests since they are mature, but croplands and grasslands may be different.   If you plot more than 50 years with annual data, it can be hard to see behavior.')
parser.add_argument('--global_operation', dest='global_operation', action='store',
                   default="sum", choices=['sum','ave'],
                   help='how to combine the pixels for plotting the global timeseries.  Choices are sum and ave.')
parser.add_argument('--year_increment', dest='year_increment', action='store',
                   default=1, type=int,
                   help='the number of years each of our history files covers (default 1, for a TEST run...10 is common for PROD runs)')
parser.add_argument('--force_annual', dest='force_annual', action='store',
                   default="False", 
                   help='if True/Yes/y/Y, then attempt to change the time axis to annual if it is not already.')
parser.add_argument('--fix_time_axis', dest='fix_time_axis', action='store',
                   default="False", 
                   help='if True/Yes/y/Y, then attempt to create a time axis that goes from the starting date to the ending date.  Used in the case that the existing time axis is incorrect (a subroutine checks to make sure the existing time axis is always increasing in value, and if not, stops the analysis)')

parser.add_argument('--extract_only', dest='extract_only', action='store',
                   default="False", 
                   help='if True/Yes/y/Y, then stop the code after the variables have been extracted (do not classify them or make a map)')


args = parser.parse_args()

print("######################### INPUT VALUES #########################")
pft_selected=args.pft_selected
print("Targetting PFT number {} (use the --pft flag to change)".format(pft_selected))
pft_selected=pft_selected-1 # Accounting for Python arrays starting from 0

veget_max_threshold=args.veget_max_threshold
print("Using a veget_max cutoff of {} (use the --vmt flag to change)".format(veget_max_threshold))

timeseries_flag=args.timeseries_flag
if timeseries_flag == "LAI_MEAN1":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN information. (use the --classification flag to change)")
elif timeseries_flag == "LAI_MEAN2":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN and IND information. (use the --classification flag to change)")
elif timeseries_flag == "LAI_MEAN_BIMODAL":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN.  Flagging bimodal distributions (use the --classification flag to change)")
elif timeseries_flag == "TWBR":
    print("Extracting the TWBR timeseries for every pixel, and classifying only using it.  Checking for poor water conservation (use the --classification flag to change)")

else:
    print("Do not understand what classification you want me to do. (unknown --classification flag value {}".format(timeseries_flag))
    sys.exit(1)
#endif

input_file_string=args.input_file_string
match = re.search(r"(.+)\,", input_file_string)
if not match:
    print("Could not parse this input_file_string start!")
else:
    input_file_name_start=match[1]
#endif
match = re.search(r"\,(.+)", input_file_string)
if not match:
    print("Could not parse this input_file_string end!")
else:
    input_file_name_end=match[1]
#endif

data_years=args.data_years
match = re.search(r"(\d+)\,", data_years)
if not match:
    print("Could not parse this data_years string start!")
else:
    syear_data=int(match[1])
#endif
match = re.search(r"\,(\d+)", data_years)
if not match:
    print("Could not parse this data_years string end!")
else:
    eyear_data=int(match[1])
#endif

plot_years=args.plot_years
match = re.search(r"(\d+)\,", plot_years)
if not match:
    print("Could not parse this plot_years string start!")
else:
    syear_plot=int(match[1])
#endif
match = re.search(r"\,(\d+)", plot_years)
if not match:
    print("Could not parse this plot_years string end!")
else:
    eyear_plot=int(match[1])
#endif

possible_true_values=["true","t","yes","y"]
do_test=args.do_test
if do_test.lower() in possible_true_values:
    do_test=True
else:
    do_test=False
#endif
if do_test:
    print("Running a shortened TEST RUN. (use the --do_test flag to change)")
else:
    print("Running a full run. (use the --do_test flag to change)")
#endif

force_annual=args.force_annual
if force_annual.lower() in possible_true_values:
    force_annual=True
else:
    force_annual=False
#endif
if force_annual:
    print("Attempting to regrid the time axis to annual. (use the --force_annual flag to change)")
else:
    print("Leaving the time axis alone. (use the --force_annual flag to change)")
#endif

fix_time_axis=args.fix_time_axis
if fix_time_axis.lower() in possible_true_values:
    fix_time_axis=True
else:
    fix_time_axis=False
#endif
if fix_time_axis:
    print("Attempting to overwrite the existing time axis. (use the --fix_time_axis flag to change)")
else:
    print("Leaving the time axis alone. (use the --fix_time_axis flag to change)")
#endif

extract_only=args.extract_only
if extract_only.lower() in possible_true_values:
    extract_only=True
else:
    extract_only=False
#endif
if extract_only:
    print("Stopping the script after the variables have been extracted from the ORCHIDEE data files. (use the --extract_only flag to change)")
else:
    print("Running a full extraction and classification. (use the --extract_only flag to change)")
#endif

global_operation=args.global_operation
if global_operation == "sum":
    print("Summing together all the pixels to create a global timeseries (use the --global_operation flag to change)")
else:
    print("Not yet ready for this global operation! ",global_operation)
    sys.exit(1)
#endif

year_increment=args.year_increment
print("{} years between our history files (use the --year_increment flag to change)".format(year_increment))


print("######################## END INPUT VALUES ######################")

# To make things easier, pass around a stucture with simulation parameters.
sim_parms=simulation_parameters(pft_selected,veget_max_threshold,timeseries_flag,do_test,global_operation,force_annual,fix_time_axis)

###############################################


###############################################
# Extract the variables of interest from the .nc files
# I assume that a bunch of files are in a directory
# FG1.PFTCHECK.r6811_21100101_21101231_1Y_stomate_history.nc
# I tried to use ncrcat for this, but ORCHIDEE uses the variable
# time_counter instead of time, so it didn't work for me.
# This creates a new .nc file that I can read in later. 
sim_parms.set_extraction_information(syear_data,eyear_data,year_increment,input_file_name_start,input_file_name_end)

# If the data file exists, we don't have to create it.
try:
    srcnc = NetCDFFile(sim_parms.condensed_nc_file_name,"r")
    srcnc.close()
    print("Data file already exists: ",sim_parms.condensed_nc_file_name)
except:
    print("Need to create file: ",sim_parms.condensed_nc_file_name)
    extract_variables_from_file(sim_parms)
#endtry

if extract_only:
    print("Finishing script after variable extraction.")
    sys.exit(0)
#endif

# From here, I want to check some of the variable names.
srcnc = NetCDFFile(sim_parms.condensed_nc_file_name,"r")
veget_max_name=find_variable(["VEGET_MAX","VEGET_COV_MAX"],srcnc,False,"",lcheck_units=False)
lai_mean_name=find_variable(["LAI","LAI_MEAN"],srcnc,False,"",lcheck_units=False)
lai_max_name=find_variable(["LAI","LAI_MAX"],srcnc,False,"",lcheck_units=False)
sim_parms.set_variable_names(veget_max_name,lai_mean_name,lai_max_name)
srcnc.close()

##########################################################

###############################################################
############### combine the next two steps, since our #########
############### final timeseries may depend on others #########
###############################################################

##########################################################
# Now extract the timeseries from the newly created output file
# This creates an array of observations and an array of
# latitude/longitude to identify each of the observations.
# timeseries_array[i,time] where i is the observation
# timeseries_lat[i]
# timeseries_lon[i]
## This takes a couple minutes
#timeseries_array,timeseries_lat,timeseries_lon=extract_timeseries(output_file_name,timeseries_variable,pft_selected,veget_max_threshold)
# This uses output printed by the above routine
#timeseries_array,timeseries_lat,timeseries_lon=read_in_extracted_timeseries()
##########################################################
##########################################################
# For each timeseries, create a classification vector.  Each point
# in this vector consists of some test: number of NaNs, maximum value, etc.
# classification_array[i,time]
#classification_array=create_classification_criteria(timeseries_array,timeseries_variable,pft_selected)
##########################################################

##########################################################
# Now extract the timeseries from the newly created output file
# and create classification metrics, taking into account information from any
# other timeseries. We aim to identify good, ok, and bad points.
# This creates an array of observations, an array of
# latitude/longitude to identify each of the observations, and an array
# to use in the classification with values of all metrics.
# timeseries_array[i,time] where i is the observation
# timeseries_lat[i]
# timeseries_lon[i]
# classification_array[i,time]
# timeseries variable
timeseries_array,timeseries_lat,timeseries_lon, classification_array=extract_and_calculate_classifiction_metrics(sim_parms)


##########################################################
# Based on the classification vector for each timeseries, classify
# it.  We aim to identify good, ok, and bad points.
# classification_vector[i]
classification_vector=classify_observations(classification_array,sim_parms)
##########################################################

##########################################################
# Generate a few random plots of each classification.
level_colors,colors_by_level=plot_classified_observations(classification_vector,timeseries_array,timeseries_lat,timeseries_lon,sim_parms,syear_plot,eyear_plot)
##########################################################

##########################################################
# Plot a map of all classified pixels.
map_classified_pixels(classification_vector,timeseries_lat,timeseries_lon,level_colors,colors_by_level,sim_parms)
##########################################################


print("Finished with program.")
