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

###########################################################################
# OUTPUT SUMMARY
# This script products a series of figures, primarily a map with pixels
# colored green, light green, yellow, orange, and red.  These colors represent
# a judgement on the quality of the timeseries for that pixel.  Green pixels
# represent behavior that we judge to be "good" from a scientific standpoint,
# for example a stable LAI timeseries over the course of a hundred years,
# reaching a value that represents a PFT in robust health.  Red pixels indicate
# pixels with problematic behavior, such as an LAI timeseries that falls to 
# zero in some years before jumping back.  Such a timeseries may be an 
# indication that phenology is not working (and the tree doesn't sprout leaves
# in a given years, which is not realistic...a tree will always try to sprout
# leaves, even if they die), or that a PFT dies and is replanted every couple
# years (also unusual for trees).  The other colors represent in-between cases.
#
# In addition to the map, example plots of the colored timeseries are also
# created, in order to give the viewer an idea of what the colors represent.
# On these plots is listed the criteria by which we judge the timeseries.
# A maximum of ten timeseries is shown, but in the case where more than
# 20 timeseries are classified by that color, a clustering algorithm is
# first applied to group the timeseries into ten different clusters.  One
# timeseries is selected at random from each of these clusters, under the
# assumption that this produces the most diverse sample for viewing.  The
# latitude and longitude coordinates of each plotted timeseries are included
# for reference.
#
###############################################
# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys
import argparse
import re

# Import from local routines
from grid import Grid
from extraction_subroutines import extract_variables_from_file,extract_timeseries,read_in_extracted_timeseries
from classification_subroutines import classify_observations,plot_classified_observations,map_classified_pixels,extract_and_calculate_classifiction_metrics,simulation_parameters,create_netcdf_classification_map
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

parser.add_argument('--do_test', dest='do_test', action='store',default="False",
                   help='if TRUE, we will limit the number of timeseries we extract to ntest_points.  This makes the process faster, and is good for checking the format of output plots.')
parser.add_argument('--data_years', dest='data_years', action='store',
                   default="1901,2241",
                   help='the years of data to analyze. If a file with these data years is not found, we attempt to create one (which takes longer).')
parser.add_argument('--plot_years', dest='plot_years', action='store',
                   default="101,150",
                   help='the years of the timeseries to plot in the example timeseries.  101,150 is good for forests since they are mature, but croplands and grasslands may be different.   If you plot more than 50 years with annual data, it can be hard to see behavior.')
parser.add_argument('--global_operation', dest='global_operation', action='store',
                   default='simplesum', choices=["simplesum",'weightedpftsum',"weightedareasum",'weightedpftave'],
                   help='how to combine the pixels for plotting the global timeseries.  Choices are simplesum, weightedpftsum, weightedareasum, and weightedpftave.')

parser.add_argument('--year_increment', dest='year_increment', action='store',
                   default=1, type=int,
                   help='the number of years each of our history files covers (default 1, for a TEST run...10 is common for PROD runs)')
parser.add_argument('--force_annual', dest='force_annual', action='store',
                   default="False", 
                   help='if True/Yes/y/Y, then attempt to change the time axis to annual if it is not already.')
parser.add_argument('--fix_time_axis', dest='fix_time_axis', action='store',
                   default=None, choices=[None,"daily","monthly","annual"] ,
                    help='if given a value, then attempt to create a time axis that goes from the starting date to the ending date with this timestep.  Used in the case that the existing time axis is incorrect (a subroutine checks to make sure all the values are increasing, which is not the case with ORCHIDEE between subsquent history files...they reset to zero every year).  NOTE: if used with --force_annual, the value of --fix_time_axis should be that of the original data.')
parser.add_argument('--force_calendar', dest='force_calendar', action='store',
                   default="noleap", 
                   help='if fix_time_axis is set to daily, this calendar will be used.')

parser.add_argument('--extract_only', dest='extract_only', action='store',
                   default="False", 
                   help='if True/Yes/y/Y, then stop the code after the variables have been extracted (do not classify them or make a map)')

##
parser.add_argument('--print_all_timeseries', dest='print_all_ts', action='store',
                   default="False",
                   help='If True/Yes/y/Y, create a .csv file with each class of timeseries using the rows as the dates and the columns as sat/lon strings.  This is primarily useful for re-analsis.')
parser.add_argument('--print_ts_region', dest='print_ts_region', action='store',
                   #default="35N,75N,-20E,40E",
                   default="-90N,90N,-180E,180E",
                    help='Limit printing of all timeseries to the specified region (specifying a rectangle using a format "35N,75N,-20E,40E", using negative numbers for west and south).  Requires print_all_timeseries to also be True/Yes/y/Y.')

parser.add_argument('--supp_title_string', dest='supp_title_string', action='store',
                   default="_",
                    help='Something to add to the simulation name to make distinct files.  Default is nothing.')

parser.add_argument('--plot_points_from_file', dest='plot_points_filename', action='store',
                   default=None,
                    help='If this is the name of a .csv file with format Index,Latitude,Longitude, these are the points that will be plotted on the timeseries graphs.  If not specified, points are selected at random based on a clustering algorithm.')

parser.add_argument('--extract_region', dest='extract_region', action='store',
                   default=None,
                    help='Extract variables only for a given point or region.  For a region, specify a rectangle using a format "35N,75N,-20E,40E", using negative numbers for west and south).  For a point, use the same latitude and longitude values (45.0N,45.0N,-1.0E,-1.0E).  This will override print_ts_region.')

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
elif timeseries_flag == "LAI_MEAN_RMSD":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN information, primarily the RMSD. (use the --classification flag to change)")
elif timeseries_flag == "LAI_MEAN2":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN and IND information. (use the --classification flag to change)")
elif timeseries_flag == "LAI_MEAN_BIMODAL":
    print("Extracting the LAI_MEAN timeseries for every pixel, and classifying only using LAI_MEAN.  Flagging bimodal distributions (use the --classification flag to change)")
elif timeseries_flag == "TWBR":
    print("Extracting the TWBR timeseries for every pixel, and classifying only using it.  Checking for poor water conservation (use the --classification flag to change)")

elif timeseries_flag == "N_RESERVES":
    print("Extracting the timeseries for the nitrogen reserve and labile pools, and classyfing the timeseries only on them.  Checking for accumulating N reserves on decennial timescales, as that has been a sign of problems before (use the --classification flag to change)")

elif timeseries_flag == "TOTAL_M_c":
    print("Extracting the timeseries for the total carbon biomass, and classyfing the timeseries only on them.  Checking for accumulating biomass over hundreds of years, as pixels should in theory reach maximum biomass in 300 years or so (use the --classification flag to change)")

elif timeseries_flag == "multiplate_stable_states_tbmc":
    print("Extracting the timeseries for the total carbon biomass, and classyfing the timeseries only on them.  Check for sudden jumps in the timeseries, indicating the presense of multiple stable states (use the --classification flag to change)")

elif timeseries_flag == "HEIGHT":
    print("Extracting the timeseries for the vegetation height, and classyfing the timeseries only on them.  Checking for frequent drops of vegetation height back to some initial value, which suggests frequent die-offs (use the --classification flag to change)")

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
if fix_time_axis:
    print("Attempting to overwrite the existing time axis. (use the --fix_time_axis flag to change)")
    print("   ",fix_time_axis)
    force_calendar=args.force_calendar
    print("   ",force_calendar)
else:
    print("Leaving the time axis alone. (use the --fix_time_axis flag to change)")
    force_calendar=None
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
if global_operation == "simplesum":
    print("Summing together all the pixels WITHOUT taking into account area to create a global timeseries (use the --global_operation flag to change)")
elif global_operation == "weightedpftsum":
    print("Summing together all the pixels taking into account the fraction of a pixel covered by this PFT AND the pixel area to create a global timeseries (use the --global_operation flag to change)")
elif global_operation == "weightedareasum":
    print("Summing together all the pixels taking into account only the area of the pixel to create a global timeseries (use the --global_operation flag to change)")
elif global_operation == "weightedpftave":
    print("Summing together all the pixels taking into account the fraction of a pixel covered by this PFT AND the pixel area, and then divide by the total PFT area to create a global timeseries (use the --global_operation flag to change)")
else:
    print("Not yet ready for this global operation! ",global_operation)
    sys.exit(1)
#endif

year_increment=args.year_increment
print("{} years between our history files (use the --year_increment flag to change)".format(year_increment))

print_all_ts=args.print_all_ts
print_ts_region=args.print_ts_region
if print_all_ts.lower() in possible_true_values:
    print_all_ts=True
else:
    print_all_ts=False
#endif
if print_all_ts:
    print("Printing all timeseries to a .csv file (as a debugging option). (use the --print_all_timeseries flag to change)")
    print("Printing timeseries for region: ",print_ts_region)
else:
    print("Not printing all timeseries to a .csv file. (use the --print_all_timeseries to change)")
#endif

extract_region=args.extract_region
if extract_region is not None:
    print("Extracting just a pixel or region: ",extract_region)
    print("   (use --extract_region to change)")
    print("   NOTE: this overrides print_ts_region.")
    print_ts_region=extract_region
else:
    print("Extracting the whole map.")
    print("   (use --extract_region to change)")
#endif

supp_title_string=args.supp_title_string

plot_points_filename=args.plot_points_filename
if plot_points_filename:
    print("Plotting the points specified in the file {}. (use the --plot_points_from_file flag to change)".format(plot_points_filename))
else:
    print("Plotting points selected by a clustering algorithm at random. (use the --plot_points_from_file flag to change)")
#endif

print("######################## END INPUT VALUES ######################")

# To make things easier, pass around a stucture with simulation parameters.
sim_params=simulation_parameters(pft_selected,veget_max_threshold,timeseries_flag,do_test,global_operation,force_annual,fix_time_axis,print_all_ts,print_ts_region,supp_title_string,plot_points_filename,force_calendar,extract_region)

###############################################


###############################################
# Extract the variables of interest from the .nc files
# I assume that a bunch of files are in a directory
# FG1.PFTCHECK.r6811_21100101_21101231_1Y_stomate_history.nc
# I tried to use ncrcat for this, but ORCHIDEE uses the variable
# time_counter instead of time, so it didn't work for me.
# This creates a new .nc file that I can read in later. 
sim_params.set_extraction_information(syear_data,eyear_data,year_increment,input_file_name_start,input_file_name_end)

# If the data file exists, we don't have to create it.
try:
    srcnc = NetCDFFile(sim_params.condensed_nc_file_name,"r")
    srcnc.close()
    print("Data file already exists: ",sim_params.condensed_nc_file_name)
except:
    print("Need to create file: ",sim_params.condensed_nc_file_name)
    extract_variables_from_file(sim_params)
#endtry

if extract_only:
    print("Finishing script after variable extraction.")
    sys.exit(0)
#endif


##########################################################

##########################################################
# Set up some information for the classification procedure
# This routine also checks to make sure we have
# the variables we need for the classification.
sim_params.set_classification_filename_information()
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
timeseries_array,timeseries_lat,timeseries_lon, classification_array,classified_points=extract_and_calculate_classifiction_metrics(sim_params)


##########################################################
# Based on the classification vector for each timeseries, classify
# it.  We aim to identify good, ok, and bad points.
# classification_vector[i]
classification_vector=classify_observations(classification_array,sim_params)
##########################################################

##########################################################
# Generate a few random plots of each classification.
level_colors,colors_by_level=plot_classified_observations(classification_vector,timeseries_array,timeseries_lat,timeseries_lon,sim_params,syear_plot,eyear_plot,classification_array,classified_points)
##########################################################

##########################################################
# Plot a map of all classified pixels.
map_classified_pixels(classification_vector,timeseries_lat,timeseries_lon,level_colors,colors_by_level,sim_params)
##########################################################

##########################################################
# Print a NetCDF file containing with the classification and the
# timeseries for every pixel to make it easier to hunt down bad pixels.
# This is not perfect, since it's only one of the timeseries used in the
# classification, but 
create_netcdf_classification_map(classification_vector,timeseries_lat,timeseries_lon,sim_params)
##########################################################


print("Finished with program.")
