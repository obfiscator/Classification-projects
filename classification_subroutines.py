# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys,traceback
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from sklearn.cluster import KMeans
import math
import matplotlib.gridspec as gridspec
from textwrap import fill # legend text can be too long
import matplotlib as mpl
import re
import statistics
from scipy.stats import pearsonr

# Import from local routines myself and colleagues have written
from grid import Grid
from netcdf_subroutines import find_orchidee_coordinate_names,find_variable

###################################

class classified_pixel_information:
    def __init__(self, primary_timeseries, latitude, longitude, area, contfrac, veget_max):
        self.primary_timeseries = primary_timeseries 
        self.veget_max = veget_max
        self.area=area
        self.contfrac=contfrac
        self.latitude=latitude
        self.longitude=longitude
    #enddef
#endclass

class simulation_parameters:
    def __init__(self, pft_selected,veget_max_threshold,timeseries_flag,do_test,global_operation,force_annual,fix_time_axis,print_all_ts,print_ts_region,supp_title_string,plot_points_filename,force_calendar,extract_region):
        self.pft_selected = pft_selected # Note that this is stored in
        # here with Python indexing!  So PFT6 is stored here as 5.
        self.veget_max_threshold = veget_max_threshold
        self.timeseries_flag=timeseries_flag
        self.do_test=do_test
        self.ntest_points=100
        self.global_operation=global_operation
        self.force_annual=force_annual
        self.fix_time_axis=fix_time_axis
        self.force_calendar=force_calendar
        self.print_all_ts=print_all_ts
        self.print_ts_region=print_ts_region
        self.supp_title_string=supp_title_string
        self.plot_points_filename=plot_points_filename
        self.extract_region=extract_region

        # Based on the above, set some other flags.

        ######## Better to always extract all variables.  Otherwise, I
        ######## cannot reuse extracted files and have to redo them.
        # This can be tricky, as one variable may have a name in one run
        # and another in a different (e.g., LAI_MEAN in TRUNK 4.0 but LAI in 
        # TRUNK 2.2).  I would like this to work in the same way for both,
        # so the code needs to figure out which variable is present.
        # Try putting both the equivalent names here, and then collapsing it
        # to a single name when the variables are read in.

        # Note that "ncrcat -v LEAF_AGE_CRIT,LEAF_AGE,LEAF_TURN_c,LAI_MEAN_GS,
        # FRUIT_M_c,WSTRESS_SEASON,NPP,GPP,LEAF_M_MAX_c,LEAF_TURN_AGEING_c 
        # FG1.test.n_192* all.nc"
        # is much faster than this script, but doesn't create a good time axis.

        # If a variable is not found, skip it.  Extracting variables takes
        # a long time, so I don't want to have to redo it for every type
        # of simulations.
        #self.variables_to_extract=[ ["LAI_MEAN","LAI"], ["LAI_MEAN_GS","LAI"], ["LAI_MAX","LAI"],["VEGET_MAX","VEGET_COV_MAX"],"IND","TWBR","LABILE_M_n","RESERVE_M_n","NPP","GPP","NBP_pool","Areas","CONTFRAC","LEAF_AGE_CRIT","LEAF_AGE","LEAF_TURN_c","LAI_MEAN_GS","FRUIT_M_c","WSTRESS_SEASON","LEAF_M_MAX_c",'LEAF_TURN_AGEING_c',"LABILE_M_c","RESERVE_M_c","SAP_M_AB_c","SAP_M_BE_c","TOTAL_M_c","HEIGHT_DOM","RECRUITS_IND","tair","swdown","rain","snowf"]

        # For CRUERA, daily output for ten years, I'm trying to find the
        # minimum set of values I need for the analysis, since it appears
        # to be having memory issues.
        self.variables_to_extract=[ ["LAI_MEAN","LAI"], ["VEGET_MAX","VEGET_COV_MAX"],"tair","swdown","rain","snowf","Areas","CONTFRAC"]

        # TEST TO MAKE THINGS FASTER
        #self.variables_to_extract=[ "Areas","tair",["LAI_MEAN","LAI"]]
        #self.variables_to_extract=[ "TWBR"]
        # not sure what to do with time_counter_bounds.  I am trying to always create it myself.



        #####################
        # This is for a different purpose.  Sometimes I just want to extract variables.
#        self.variables_to_extract=[ ["VEGET_MAX","VEGET_COV_MAX"],"SAP_M_AB","SAP_M_BE","HEART_M_AB","HEART_M_BE","TOTAL_SOIL_CARB","TOTAL_BM_LITTER","LITTER_STR_AB","LITTER_MET_AB","LITTER_STR_BE","LITTER_MET_BE","PROD10","PROD100","PROD10_HARVEST","PROD100_HARVEST"]
# I may need the product pools, but for right now I do not know how to deal
# with a fourth non-veget axis in the array processing.
#        self.variables_to_extract=[ ["VEGET_MAX","VEGET_COV_MAX"],"SAP_M_AB","SAP_M_BE","HEART_M_AB","HEART_M_BE","TOTAL_SOIL_CARB","TOTAL_BM_LITTER","LITTER_STR_AB","LITTER_MET_AB","LITTER_STR_BE","LITTER_MET_BE","WOOD_HARVEST_PFT"]
        #####################

        self.variables_in_which_file={"LAI_MEAN":"stomate","LAI_MAX":"stomate","LAI" : "stomate","VEGET_MAX":"stomate","VEGET_COV_MAX":"stomate","IND":"stomate","TWBR":"sechiba","SAP_M_AB_c":"stomate","SAP_M_BE_c":"stomate","HEART_M_AB":"stomate","HEART_M_BE":"stomate","TOTAL_SOIL_CARB" : "stomate","TOTAL_BM_LITTER":"stomate","LITTER_STR_AB":"stomate","LITTER_MET_AB":"stomate","LITTER_STR_BE":"stomate","LITTER_MET_BE":"stomate","PROD10":"stomate","PROD100":"stomate","PROD10_HARVEST":"stomate","PROD100_HARVEST":"stomate","WOOD_HARVEST_PFT":"stomate","RESERVE_M_n":"stomate","LABILE_M_n":"stomate", "LAI_MEAN_GS":"stomate","NPP":"stomate","GPP":"stomate","NBP_pool":"stomate","Areas":"stomate","CONTFRAC":"stomate","LEAF_AGE_CRIT":"stomate","LEAF_AGE":"stomate","LEAF_TURN_c":"stomate","LAI_MEAN_GS":"stomate","FRUIT_M_c":"stomate","WSTRESS_SEASON":"stomate","LEAF_M_MAX_c":"stomate",'LEAF_TURN_AGEING_c':"stomate","RESERVE_M_c":"stomate","LABILE_M_c":"stomate","TOTAL_M_c" : "stomate","HEIGHT_DOM" : "stomate","RECRUITS_IND" : "stomate","tair" : "sechiba", "swdown" : "sechiba", "rain" : "sechiba", "snowf" : "sechiba"}

        if self.timeseries_flag in ("LAI_MEAN1","LAI_MEAN2","LAI_MEAN_BIMODAL","LAI_MEAN_RMSD"):
            #self.variables_to_extract=["LAI_MEAN","LAI_MAX","VEGET_MAX","IND","time_counter_bounds"]
            #self.variables_in_which_file=["stomate","stomate","stomate","stomate","stomate"]
#            self.timeseries_variable="LAI_MEAN_GS"
            #self.timeseries_variable="LAI_MEAN"
            print("Not selecting LAI variable here.  That should be done below?")
        elif self.timeseries_flag in ("TWBR"):
            #self.variables_to_extract=["time_counter_bounds","TWBR"]
            #self.variables_in_which_file=["stomate","sechiba"]
            self.timeseries_variable="TWBR"
        elif self.timeseries_flag in ("N_RESERVES"):
            #self.variables_to_extract=["time_counter_bounds","TWBR"]
            #self.variables_in_which_file=["stomate","sechiba"]
            self.timeseries_variable="RESERVE_M_n" # We actually have
            # two timeseries for this case.  Not yet sure the best way to
            # make it general
        elif self.timeseries_flag in ("TOTAL_M_c"):
            self.timeseries_variable="TOTAL_M_c"
        elif self.timeseries_flag in ("HEIGHT"):
            self.timeseries_variable="HEIGHT_DOM"
        else:
            print("I do not recognize this timeseries flag in init sim param!")
            print(self.timeseries_flag)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # LAI thresholds suggested by Sebastiaan:
        # No LAI for PFT 0 (bare soil)
        if pft_selected == 1: # Remember that Python arrays start from 0!
            #self.lai_good_threshold=5.0   8400 pixels out of 14000 are classifed as dark green on Trendyv9 for Tag 2.2
            self.lai_good_threshold=6.0
        elif pft_selected == 2: 
            #self.lai_good_threshold=5.0 # original
            self.lai_good_threshold=4.0  # modified for TAG 2.2
        elif pft_selected == 3: 
            self.lai_good_threshold=4.0
        elif pft_selected == 4: 
            #self.lai_good_threshold=2.0 # original 
            self.lai_good_threshold=3.0 # modified for TAG 2.2
        elif pft_selected == 5: 
            #self.lai_good_threshold=4.0 
            self.lai_good_threshold=2.5 # modified to give TAG 2.1 more green
        elif pft_selected == 6: 
            self.lai_good_threshold=3.0 
        elif pft_selected == 7: 
#            self.lai_good_threshold=3.0 # original
            self.lai_good_threshold=1.5 # modified for TAG 2.2
        elif pft_selected == 8: 
            #self.lai_good_threshold=2.0 # original
            self.lai_good_threshold=1.5 # modified for TAG 2.2
        elif pft_selected == 9: 
#            self.lai_good_threshold=3.0 # original
            self.lai_good_threshold=2.0 # modified for TAG 2.2
        elif pft_selected == 10: 
#            self.lai_good_threshold=3.0 # original
            self.lai_good_threshold=2.0 # modified for TAG 2.2
        elif pft_selected == 11: 
#            self.lai_good_threshold=3.0 # original
            self.lai_good_threshold=2.5 # modified for TAG 2.2
        elif pft_selected == 12: 
            self.lai_good_threshold=3.0
        elif pft_selected == 13: 
#            self.lai_good_threshold=2.0 # original
            self.lai_good_threshold=1.5 # modified for TAG 2.2
        elif pft_selected == 14: 
#            self.lai_good_threshold=2.0 # original
            self.lai_good_threshold=1.5 # modified for TAG 2.2
        else:
            print("Not sure what to do with this PFT in sim_params def!")
            print(pft_selected)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # The basic strategy for ideal growth is to have an LAI threshold
        # and check to see how many years are above that.  That
        # threshold is multiplied by this number to see what pixels
        # have good growth (for species not in the core of their range).
        self.good_growth=0.75 

        # Below this value, we may consider an LAI value to be zero in
        # certain criteria
        self.zero_threshold=0.05

        # What fraction of our datapoints need to be above this in order for it to be good?
        self.lai_high_threshold_good_fraction=0.8
        self.lai_high_threshold_bad_fraction=0.1 # if it's never more than this, it's bad


        # A lower limit to the LAI
        if pft_selected in (1,2,3,5,6,7,9,10,11,12): 
            self.lai_bad_threshold=1.0
        elif pft_selected in (4,8,13,14): # a few PFTs with lower LAI
            self.lai_bad_threshold=0.5
        else:
            print("I don't know how to create criteria for this PFT!")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # What fraction of our datapoints need to be below this in order for it to be bad?
        self.lai_low_threshold_bad_fraction=0.8
        self.lai_low_threshold_good_fraction=0.1 # this can be good

        # How big of LAI jumps do we start getting concerned about?
        self.lai_diff_threshold=0.5
        # What fraction of the timesteps do we need to see them in to get concerned?
        self.lai_diff_bad_threshold=0.3

        # For the threshold on the number of individuals, I look at the number of times the
        # number of individuals is greater than the number planted...NMAXTREES
        if pft_selected in (1,2,3,4,5,6,7,8): 
            self.ind_upper_threshold=14000*0.7
        elif pft_selected in (9,10,11,12,13,14):
            self.ind_upper_threshold=10000*0.7
        else:
            print("I don't know how to create criteria for this PFT for ind_upper_threshold!")
            print(self.pft_selected)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Above which fraction of values are NaN do we flag it as bad?
        self.nan_bad_threshold=0.05
        # Below which fraction of values are NaN can we accept it as good?
        self.nan_good_threshold=0.01

        ## This is for the TWBR
        self.twbr_high_threshold=1e-15
        self.twbr_low_threshold=1e-16
        self.twbr_high_threshold_bad_fraction=0.1
        self.twbr_low_threshold_good_fraction=0.5

        # This is for the nitrogen reserve pools
        self.nreserves_ndecades=5
        self.nreserves_good_fraction=0.95
        self.nreserves_okay_fraction=0.8

        # This is for the total biomass
        self.totalmc_ndecades=34

        # This is for the height.
        # I got these values by looking at a bunch of annual values
        # for the heights in a run starting from scratch on various pixels.
        # The results seem to be within 10% of each other.
        height_thresholds=np.zeros((15))
        height_thresholds[0]=0.0
        height_thresholds[1]=4.2
        height_thresholds[2]=4.1
        height_thresholds[3]=3.8
        height_thresholds[4]=4.1
        height_thresholds[5]=3.0
        height_thresholds[6]=3.0
        height_thresholds[7]=2.1
        height_thresholds[8]=3.1
        # Crops and grasslands have a height of 0.0.
        # Not sure it makes sense to do anything for them.
        height_thresholds[9]=0.0
        height_thresholds[10]=0.0
        height_thresholds[11]=0.0
        height_thresholds[12]=0.0
        height_thresholds[13]=0.0
        height_thresholds[14]=0.0
        self.height_threshold=height_thresholds[pft_selected]

        # This is for the RMSD of the LAI.
        # I got these values by looking at the distributions of RMSD for
        # the LAI for each PFT.  I tried to get about a quarter of the pixels
        # in each classification (1: less than low; 2: less than medium; 
        # 3: less than high; 4: greater than high)
        lai_rmsd_thresholds=np.zeros((15,3))
        lai_rmsd_thresholds[0,:]=(0.0,0.0,0.0)
        lai_rmsd_thresholds[1,:]=(0.4,0.6,0.9)
        lai_rmsd_thresholds[2,:]=(0.5,0.8,1.3)
        lai_rmsd_thresholds[3,:]=(0.4,0.8,1.1)
        lai_rmsd_thresholds[4,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[5,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[6,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[7,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[8,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[9,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[10,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[11,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[12,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[13,:]=(0.2,1.0,2.0)
        lai_rmsd_thresholds[14,:]=(0.2,1.0,2.0)
        
        self.lai_high_threshold_rmsd=lai_rmsd_thresholds[pft_selected,2]
        self.lai_medium_threshold_rmsd=lai_rmsd_thresholds[pft_selected,1]
        self.lai_low_threshold_rmsd=lai_rmsd_thresholds[pft_selected,0]

    #enddef

    def set_extraction_information(self,syear,eyear,year_increment,input_file_name_start,input_file_name_end):
        self.syear=syear
        self.eyear=eyear
        self.ntotal_data_years=eyear-syear+1
        self.year_increment=year_increment
        self.input_file_name_start=input_file_name_start
        self.input_file_name_end=input_file_name_end

        # Drop the trailing underscore to get an identifier for
        # the simulation.  Useful in plotting.
        if self.input_file_name_start[-1] == "_":
            self.sim_name=self.input_file_name_start[0:-1]
        else:
            self.sim_name=self.input_file_name_start
        #endif

        # This the string we use in naming the output files.
        self.output_file_string=self.sim_name+self.supp_title_string

        # Figure out which tagged version this is, also for plotting:
        if self.sim_name in ["FG2low.ORCH21r5695"]:
            self.tag_version="2.1"
        elif self.sim_name in ["FG2.TRENDY9.S3"]:
            self.tag_version="2.2"
        elif self.sim_name in ["FG2-r6830-param1","S3-r6863-TRENDY-twodeg"]:
            self.tag_version="3.0"
        else:
            self.tag_version="4.0"
        #endif
        print("Simulation {} is TAG {}.".format(self.sim_name,self.tag_version))

        # This is where we put the maps of every variable that
        # we extract, in NetCDF format
        # I'm not sure why I have this if statement.
#        if year_increment == 1:
        region_string=self.print_ts_region.replace(",","x")
        self.condensed_nc_file_name="extracted_variables_{}_{}{}.{}.nc".format(region_string,input_file_name_start,syear,eyear)
#        else:
#            self.condensed_nc_file_name="extracted_variables_{}{}.{}.nc".format(input_file_name_start,syear,eyear)
        #endif

        # Figure out a window where we print timeseries.  Notice this is
        # also automatically set if extract_region is specified.
        self.nlat_window,self.slat_window,self.wlon_window,self.elon_window=parse_latlon_string(self.print_ts_region)

        # These are the time axis units we want, in case of regridding
        self.desired_oyear=1901
        self.desired_omonth=1
        self.desired_oday=1
        self.desired_ohour=0
        self.desired_omin=0
        self.desired_osec=0
        ## Note that this should be the same as found in the data files
        ## if you are forcing the time axis
        self.desired_ounits="seconds" # seconds may be problematic for 340Y?
        if self.desired_ohour != "":
            self.desired_timeorigin="{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(self.desired_oyear,self.desired_omonth,self.desired_oday,self.desired_ohour,self.desired_omin,self.desired_osec)
        else:
            self.desired_timeorigin="{0:04d}-{1:02d}-{2:02d}".format(self.desired_oyear,self.desired_omonth,self.desired_oday)
        #endif
        self.desired_timeunits="{} since ".format(self.desired_ounits) + self.desired_timeorigin


    #enddef

    # In order for this to work, the NC file must already exist!
    def set_classification_filename_information(self):

        # From here, I want to check some of the variable names.  They
        # may be different in different tagged versions.  For example,
        # LAI in TAG 2.1 is the LAI averaged over the full year.  
        # In TAG 4.0, the LAI averaged over the full year is LAI_MEAN.
        # But in theory they measure the same thing.
        srcnc = NetCDFFile(self.condensed_nc_file_name,"r")
        veget_max_name=find_variable(["VEGET_MAX","VEGET_COV_MAX"],srcnc,False,"",lcheck_units=False)
        # It's probably more accurate to compare LAI to LAI_MEAN.
        lai_mean_name=find_variable(["LAI","LAI_MEAN"],srcnc,False,"",lcheck_units=False)
        #lai_mean_name=find_variable(["LAI","LAI_MEAN_GS"],srcnc,False,"",lcheck_units=False)
        ######
        lai_max_name=find_variable(["LAI","LAI_MAX"],srcnc,False,"",lcheck_units=False)
        self.set_variable_names(veget_max_name,lai_mean_name,lai_max_name)


        # Check that we have all the variables that we need.
        self.required_vars=[]
        if self.timeseries_flag == "LAI_MEAN1":
            self.required_vars=[veget_max_name,lai_mean_name,"CONTFRAC","Areas"]
            self.timeseries_variable=lai_mean_name
        elif self.timeseries_flag == "LAI_MEAN_BIMODAL":
            self.required_vars=[veget_max_name,lai_mean_name,"CONTFRAC","Areas"]
            self.timeseries_variable=lai_mean_name
        elif self.timeseries_flag == "LAI_MEAN2":
            self.required_vars=[veget_max_name,lai_mean_name,"CONTFRAC","Areas"]
            self.timeseries_variable=lai_mean_name
        elif self.timeseries_flag == "LAI_MEAN_RMSD":
            self.required_vars=[veget_max_name,lai_mean_name,"CONTFRAC","Areas"]
            self.timeseries_variable=lai_mean_name
        elif self.timeseries_flag == "TWBR":
            self.required_vars=['TWBR',"CONTFRAC","Areas"]
        elif self.timeseries_flag == "HEIGHT":
            self.required_vars=['HEIGHT','IND','RECRUITS_IND',"CONTFRAC","Areas"]
        elif self.timeseries_flag == "N_RESERVES":
            self.required_vars=[veget_max_name,"RESERVE_M_n","LABILE_M_n","RESERVE_M_c","LABILE_M_c","SAP_M_AB_c","SAP_M_BE_c","CONTFRAC","Areas"]
        elif self.timeseries_flag == "TOTAL_M_c":
            self.required_vars=['TOTAL_M_c',"CONTFRAC","Areas"]
        else:
            print("Do not recognize timeseries flag in set_classification_filename_information!")
            print(self.timeseries_flag)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        for varname in self.required_vars:
            if varname not in srcnc.variables.keys():
                print("Cannot find an analysis variable I need in the extracted data file!")
                print("File location: ",srcnc.filepath())
                print("Looking for: ",varname)
                print("Variables in file: ",srcnc.variables.keys())
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        #endfor

        # Some information for the a global plot of this variable
        self.global_plot_scale=1.0
        self.global_plot_units='-'
        
        # Store the names of all the variables we will need.
        self.variables_used=[]

        # These names will depend on the analysis we are doing
        if self.timeseries_flag == "LAI_MEAN1":

            # For the name of the file with the map.  I will use this
            # identifier elsewhere, too.
            self.cmap_identifier="{}PFT{}_LAIMEAN".format(self.output_file_string,self.pft_selected+1)

            # And the title of the map itself
            self.classified_map_title="{} - TAG {}\nEach pixel classified according to\nthe {} timeseries for PFT {}".format(self.sim_name,self.tag_version,self.lai_mean_name,self.pft_selected+1)

            # Units on the y-axis of the plot
            self.ylabel="{} [{}]".format(lai_mean_name,srcnc[lai_mean_name].units)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN.png".format(self.output_file_string,self.pft_selected+1)


            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            # Level 1
            #self.plot_subtitle.append("Problem pixels for PFT {0}, classification based only on {1}\nA pixel in this category has more than {2}% of the annual {1} values as NaNs.".format(self.pft_selected+1,self.lai_mean_name,self.nan_bad_threshold*100.0))
#            self.plot_subtitle.append("Problem pixels for PFT {0}, classification based only on {1}\nA pixel in this category has more than {2}% of the annual {1} values as NaNs\n    OR more than {2}% of the annual {1} values as 0.0.".format(self.pft_selected+1,self.lai_mean_name,self.nan_bad_threshold*100.0))
            self.plot_subtitle.append("Problem pixels for PFT {0}, classification based only on {1}\nA pixel in this category has more than {2}% of the annual {1} values as NaNs\n    OR more than {2}% of the annual {1} values as 0.0\n    OR more than {2}% of the annual {1} values less than {3}.".format(self.pft_selected+1,self.lai_mean_name,self.nan_bad_threshold*100.0,self.zero_threshold))
            # Level 2
            self.plot_subtitle.append("Weak growth for PFT {0}, classification based only on {1}\nPixels in this category has at least {2}% of annual {1} values below {3},\n    AND no more than {4}% of annual {1} values above {5}".format(self.pft_selected+1,self.lai_mean_name,self.lai_low_threshold_bad_fraction*100.0,self.lai_bad_threshold,self.lai_high_threshold_bad_fraction*100.0,self.lai_good_threshold))

            # Level 3
            self.plot_subtitle.append("PFT {}, classification based only on {}\nPixels in this category do not satisfy the rules for any of the rest of the levels.".format(self.pft_selected+1,self.lai_mean_name))
            # Level 4
            self.plot_subtitle.append("Good growth for PFT {0}, classification based only on {1}\nA pixel in this category has at least {2}% of annual {1} values above {3},\n    AND no more than {4}% of annual {1} values below {5}".format(self.pft_selected+1,self.lai_mean_name,self.lai_high_threshold_good_fraction*100.0*self.good_growth,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold))
            # Level 5
            self.plot_subtitle.append("Excellent growth for PFT {0}, classification based only on {1}\nA pixel in this category has at least {2}% of annual {1} values above {3},\n    AND no more than {4}% of annual {1} values below {5},\n    AND no more than {6}% of the annual {1} values are NaNs.".format(self.pft_selected+1,self.lai_mean_name,self.lai_high_threshold_good_fraction*100.0,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold*100.0))

            ########
        elif self.timeseries_flag == "LAI_MEAN_RMSD":

            # For the name of the file with the map.  I will use this
            # identifier elsewhere, too.
            self.cmap_identifier="{}PFT{}_LAIMEANRMSD".format(self.output_file_string,self.pft_selected+1)

            # And the title of the map itself
            self.classified_map_title="{} - TAG {}\nEach pixel classified according to\nthe {} timeseries for PFT {}".format(self.sim_name,self.tag_version,self.lai_mean_name,self.pft_selected+1)

            # Units on the y-axis of the plot
            self.ylabel="{} [{}]".format(lai_mean_name,srcnc[lai_mean_name].units)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEANRMSD.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEANRMSD.png".format(self.output_file_string,self.pft_selected+1)


            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            # Level 1
            self.plot_subtitle.append("Problem pixels for PFT {0}, classification based only on {1}\nA pixel in this category has more than {2}% of the annual {1} values as NaNs\n    OR more than {2}% of the annual {1} values as 0.0\n    OR more than {2}% of the annual {1} values less than {3}.".format(self.pft_selected+1,self.lai_mean_name,self.nan_bad_threshold*100.0,self.zero_threshold))
            # Level 2
            self.plot_subtitle.append("Extreme fluctuation for PFT {0}, classification based only on {1}\nPixels in this category has an RMSD of the annual {1} values greater than {2}".format(self.pft_selected+1,self.lai_mean_name,self.lai_high_threshold_rmsd))

            # Level 3
            self.plot_subtitle.append("High fluctuation for PFT {0}, classification based only on {1}\nPixels in this category has an RMSD of the annual {1} values greater than {2}".format(self.pft_selected+1,self.lai_mean_name,self.lai_medium_threshold_rmsd))

            # Level 4
            self.plot_subtitle.append("Medium fluctuation for PFT {0}, classification based only on {1}\nPixels in this category has an RMSD of the annual {1} values greater than {2}".format(self.pft_selected+1,self.lai_mean_name,self.lai_low_threshold_rmsd))

            # Level 5
            self.plot_subtitle.append("Low fluctuation for PFT {0}, classification based only on {1}\nPixels in this category has an RMSD of the annual {1} values less than {2}".format(self.pft_selected+1,self.lai_mean_name,self.lai_low_threshold_rmsd))


        elif self.timeseries_flag == "LAI_MEAN_BIMODAL":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_LAIMEAN.png".format(self.output_file_string,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="Each pixel classified according to the LAI timeseries for PFT {}".format(self.pft_selected+1)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN.png".format(self.output_file_string,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA red pixel has more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.nan_bad_threshold*100.0))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nAn orange pixel has at least {}% of the changes between two consequitive LAI_MEAN values greater than {}.  This should indicate spikes.".format(self.pft_selected+1,self.lai_diff_bad_threshold*100.0,self.lai_diff_threshold))
#           self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nAn orange pixel has at least 40% of annual LAI_MEAN values below {},\n    AND at least 40% of annual LAI_MEAN values above {}.  This should indicate a bimodal distribution.".format(self.pft_selected+1,self.lai_bad_threshold,self.lai_good_threshold))

            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nThis is a pixel that doesn't fall in any other category.".format(self.pft_selected+1))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nThis color should not be used.".format(self.pft_selected+1))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA green pixel has at least {}% of annual LAI_MEAN values above {},\n    AND no more than {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_high_threshold_good_fraction*100.0,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold*100.0))



        elif self.timeseries_flag == "LAI_MEAN2":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_LAIMEAN_IND.png".format(self.output_file_string,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="Each pixel classified according to the LAI timeseries for PFT {}".format(self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN_IND.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN_IND.png".format(self.output_file_string,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle="PFT {}, classification based on LAI_MEAN and IND".format(self.pft_selected+1)

            ######
        elif self.timeseries_flag == "TWBR":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}TWBR.png".format(self.output_file_string)
            # And the title of the map itself
            self.classified_map_title="{}\nEach pixel classified according to the TWBR timeseries".format(self.sim_name)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}TWBR.png".format(self.output_file_string)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}TWBR.png".format(self.output_file_string)


            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("Classification based only on TWBR\nA bad pixel has at least {}% of annual TWBR values above a magitude of {},\n    AND less than {}% of annual TWBR values below a magnitude of {}.".format(self.twbr_high_threshold_bad_fraction*100.0,self.twbr_high_threshold,self.twbr_low_threshold_good_fraction*100.0,self.twbr_low_threshold))
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append("Classification based only on TWBR\nA good pixel has at least {}% of annual TWBR values below a magnitude of {},\n    AND less than {}% of annual TWBR values above a magnitude of {}.".format(self.twbr_low_threshold_good_fraction*100.0,self.twbr_low_threshold,self.twbr_high_threshold_bad_fraction*100.0,self.twbr_high_threshold))

            #####
        elif self.timeseries_flag == "N_RESERVES":

            # For the name of the file with the map.  I will use this
            # identifier elsewhere, too.
            self.cmap_identifier="{}PFT{}_NRESERVES".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_NRESERVES.png".format(self.output_file_string,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="{} - TAG {}\nEach pixel classified according to\nthe nitrogen reserve pools (labile, reserve) timeseries for PFT {}".format(self.sim_name,self.tag_version,self.pft_selected+1)

            # Units on the y-axis of the plot
            self.ylabel="{}\n[{}]".format("RESERVE_M_n+LABILE_M_n",srcnc[self.timeseries_variable].units)
            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_NRESERVES.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_NRESERVES.png".format(self.output_file_string,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("PFT {}, classification based on the sum of the nitrogen labile and reserve pools.\nA red pixel has a series of at least {} continuous decennial mean values\nwhere each value is greater than the previous decennial mean plus the standard deviation of the previous decade.".format(self.pft_selected+1,self.nreserves_ndecades))
            self.plot_subtitle.append("PFT {}, classification based on the sum of the nitrogen labile and reserve pools.\nOrange is not currently used.")

            self.plot_subtitle.append("PFT {}, classification based on the sum of the nitrogen labile and reserve pools.\nThis is a pixel that doesn't fall in any other category.".format(self.pft_selected+1))
            self.plot_subtitle.append("PFT {}, classification based on the sum of the nitrogen labile and reserve pools.\n   A light green pixel has at least {}% of decenniel mean N_RESERVE+N_LABILE values\n   within the average decennial standard deviation of the timeseries mean.".format(self.pft_selected+1, self.nreserves_okay_fraction*100.0))
            self.plot_subtitle.append("PFT {}, classification based on the sum of the nitrogen labile and reserve pools.\n   A dark green pixel has at least {}% of decenniel mean N_RESERVE+N_LABILE values\n   within the average decennial standard deviation of the timeseries mean.".format(self.pft_selected+1, self.nreserves_good_fraction*100.0))

            #####
        elif self.timeseries_flag == "TOTAL_M_c":

            # For the name of the file with the map.  I will use this
            # identifier elsewhere, too.
            self.cmap_identifier="{}PFT{}_TOTALMC".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_TOTALMC.png".format(self.output_file_string,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="{} - TAG {}\nEach pixel classified according to\nthe total carbon biomass timeseries for PFT {}".format(self.sim_name,self.tag_version,self.pft_selected+1)

            # Units on the y-axis of the plot
            self.ylabel="{}\n[{}]".format("TOTAL_M_c",srcnc[self.timeseries_variable].units)
            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_TOTALMC.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_TOTALMC.png".format(self.output_file_string,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("PFT {}, classification based on the total biomass of carbon.\nA red pixel has a series of at least {} continuous decennial mean values\nwhere each value is greater than the mean of the previous decennial mean plus the mean of the standard deviation of the first 5 decades.".format(self.pft_selected+1,self.totalmc_ndecades))
            self.plot_subtitle.append("PFT {}, classification based on the total biomass of carbon.\nOrange is not currently used.")

            self.plot_subtitle.append("PFT {}, classification based on the total biomass of carbon.\nThis is a pixel that doesn't fall in any other category.".format(self.pft_selected+1))

            self.plot_subtitle.append("PFT {}, classification based on the total biomass of carbon.\nLight green is not currently used.")
            self.plot_subtitle.append("PFT {}, classification based on the total biomass of carbon.\nDark green is not currently used.")

            # Some information for the a global plot of this variable
            self.global_plot_scale=1e-15 # assuming we are in grams
            self.global_plot_units='Pg C' # this is a stock, not a flux

            #####
        elif self.timeseries_flag == "HEIGHT":

            # For the name of the file with the map.  I will use this
            # identifier elsewhere, too.
            self.cmap_identifier="{}PFT{}_HEIGHT".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_HEIGHT.png".format(self.output_file_string,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="{} - TAG {}\nEach pixel classified according to\nthe height timeseries for PFT {}".format(self.sim_name,self.tag_version,self.pft_selected+1)

            # Units on the y-axis of the plot
            self.ylabel="{}\n[{}]".format("HEIGHT",srcnc[self.timeseries_variable].units)
            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_HEIGHT.png".format(self.output_file_string,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_HEIGHT.png".format(self.output_file_string,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("PFT {}, classification based on the height of the vegetation.\nThis classification is complicated.\nA red pixel has a low maximum height and/or many disturbances.".format(self.pft_selected+1))
            self.plot_subtitle.append("PFT {}, classification based on the height of the vegetation.\nAn orange pixel means that the correlation between NPP and height is less than -0.5.".format(self.pft_selected+1))

            self.plot_subtitle.append("PFT {}, classification based on the height of the vegetation.\nThis is a pixel that doesn't fall in any other category.".format(self.pft_selected+1))

            self.plot_subtitle.append("PFT {}, classification based on the height of the vegetation.\nA light green means the timeseries has not been disturbed\n   AND 80% of annual height values are above {:6.2f} meters.".format(self.pft_selected+1,self.height_threshold*2.0))
            self.plot_subtitle.append("PFT {}, classification based on the height of the vegetation.\nA dark green means the timeseries has not been disturbed\n   AND 80% of annual height values are above {:6.2f} meters.".format(self.pft_selected+1,self.height_threshold*3.0))

            # Some information for the a global plot of this variable
            self.global_plot_scale=1e-15 # assuming we are in grams
            self.global_plot_units='Pg C' # this is a stock, not a flux



        else:
            print("Do not recognize timeseries flag in set_classification_filename_information!")
            print(self.timeseries_flag)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # For the name of the file with the map
        self.classified_map_filename="classified_map_{}.png".format(self.cmap_identifier)

        # And close the file
        srcnc.close()

    #enddef

    def set_variable_names(self,veget_max_name,lai_mean_name,lai_max_name):
        self.veget_max_name=veget_max_name
        self.lai_mean_name=lai_mean_name
        self.lai_max_name=lai_max_name
        
        # Our timeseries variable depends on this, too.

        if self.timeseries_flag in ["LAI_MEAN1","LAI_MEAN_BIMODAL","LAI_MEAN2","LAI_MEAN_RMSD"]:
            self.timeseries_variable=lai_mean_name
     

            ######
        elif self.timeseries_flag == "TWBR":
            self.timeseries_variable="TWBR"

        elif self.timeseries_flag == "N_RESERVES":

            self.timeseries_variable="RESERVE_M_n" # We actually have
            # two timeseries for this case.  Not yet sure the best way to
            # make it general

        elif self.timeseries_flag == "TOTAL_M_c":

            self.timeseries_variable="TOTAL_M_c"

        elif self.timeseries_flag == "HEIGHT":

            self.timeseries_variable="HEIGHT"

        else:
            print("Do not recognize timeseries flag in set_variable_names!")
            print(self.timeseries_flag)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

    #enddef

#endclass

# This loops through all our classification criteria and tries to classify
# every point.  I aim for "good", "ok", and "bad".
def classify_observations(classification_array,sim_params):

    lai_high_threshold_good_fraction=sim_params.lai_high_threshold_good_fraction
    lai_low_threshold_good_fraction=sim_params.lai_low_threshold_good_fraction
    lai_high_threshold_bad_fraction=sim_params.lai_high_threshold_bad_fraction
    lai_low_threshold_bad_fraction=sim_params.lai_low_threshold_bad_fraction

    print("Classifying points.")
    nsamples=classification_array.shape[0]

    classification_vector=np.zeros((nsamples), dtype=np.int32)

    #print("For criteria 0: ",lai_high_threshold_good_fraction,lai_high_threshold_bad_fraction)
    #print("For criteria 1: ",lai_low_threshold_good_fraction,lai_low_threshold_bad_fraction)
    #print("For criteria 2: ",sim_params.nan_good_threshold,sim_params.nan_bad_threshold)

    ####
    # Try doing a KMeans clustering.  Just to see what it gives.
    # Not very useful, in the end.
    #kmeans = KMeans(n_clusters=10,random_state=0).fit(classification_array)
    #classification_vector=kmeans.labels_
    ####
    

    if sim_params.timeseries_flag == "LAI_MEAN1":

        for isample in range(nsamples):

            # "Good" is where all criteria are positive.  "Bad" is where all are negative.  "Semi-good" is where
            # at least one is positve, and none are negative.  "Semi-bad" is where at least one is negative, and none
            # are positive.  "OK" is everything else.
            # 0 : Fraction of annual LAI_MEAN values above an ideal threshold
            # 1 : Fraction of annual LAI_MEAN values below a different threshold
            # 2 : Fraction of annual LAI_MEAN values are NaNs
            # 3 : Fraction of timesteps when LAI_MEAN changes by more than 0.2
            # 4 : Fraction of annual LAI_MEAN values are zero
            # 5 : Fraction of annual LAI_MEAN values are close to zero
            # 6 : Fraction of annual LAI_MEAN values above a good threshold

            if (classification_array[isample,0] > lai_high_threshold_good_fraction) and (classification_array[isample,1] < lai_low_threshold_good_fraction) and (classification_array[isample,2] < sim_params.nan_good_threshold):
                # Good
                classification_vector[isample]=5
#            elif (classification_array[isample,2] > sim_params.nan_bad_threshold):
#            elif (classification_array[isample,2] > sim_params.nan_bad_threshold) or (classification_array[isample,4] > sim_params.nan_bad_threshold):
            elif (classification_array[isample,2] > sim_params.nan_bad_threshold) or (classification_array[isample,4] > sim_params.nan_bad_threshold) or (classification_array[isample,5] > sim_params.nan_bad_threshold):
                # Bad
                classification_vector[isample]=1
            elif (classification_array[isample,6] > lai_high_threshold_good_fraction) and (classification_array[isample,1] < lai_low_threshold_good_fraction) and (classification_array[isample,2] < sim_params.nan_good_threshold):
                # Semi-good
                classification_vector[isample]=4
            elif (classification_array[isample,0] < lai_high_threshold_bad_fraction and classification_array[isample,1] > lai_low_threshold_bad_fraction):
                # Semi-bad
                classification_vector[isample]=2
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
            #endif
        #endfor

    elif sim_params.timeseries_flag == "LAI_MEAN_RMSD":

        for isample in range(nsamples):

            # Set three thresholds.  "Bad" is seperate, the same as LAI_MEAN1.
            # 0 : Fraction of annual LAI_MEAN values above an ideal threshold
            # 1 : Fraction of annual LAI_MEAN values below a different threshold
            # 2 : Fraction of annual LAI_MEAN values are NaNs
            # 3 : Fraction of timesteps when LAI_MEAN changes by more than 0.2
            # 4 : Fraction of annual LAI_MEAN values are zero
            # 5 : Fraction of annual LAI_MEAN values are close to zero
            # 6 : Fraction of annual LAI_MEAN values above a good threshold
            # 7 : RMSD of the annual LAI_MEAN values

            if (classification_array[isample,2] > sim_params.nan_bad_threshold) or (classification_array[isample,4] > sim_params.nan_bad_threshold) or (classification_array[isample,5] > sim_params.nan_bad_threshold):
                # Bad
                classification_vector[isample]=1
            elif (classification_array[isample,7] > sim_params.lai_high_threshold_rmsd):
                # Semi-bad
                classification_vector[isample]=2
            elif (classification_array[isample,7] > sim_params.lai_medium_threshold_rmsd):
                # Okay
                classification_vector[isample]=3
            elif (classification_array[isample,7] > sim_params.lai_low_threshold_rmsd):
                # Semi-good
                classification_vector[isample]=4
            elif (classification_array[isample,7] <= sim_params.lai_low_threshold_rmsd):
                # Good
                classification_vector[isample]=5
                #endif
            else:
                # Should not be here
                print("Not sure how to classify this pixel in LAI_MEAN_RMSD!")
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        #endfor

    elif sim_params.timeseries_flag == "LAI_MEAN_BIMODAL":

        for isample in range(nsamples):

            # For this classification, only use four colors.
            # "Good": all criteria are positive.
            # "Semi-good": Not used
            # "OK" : Everything else.
            # "Semi-bad": A lot of large jumps in LAI.
            # "Bad": Too many NaNs.
            # 0 : Fraction of annual LAI_MEAN values above a threshold
            # 1 : Fraction of annual LAI_MEAN values below a different threshold
            # 2 : Fraction of annual LAI_MEAN values are NaNs.
            # 3 : Number of times LAI_MEAN changes by more than a threshold

            # The last criteria is essential.  If there are a lot of NaNs, we automatically classify it as bad, since it can
            # mess up the rest of the calculations.

            if classification_array[isample,0] > lai_high_threshold_good_fraction and classification_array[isample,2] <= sim_params.nan_good_threshold and classification_array[isample,1] < lai_low_threshold_good_fraction:
                # Good
                classification_vector[isample]=5
            elif (classification_array[isample,2] > sim_params.nan_bad_threshold):
                # Bad
                classification_vector[isample]=1
#            elif (classification_array[isample,0] > 0.4 and classification_array[isample,1] > 0.4):
            elif (classification_array[isample,3] > sim_params.lai_diff_bad_threshold):
                # Semi-bad
                classification_vector[isample]=2
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
            #endif
        #endfor

    elif sim_params.timeseries_flag == "HEIGHT":

        for isample in range(nsamples):

            # For this classification, only use three colors.
            # "Good": 98% above a certain threshold
            # "Semi-good": Not used
            # "OK" : Everything else.
            # "Semi-bad": Not used.
            # "Bad": 20% below a certain threshold
            # 0 : Maximum value.  
            # 1 : Minimum value.  
            # 2 : Fraction of values below a threshold
            # 3 : Penalizing drops

            # I pick these thresholds by looking at the historgram
            # after a run.
            # Ok?
            classification_vector[isample]=3
            if classification_array[isample,0] < sim_params.height_threshold*1.5:
                # Bad.  Maximum height should be more than 50% of the starting
                # value!
                classification_vector[isample]=1
            elif classification_array[isample,3] < 100:
                # Bad...maximum 100 years between cuts
                classification_vector[isample]=1
            # Oddly behaving pixels
            #elif not np.isnan(classification_array[isample,5]) and classification_array[isample,5] < -0.5:
                # correlation between GPP and LAI
            #    classification_vector[isample]=2
            elif not np.isnan(classification_array[isample,7]) and classification_array[isample,7] < -0.5:
                # correlation between NPP and height
                classification_vector[isample]=2
            # These are "good" pixels
            elif classification_array[isample,4] == 0:
                #print("ijfeow ",isample,classification_array[isample,8],classification_array[isample,9])
                # No breaks.
                if classification_array[isample,8] > 0.8:
                    # Lot of values above a threshold
                    classification_vector[isample]=5
                elif classification_array[isample,9] > 0.8:
                    # lot of values above a lower threshold
                    classification_vector[isample]=4
                #endif
                
            #endif
        #endfor

    elif sim_params.timeseries_flag == "TWBR":

        for isample in range(nsamples):

            # For this classification, only use three colors.
            # "Good": all criteria are positive.
            # "Semi-good": Not used
            # "OK" : Everything else.
            # "Semi-bad": Not used.
            # "Bad": All criteria are negative.
            # 0 : Fraction of points above a certain threshold (this is bad)
            # 1 : Fraction of points below a certain threshold (this is good)

            if classification_array[isample,0] < sim_params.twbr_low_threshold_good_fraction and not (classification_array[isample,0] > sim_params.twbr_high_threshold_bad_fraction):
                # Good
                classification_vector[isample]=5
            elif (classification_array[isample,0] > sim_params.twbr_high_threshold_bad_fraction) and not (classification_array[isample,0] < sim_params.twbr_low_threshold_good_fraction):
                # Bad
                classification_vector[isample]=1
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
            #endif
        #endfor

    elif sim_params.timeseries_flag == "N_RESERVES":

        for isample in range(nsamples):

            # For this classification, only use three colors.
            # "Good": Criterion 1 is above a certain threshold and criterion 0 is 0
            # "Semi-good": Criterion 1 is above a slightly lower threshold and criterion 0 is 0
            # "OK" : Everything else.
            # "Semi-bad": Not used.
            # "Bad": Criterion 0 has a value of 1
            # 0 : Presence of a 50-year stretch with all decennial means increasing and outside previous variance (1 True, 0 False)
            # 1 : Fraction of decennial means that fall within the average standard
            #     deviation of one decade from the whole timeseries mean
        
            # 2 : Not used.

            if classification_array[isample,0] == 0.0 and (classification_array[isample,1] > sim_params.nreserves_good_fraction):
                # Good
                classification_vector[isample]=5
            elif classification_array[isample,0] == 0.0 and (classification_array[isample,1] > sim_params.nreserves_okay_fraction):
                # Resonably good
                classification_vector[isample]=4
            elif classification_array[isample,0] == 1.0:
                # Bad
                classification_vector[isample]=1
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
            #endif
        #endfor

    elif sim_params.timeseries_flag == "TOTAL_M_c":

        for isample in range(nsamples):

            # For this classification, only use two colors.
            # "Good": Not used.
            # "Semi-good": Not used.
            # "OK" : Everything else.
            # "Semi-bad": Not used.
            # "Bad": Criterion 0 has a value of 1
            # 0 : Presence of a 300-year stretch with all decennial means 
            #     increasing and outside previous variance (1 True, 0 False)
            # 1 : Not used
            # 2 : Not used.

            if classification_array[isample,0] == 1.0:
                # Bad
                classification_vector[isample]=1
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
            #endif
        #endfor

    else:
        print("Cannot yet classify points with this flag in classify_observations!")
        print(sim_params.timeseries_flag)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    return classification_vector
#enddef

# Choose some of the classified points at random and plot them as timeseries
def plot_classified_observations(classification_vector,timeseries_array,timeseries_lat,timeseries_lon,sim_params,syear,eyear,classification_array,classified_points):

        

    # Based on the graphs from plot_classified_observations subroutines,
    # assign a color to each level.  colors[0] is that given to pixels
    # with not enough veget_max to be included.
    level_colors=["darkgray","crimson","orange","gold","yellowgreen","green"]
    colors_by_level=[0,1,2,3,4,5] # these are the actual colors per level

    xvalues=range(1,timeseries_array.shape[1]+1)

    # I need a vector of colors.  Not ideal
    # since all graphs will look the same.  But I need to be able to 
    # identify latitude and longitude and distinguish between the lines.
    colors=["black","darkgray","lightgray"]
    icolor=0

    # I would like to make the background of the plot the classification color.  So I want to use only gray
    # colored lines and change symbols and linestyles to differentiate.
    linestyles=["solid","dotted","dashed","dashdot"]
    markersymbols=["v","^","."]

    for ilevel in colors_by_level:

        # This should be for gray pixels.
        if ilevel == 0:
            continue
        #endif

        print("On classification level: ",ilevel)

        # How many of these do we have?
        temp_array=np.where(classification_vector[:] == ilevel, True, False)
        ntotal=np.sum(temp_array)
        print("Have a total of {} timeseries at this level.".format(ntotal))

        # Either we have timeseries to plot, or we don't.  If don't, we create
        # a spacefiller graph, since I want to make collections of graphs in 
        # an automated fashion afterwards and not having a file messes up the
        # alignment.
        
        # Create the plots.  I want to have a lot of room at the bottom for
        # legends that list the latitude and longitude of each point.

        if ntotal != 0:

            # I am going to try to use a clustering algorithm to group
            # all of these timeseries into 10 different clusters, and then
            # choose one member of each cluster at random.  In theory, I
            # hope this shows the widest range of different behaviors.



            fig=plt.figure(2,figsize=(13, 8))
            gs = gridspec.GridSpec(3, 1, height_ratios=[4.7,1,1])
            ax1=plt.subplot(gs[0])

           # I want to plot 10 of these timeseries.
            npoints=10
            classified_series=timeseries_array[classification_vector == ilevel]
            classified_lat=timeseries_lat[classification_vector == ilevel]
            classified_lon=timeseries_lon[classification_vector == ilevel]
            classified_array=classification_array[classification_vector == ilevel,:]

            # I need these points to be as distinct as possible
            if npoints >  classified_series.shape[0]:

                # Take all the points
                selected_timeseries=classified_series.copy()
                selected_lat=classified_lat.copy()
                selected_lon=classified_lon.copy()
                selected_class_criteria=classified_array.copy()
                #selected_points=np.arange(0,npoints)
                #selected_points=np.where(selected_points >= ntotal, 0, selected_points)
                #selected_points=list(selected_points)
            elif 2*npoints >= classified_series.shape[0]:

                # Just take the first npoints points.  Not really enough
                # points to cluster, and seems to cause problems in the
                # kmeans algorithm below.
                selected_timeseries=classified_series[0:npoints,:].copy()
                selected_lat=classified_lat[0:npoints].copy()
                selected_lon=classified_lon[0:npoints].copy()
                selected_class_criteria=classified_array.copy()
            else:

                # Extract all the timeseries with this classification level
                # Unfortunately, kmeans, and ML algorithms in general, can't
                # deal with NaN values.  Sometimes such values pop up in these
                # timeseries.  What do we do?
                temp_array=np.isnan(classified_series.copy())
                if temp_array.any():

                    # Just take some at random
                    # Above we make sure that we have more points in classified_series than
                    # npoints, so we don't have to worry about an infinite loop.
                    # Still, check to be sure.
                    selected_timeseries=np.zeros((npoints,classified_series.shape[1]),dtype=float)
                    selected_lat=np.zeros((npoints),dtype=float)
                    selected_lon=np.zeros((npoints),dtype=float)
                    selected_class_criteria=np.zeros((npoints,classification_array.shape[1]),dtype=float)

                    selected_points=[]
                    icounter=0
                    while len(selected_points) != npoints:
                        random_point=np.random.randint(0,classified_series.shape[0])
                        if random_point not in selected_points:
                            selected_points.append(random_point)
                        #endif
                        icounter=icounter+1
                        if icounter > 100*npoints:
                            print("Trying to pick many points at random!")
                            print("icounter = ",icounter)
                            print("npoints = ",npoints)
                            print("Perhaps increase the limit at this part of the code?")
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        #endif
                    #end while

                    selected_timeseries[:,:]=classified_series[selected_points,:]
                    selected_lat[:]=classified_lat[selected_points]
                    selected_lon[:]=classified_lon[selected_points]
                    selected_class_criteria[:,:]=classified_array[selected_points,:]
                else:
                    # here we can actually do kmeans

                    kmeans = KMeans(n_clusters=npoints,random_state=0).fit(classified_series)
                    cluster_vector=kmeans.labels_

                    selected_timeseries=np.zeros((npoints,classified_series.shape[1]),dtype=float)
                    selected_lat=np.zeros((npoints),dtype=float)
                    selected_lon=np.zeros((npoints),dtype=float)
                    selected_class_criteria=np.zeros((npoints,classification_array.shape[1]),dtype=float)

                    # Now choose one timeseries for each of the clusters
                    for ipoint in range(npoints):

                        cluster_timeseries=classified_series[cluster_vector == ipoint]
                        cluster_lat=classified_lat[cluster_vector == ipoint]
                        cluster_lon=classified_lon[cluster_vector == ipoint]
                        cluster_class_criteria=classified_array[cluster_vector == ipoint,:]

                        if len(cluster_timeseries.shape) != 2:
                            print("Problem with clusters!")
                            print(cluster_timeseries.shape)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        #endif

                        # This happens, but rarely, and for unknown reasons
                        if cluster_timeseries.shape[0] == 0:
                            continue
                        #endif

                        # Need a random integer between 0 and the number of timeseries
                        # we have
                        random_point=np.random.randint(0,cluster_timeseries.shape[0])
                        
                        selected_timeseries[ipoint,:]=cluster_timeseries[random_point,:]
                        selected_lat[ipoint]=cluster_lat[random_point]
                        selected_lon[ipoint]=cluster_lon[random_point]
                        selected_class_criteria[:,:]=cluster_class_criteria[random_point,:]
                    #endfor
                #endif

                


            #endfor
           
            
            # Try to combine all my different legends in one
            legend_axes=[]
            legend_titles=[]
            
            icolor=0
            imarkerstyle=0
            ilinestyle=0
            
            for ipoint in range(selected_timeseries.shape[0]):

                legend_string=create_latlon_string(selected_lat[ipoint],selected_lon[ipoint])
                #### This is just for debugging.
                #filename="{}_Class{}.txt".format(legend_string,ilevel)
                #np.savetxt(filename,selected_class_criteria[ipoint,:])
                #file = open(filename,"a") 
                #for ival,val in enumerate(xvalues):
                #    file.write("{},{}\n".format(val,selected_timeseries[ipoint,ival]))
                #endfor
                ####
                p1=ax1.plot(xvalues,selected_timeseries[ipoint,:], linestyle=linestyles[ilinestyle], marker=markersymbols[imarkerstyle],color=colors[icolor],label=legend_string)
                legend_axes.append(p1)
                legend_titles.append(legend_string)
                # Incrememnt the counters
                icolor=icolor+1
                if icolor == len(colors):
                    icolor=0
                    ilinestyle=ilinestyle+1
                    if ilinestyle == len(linestyles):
                        ilinestyle=0
                        imarkerstyle=imarkerstyle+1
                    #endif
                #endif

            #endfor

            #ax2 =plt.subplot(gs[2])

            # Add the legend, change the title, do some cleaning up
            labels = [fill(l, 30) for l in legend_titles]
            #ax1.legend(legend_axes,labels,bbox_to_anchor=(0,0,1,1), loc="lower left",mode="expand", borderaxespad=0, ncol=3,fontsize='large') 
            plt.legend(bbox_to_anchor=(0,-0.5,1,1), loc="lower left",mode="expand", borderaxespad=0, ncol=3,fontsize='large')
            plt.xlabel(r"""Time [yr]""", fontsize=20)
            plt.ylabel(r"""{}""".format(sim_params.ylabel), fontsize=20)
            
            ax1.set_xlim(syear,eyear)
            #ax2.axis('off')
            ax1.set_facecolor(level_colors[colors_by_level[ilevel]])
            
            ax1.set_title("Randomly-selected examples of the {} timeseries classified as {}".format(ntotal,ilevel),fontsize=16)
            
            ax2=plt.subplot(gs[2])
            ax2.text(0.1,0.02,sim_params.plot_subtitle[ilevel-1],transform=fig.transFigure,fontsize=12)
            ax2.set_axis_off()
            fig.savefig(sim_params.classified_timeseries_filename.replace("XXXX","{}".format(ilevel)),dpi=300)
            plt.close(fig)

            # Record all of these timeseries
            if sim_params.print_all_ts:
                all_classifications=np.zeros((classification_array.shape[1],ntotal))
                all_timeseries=np.zeros((len(xvalues),ntotal))
                #all_timeseries[:,0]=xvalues # want to create a dataframe with all values

                timeseries_names=[]
                #timeseries_names.append("Year")
                for ipoint in range(classified_series.shape[0]):
                    legend_string=create_latlon_string(classified_lat[ipoint],classified_lon[ipoint])
                    timeseries_names.append(legend_string)

                    all_timeseries[:,ipoint]=classified_series[ipoint,:]
                    all_classifications[:,ipoint]=classification_array[ipoint,:]
                #endfor
                
                timeseriesdf=pd.DataFrame(data=all_timeseries,columns=timeseries_names)
                timeseriesdf.to_csv(path_or_buf="classified_ts_level_{}.csv".format(ilevel),sep=",")

                classificationdf=pd.DataFrame(data=all_classifications,columns=timeseries_names)
                classificationdf.to_csv(path_or_buf="classication_criteria_ts_level_{}.csv".format(ilevel),sep=",")                         
                

            #endif
     
        else:
            fig=plt.figure(2,figsize=(13, 8))
            ax1 = fig.add_subplot(111)
            plt.text(0.2, 0.5, 'Zero pixels classified as level {}.'.format(ilevel), color="black", size=24)
            ax1.set_axis_off()
            # This is the only way I could figure out how to set the background color to be a lighter value
            # of the classification color
            colorrgb=mpl_colors.to_rgba(level_colors[colors_by_level[ilevel]])
            colorlist=list(colorrgb)
            colorlist[3]=0.3 # change the alpha value to be more transparent
            fig.savefig(sim_params.classified_timeseries_filename.replace("XXXX","{}".format(ilevel)),facecolor=tuple(colorlist),dpi=300)
            plt.close(fig)
      #endif

        

    #endfor
    
    # Print out a file with all the points and their classification, just in case
    all_timeseries=np.zeros((len(timeseries_lat),3))
    columnnnames=("Latitude","Longitude","Level")
    for ipoint in range(len(timeseries_lat)):
        all_timeseries[ipoint,0]=timeseries_lat[ipoint]
        all_timeseries[ipoint,1]=timeseries_lon[ipoint]
        all_timeseries[ipoint,2]=classification_vector[ipoint]
    #endfor
    timeseriesdf=pd.DataFrame(data=all_timeseries,columns=columnnnames)
    timeseriesdf.to_csv(path_or_buf="all_classified_points_{}.csv".format(sim_params.cmap_identifier),sep=",")

    ######
    timeseriesdf=pd.DataFrame(data=all_timeseries,columns=columnnnames)
    timeseriesdf.to_csv(path_or_buf="all_classified_points_{}.csv".format(sim_params.cmap_identifier),sep=",")


    #####
    # I want to plot a global timeseries by combining all these pixels
    xmin=1
    xmax=sim_params.eyear-sim_params.syear+1
    global_timeseries=np.zeros((timeseries_array.shape[1]),dtype=np.float32)
    ntimeseries=timeseries_array.shape[0]
    if sim_params.global_operation == "simplesum":
        for itimeseries in range(ntimeseries):
            global_timeseries[:]=global_timeseries[:]+timeseries_array[itimeseries,:]
        #endfor
        axis_title="Global timeseries, where {} classified pixels are summed together".format(ntimeseries)
    elif sim_params.global_operation == "weightedpftsum":
        for itimeseries in range(ntimeseries):
            global_timeseries[:]=global_timeseries[:]+timeseries_array[itimeseries,:]*classified_points[ipoint].veget_max[:]*classified_points[ipoint].contfrac*classified_points[ipoint].area
        #endfor
        axis_title="Global timeseries for PFT{}, where {} classified pixels are summed together".format(sim_params.pft_selected+1,ntimeseries)
    elif sim_params.global_operation == "weightedpftave":
        total_area=np.zeros((len(global_timeseries)))
        for itimeseries in range(ntimeseries):
            global_timeseries[:]=global_timeseries[:]+timeseries_array[itimeseries,:]*classified_points[ipoint].veget_max[:]*classified_points[ipoint].contfrac*classified_points[ipoint].area
            total_area[:]=total_area[:]+classified_points[ipoint].veget_max[:]*classified_points[ipoint].contfrac*classified_points[ipoint].area
        #endfor
        global_timeseries[:]=global_timeseries[:]/total_area[:]
        axis_title="Global timeseries for PFT{}, where {} classified pixels are summed together".format(sim_params.pft_selected+1,ntimeseries)

    elif sim_params.global_operation == "weightedareasum":
        for itimeseries in range(ntimeseries):
            global_timeseries[:]=global_timeseries[:]+timeseries_array[itimeseries,:]*classified_points[ipoint].contfrac*classified_points[ipoint].area
        #endfor
        axis_title="Global timeseries, where {} classified pixels are summed together".format(ntimeseries)
    else:
        print("Not sure how to deal with this global operation!")
        print(sim_params.global_operation)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif
    fig=plt.figure(2,figsize=(13, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[9,1])
    ax1=plt.subplot(gs[0])
    p1=ax1.plot(xvalues[xmin-1:xmax-1],global_timeseries[xmin-1:xmax-1]*sim_params.global_plot_scale, linestyle="-", marker="o",color="black",label="Global")
    plt.xlabel(r"""Time [yr]""", fontsize=20)
    plt.ylabel(r"""{} [{}]""".format(sim_params.timeseries_variable,sim_params.global_plot_units), fontsize=20)
            
    #ax1.set_xlim(syear,eyear)
    #ax2.axis('off')
    #ax1.set_facecolor(level_colors[colors_by_level[ilevel]])
            
    ax1.set_title(axis_title,fontsize=16)
            
    #ax2=plt.subplot(gs[2])
    #ax2.text(0.1,0.02,sim_params.plot_subtitle[ilevel-1],transform=fig.transFigure,fontsize=12)
    #ax2.set_axis_off()
    fig.savefig(sim_params.classified_timeseries_filename.replace("XXXX","{}".format("_global")),dpi=300)
    plt.close(fig) 

    return level_colors,colors_by_level
#enddef


# I want a stringn that looks like 34.5N,24.5E
def create_latlon_string(lat,lon):
    if lat < 0.0:
        latstring="{}S".format(abs(lat))
    else:
        latstring="{}N".format(abs(lat))
    #endif
    if lon < 0.0:
        lonstring="{}W".format(abs(lon))
    else:
        lonstring="{}E".format(abs(lon))
    #endif
    return "{},{}".format(latstring,lonstring)
#endif

# I want to parse a string that looks like 35N,75N,-20E,40E
# The output is four fields: northern, southern, western, and eastern
# borders of a rectangle, in degrees north and east (negative values 
# indicate south and west)
def parse_latlon_string(latlon_string):

    if latlon_string is None:
        print("-- Latlon string is None.  Returning None for all boundaries.")
        return None,None,None,None
    #endif

    case1_ns=re.compile('(.+)N,(.+)N,.+,.+')
    case2_ns=re.compile('(.+)S,(.+)N,.+,.+')
    case3_ns=re.compile('(.+)S,(.+)S,.+,.+')
    #case4_ns=re.compile('()N,()S') # not possible!
    
    if case1_ns.search(latlon_string):
        m=case1_ns.search(latlon_string)
        nlat_window=float(m[2])
        slat_window=float(m[1])
    elif case2_ns.search(latlon_string):
        m=case2_ns.search(latlon_string)
        nlat_window=float(m[2])
        slat_window=-float(m[1])
    elif case3_ns.search(latlon_string):
        m=case3_ns.search(latlon_string)
        nlat_window=-float(m[2])
        slat_window=-float(m[1])
    else:
        print("Lat/lon string doesn't make sense!  ",latlon_string)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    case1_ew=re.compile('.+,.+,(.+)E,(.+)E')
    case2_ew=re.compile('.+,.+,(.+)W,(.+)E')
    case3_ew=re.compile('.+,.+,(.+)W,(.+)W')
    #case4_ew=re.compile('()N,()S') # possible...but I ignore it
    
    if case1_ew.search(latlon_string):
        m=case1_ew.search(latlon_string)
        wlon_window=float(m[1])
        elon_window=float(m[2])
    elif case2_ew.search(latlon_string):
        m=case2_ew.search(latlon_string)
        wlon_window=-float(m[1])
        elon_window=float(m[2])
    elif case3_ew.search(latlon_string):
        m=case3_ew.search(latlon_string)
        wlon_window=-float(m[1])
        elon_window=-float(m[2])
    else:
        print("Lat/lon string doesn't make sense!  ",latlon_string)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    print("Plotting timeseries in the following window: {}N,{}N,{}E,{}E".format(slat_window,nlat_window,wlon_window,elon_window))
    return nlat_window,slat_window,wlon_window,elon_window
#endif

# Choose some of the classified points at random and plot them as timeseries
def map_classified_pixels(classification_vector,timeseries_lat,timeseries_lon,level_colors,colors_by_level,sim_params):

    input_file_name=sim_params.condensed_nc_file_name
    pft_selected=sim_params.pft_selected
    veget_max_threshold=sim_params.veget_max_threshold    
    veget_max_name=sim_params.veget_max_name

    print("Creating a map.")

    # Need to get the map from the output file.  Get a blank grid, really.
    srcnc = NetCDFFile(input_file_name,"r")
    timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)
    
    nlats=len(srcnc[latcoord][:])
    nlons=len(srcnc[loncoord][:])

    cgrid = Grid(lat = srcnc[latcoord][:], lon = srcnc[loncoord][:])
    # on this grid, I create a map with only five levels.
    map_vals=np.zeros((nlats,nlons),dtype=np.int32)*np.nan

    # Now I fill in all values for which there is vegetation present, using
    # the color for the low-threshold pixels.  The rest that are classified
    # will get changed.
    print("Filling in the base map.")
    for ilat in range(nlats):
        for ilon in range(nlons):

            # This will depend on the classification scheme.  
            if sim_params.timeseries_flag in ("LAI_MEAN1","N_RESERVES","TOTAL_M_c","HEIGHT","LAI_MEAN_RMSD"):
                if not np.ma.is_masked(srcnc[veget_max_name][0, pft_selected, ilat, ilon]):
                    # Leave it blank if no PFT is there
                    if srcnc[veget_max_name][0, pft_selected, ilat, ilon] == 0.0:
                        continue
                    #endif

                    # If it's below a threshold, give it a color
                    if srcnc[veget_max_name][0, pft_selected, ilat, ilon] < veget_max_threshold:
                        map_vals[ilat,ilon]=colors_by_level[0]
                    #endif
                
                #endif
            else:
                if sim_params.timeseries_flag not in ["TWBR"]:
                    print("Do not know how to create a base map for this flag in map_classified_pixels!")
                    print(sim_params.timeseries_flag)
                    sys.exit(1)
                #endif
            #endif
    #endfor

    print("Filling in the classified pixels.")
    # Now loop over the classified points and color them.
    for ipoint in range(len(classification_vector)):
        ilat=np.where(srcnc[latcoord][:] == timeseries_lat[ipoint])
        ilon=np.where(srcnc[loncoord][:] == timeseries_lon[ipoint])
        map_vals[ilat,ilon]=colors_by_level[classification_vector[ipoint]]
    #endfor

    custom_cmap=mpl.colors.ListedColormap(level_colors)

    # Now try to map
    cgrid.plotmap(map_vals,title=sim_params.classified_map_title,filename=sim_params.classified_map_filename,dpi=300,levels=len(level_colors),ptype = "custom_cmap",custom_cmap=custom_cmap,plot_cbar=True,vmin=-0.5,vmax=len(level_colors)-0.5)



#enddef

# This routine takes a while.  So after the extraction I print all the
# data to a new file.
def extract_and_calculate_classifiction_metrics(sim_params):

    timeseries_flag=sim_params.timeseries_flag
    pft_selected=sim_params.pft_selected
    veget_max_threshold=sim_params.veget_max_threshold
    input_file_name=sim_params.condensed_nc_file_name

    print("Extracting a timeseries and creating a classification array for {} and the flag {}.".format(input_file_name,timeseries_flag))
    lat_list=[]
    lon_list=[]

    icounter=0

    # Loop through every pixel in the output file and check to see
    # if this timeseries meets our criteria
    srcnc = NetCDFFile(input_file_name,"r")
    timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)

    # I need to allocate some arrays, so first go through and see if a pixel meets our
    # criteria.  Save the indices so that we can quickly loop through it to extract
    # the data the second time.  Except the data extraction takes a while, 
    # probably because I have a subroutine call for every point?  Unsure.
    npoints=0
    for ilat,vlat in enumerate(srcnc[latcoord][:]):
        
        # Scanning latitude takes a while for higher resolution grids.
        # Don't even bother if it's outside of the window where we want to print
        # timeseries.
        if sim_params.print_all_ts:
            if vlat > sim_params.nlat_window or vlat < sim_params.slat_window:
                continue
            #endif
        #endif

        print("Scanning latitude for points: ",vlat)
        for ilon,vlon in enumerate(srcnc[loncoord][:]):
    
            # Scanning longitude also takes a while.
            if sim_params.print_all_ts:
                if vlon > sim_params.elon_window or vlon < sim_params.wlon_window:
                    continue
                #endif
            #endif

            if sim_params.do_test:
                if npoints > sim_params.ntest_points:
                    continue
                #endif
            #endif

            if keep_grid_point(srcnc,pft_selected,ilat,ilon,veget_max_threshold,timeseries_flag,sim_params.veget_max_name):
                npoints=npoints+1
                lat_list.append(ilat)
                lon_list.append(ilon)
            #endif
    
            #endif
        #endfor
    #endfor
    print("Found {} timeseries!".format(npoints))
    if npoints == 0:
        print("Stopping.  Nothing to analyze.")
        sys.exit(1)
    #endif

    # Now we allocate all our arrays
    timeseries_array=np.zeros((npoints,len(srcnc[timecoord][:])))
    timeseries_lat=np.zeros((npoints))
    timeseries_lon=np.zeros((npoints))

    # These may not be used, depending on the comparison we are doing.  But
    # one of the subroutines will crash if we don't give it some value.
    #ind_upper_threshold=np.nan
    #lai_bad_threshold=np.nan

    # Allocate one last array, and set some values depending on what classification we are
    # doing
    if timeseries_flag in ("LAI_MEAN1","LAI_MEAN_BIMODAL"):
        classification_array=np.zeros((npoints,7))

        # How many different criteria?
        # 0 : Fraction of annual LAI_MEAN values above an ideal threshold
        # 1 : Fraction of annual LAI_MEAN values below a different threshold
        # 2 : Fraction of annual LAI_MEAN values are NaNs
        # 3 : Fraction of timesteps when LAI_MEAN changes by more than 0.2
        # 4 : Fraction of annual LAI_MEAN values are zero
        # 5 : Fraction of annual LAI_MEAN values are close to zero
        # 6 : Fraction of annual LAI_MEAN values above a good threshold

    elif timeseries_flag in ("LAI_MEAN_RMSD"):
        classification_array=np.zeros((npoints,8))

        # How many different criteria?
        # 0 : Fraction of annual LAI_MEAN values above an ideal threshold
        # 1 : Fraction of annual LAI_MEAN values below a different threshold
        # 2 : Fraction of annual LAI_MEAN values are NaNs
        # 3 : Fraction of timesteps when LAI_MEAN changes by more than 0.2
        # 4 : Fraction of annual LAI_MEAN values are zero
        # 5 : Fraction of annual LAI_MEAN values are close to zero
        # 6 : Fraction of annual LAI_MEAN values above a good threshold
        # 7 : RMSD of annual LAI_MEAN values

    elif timeseries_flag == "LAI_MEAN2":
        classification_array=np.zeros((npoints,4))

        # How many different criteria?
        # 0 : Maximum value
        # 1 : Number of times below a threshold
        # 2 : Number of times IND is above a certain threshold
        # 3 : Number of NaNs present in the LAI_MEAN timeseries


    elif sim_params.timeseries_flag == "HEIGHT":

        classification_array=np.zeros((npoints,10))

        # 0 : Maximum value.  
        # 1 : Minimum value.  
        # 2 : Fraction of values below a threshold
        # 3 : Penalizing drops
        # 4 : Rewarding growth


    elif sim_params.timeseries_flag == "TWBR":

        classification_array=np.zeros((npoints,2))

        # 0 : Fraction of points above a certain threshold (this is bad)
        # 1 : Fraction of points below a certain threshold (this is good)

    elif sim_params.timeseries_flag == "N_RESERVES":

        classification_array=np.zeros((npoints,3))

        # 0 : Presence of a 50-year stretch with all decennial means increasing and outside previous variance (1 True, 0 False)
        # 1 : Fraction of decennial means that fall within the average standard
        #     deviation of one decade from the whole timeseries mean
        # 2 : Not used.

    elif sim_params.timeseries_flag == "TOTAL_M_c":

        classification_array=np.zeros((npoints,3))

        # 0 : Presence of a 300-year stretch with all decennial means increasing and outside previous variance (1 True, 0 False)
        # 1 : Not used
        # 2 : Not used.

    else:

        print("I don't know how to create criteria for this flag in extract_and_calculate_classifiction_metrics!")
        print(timeseries_flag)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    #endif


    # Now we loop through all the points, pick out our timeseries, and then compute
    # the metrics
    classified_points=[]
    for ilat,ilon,ipoint in zip(lat_list,lon_list,range(npoints)):
  
        if ipoint % 100 == 0:
            print("Extracting and classifying point {}.".format(ipoint))
        #endif
        
        timeseries_lat[ipoint]=srcnc[latcoord][ilat]
        timeseries_lon[ipoint]=srcnc[loncoord][ilon]

        # Now the behavior changes with the flag
        classification_array[ipoint,:],timeseries_array[ipoint,:]=compute_metrics_extracted_pixel(srcnc,sim_params,ilat,ilon)
         
        # Save information about this pixel in a structure
        classified_points.append(classified_pixel_information(timeseries_array[ipoint,:],srcnc[latcoord][ilat],srcnc[loncoord][ilon],srcnc["Areas"][ilat,ilon],srcnc["CONTFRAC"][ilat,ilon],srcnc[sim_params.veget_max_name][:,pft_selected,ilat,ilon]))

    #endif


    
    # Now print out a few summary statistics
    classificationdf=pd.DataFrame(data=classification_array)
    classificationdf.hist(bins=30,figsize=(9,9))
    pl.suptitle("Histogram for each classification variable")
    plt.savefig(sim_params.class_histogram_filename)
    plt.close()

                              
    srcnc.close()

    # Since finding timeseries can take a long time, save what we've found
    #timeseriesdf=pd.DataFrame(data=timeseries_array)
    #timeseriesdf["Latitude"]=timeseries_lat[:]
    #timeseriesdf["Longitude"]=timeseries_lon[:]
    #timeseriesdf.to_csv(path_or_buf="saved_timeseries.csv",index=False)

    return timeseries_array,timeseries_lat,timeseries_lon, classification_array,classified_points

#enddef

def keep_grid_point(srcnc,pft_selected,ilat,ilon,veget_max_threshold,timeseries_flag,veget_max_name):
    # Check to see if, based on the timeseries we are looking for and the PFT, we want
    # to keep this gridpoint.

    if timeseries_flag in ("LAI_MEAN1","LAI_MEAN2","LAI_MEAN_BIMODAL","N_RESERVES","TOTAL_M_c","HEIGHT","LAI_MEAN_RMSD"):
        if not np.ma.is_masked(srcnc[veget_max_name][0, pft_selected, ilat, ilon]):
            if srcnc[veget_max_name][0, pft_selected, ilat, ilon] < veget_max_threshold:
                lkeep=False
            else:
                lkeep=True
            #endif
        else:
            lkeep=False
        #endif
    elif timeseries_flag == "TWBR":
        if not np.ma.is_masked(srcnc["TWBR"][0, ilat, ilon]):
            lkeep=True
        else:
            lkeep=False
        #endif    
    else:
        print('Not sure how you want to extract the timeseries in keep_grid_point!')
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    return lkeep

#endif


def compute_metrics_extracted_pixel(srcnc,sim_params,ilat,ilon):

    if sim_params.timeseries_flag in ("LAI_MEAN1","LAI_MEAN_BIMODAL"):
        

        timeseries=srcnc[sim_params.timeseries_variable][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((7))

        # 0 : Fraction of LAI_MEAN values above an ideal threshold
        temp_array=np.where(timeseries > sim_params.lai_good_threshold, True, False)
        carray[0]=float(np.sum(temp_array))/float(len(temp_array))

        # 1 : Fraction of annual LAI_MEAN values below a different threshold
        temp_array=np.where(timeseries < sim_params.lai_bad_threshold, True, False)
        carray[1]=float(np.sum(temp_array))/float(len(temp_array))

        # 2 : Fraction of annual LAI_MEAN values are NaNs
        temp_array=np.where(np.isnan(timeseries), True, False)
        carray[2]=float(np.sum(temp_array))/float(len(temp_array))

        # 3 : Number of times LAI_MEAN changes by more than a threshold
        timestep_diff=[]
        for itime in range(1,len(timeseries)):
            if not np.isnan(timeseries[itime]) and not np.isnan(timeseries[itime-1]):
                timestep_diff.append(float(timeseries[itime]-timeseries[itime-1]))
            #endif
        #endfor
        timestep_diff=np.asarray(timestep_diff)
        temp_array=np.where(timestep_diff > sim_params.lai_diff_bad_threshold, True, False)
        if len(temp_array) != 0:
            carray[3]=float(np.sum(temp_array))/float(len(temp_array))
        #endif

        # 4 : Fraction of annual LAI_MEAN values are zero
        temp_array=np.where(timeseries==0.0, True, False)
        carray[4]=float(np.sum(temp_array))/float(len(temp_array))

        # 5 : Fraction of annual LAI_MEAN values are close to zero
        temp_array=np.where(timeseries < sim_params.zero_threshold, True, False)
        carray[5]=float(np.sum(temp_array))/float(len(temp_array))

        # 6 : Fraction of LAI_MEAN values above a good threshold
        temp_array=np.where(timeseries > sim_params.lai_good_threshold*sim_params.good_growth, True, False)
        carray[6]=float(np.sum(temp_array))/float(len(temp_array))

    elif sim_params.timeseries_flag in ("LAI_MEAN_RMSD"):
        

        timeseries=srcnc[sim_params.timeseries_variable][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((8))

        # 0 : Fraction of LAI_MEAN values above an ideal threshold
        temp_array=np.where(timeseries > sim_params.lai_good_threshold, True, False)
        carray[0]=float(np.sum(temp_array))/float(len(temp_array))

        # 1 : Fraction of annual LAI_MEAN values below a different threshold
        temp_array=np.where(timeseries < sim_params.lai_bad_threshold, True, False)
        carray[1]=float(np.sum(temp_array))/float(len(temp_array))

        # 2 : Fraction of annual LAI_MEAN values are NaNs
        temp_array=np.where(np.isnan(timeseries), True, False)
        carray[2]=float(np.sum(temp_array))/float(len(temp_array))

        # 3 : Number of times LAI_MEAN changes by more than a threshold
        timestep_diff=[]
        for itime in range(1,len(timeseries)):
            if not np.isnan(timeseries[itime]) and not np.isnan(timeseries[itime-1]):
                timestep_diff.append(float(timeseries[itime]-timeseries[itime-1]))
            #endif
        #endfor
        timestep_diff=np.asarray(timestep_diff)
        temp_array=np.where(timestep_diff > sim_params.lai_diff_bad_threshold, True, False)
        if len(temp_array) != 0:
            carray[3]=float(np.sum(temp_array))/float(len(temp_array))
        #endif

        # 4 : Fraction of annual LAI_MEAN values are zero
        temp_array=np.where(timeseries==0.0, True, False)
        carray[4]=float(np.sum(temp_array))/float(len(temp_array))

        # 5 : Fraction of annual LAI_MEAN values are close to zero
        temp_array=np.where(timeseries < sim_params.zero_threshold, True, False)
        carray[5]=float(np.sum(temp_array))/float(len(temp_array))

        # 6 : Fraction of LAI_MEAN values above a good threshold
        temp_array=np.where(timeseries > sim_params.lai_good_threshold*sim_params.good_growth, True, False)
        carray[6]=float(np.sum(temp_array))/float(len(temp_array))

        # 7 : RMSD of annual LAI_MEAN values
        timeseries_mean=np.mean(timeseries)
        rmsd=0.0
        for itime,rtime in enumerate(timeseries):
            rmsd=rmsd+(timeseries_mean-rtime)**2
        #endfor
        rmsd=np.sqrt(rmsd/len(timeseries))
        carray[7]=rmsd


    elif sim_params.timeseries_flag == "LAI_MEAN2":

        timeseries=srcnc[sim_params.timeseries_variable][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_ind=srcnc["IND"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((3))

        # 0 : Maximum value.  
        carray[0]=np.nanmax(timeseries)
        # 1 : Number of times below a threshold
        temp_array=np.where(timeseries < sim_params.lai_bad_threshold, True, False)
        carray[1]=np.sum(temp_array)

        # 2 : Number of times IND is above a certain threshold
        temp_array=np.where(timeseries > sim_params.ind_upper_threshold, True, False)
        carray[2]=np.sum(temp_array)

    elif sim_params.timeseries_flag == "HEIGHT":

        timeseries_height=srcnc[sim_params.timeseries_variable][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_ind=srcnc["IND"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_rind=srcnc["RECRUITS_IND"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_lai=srcnc["LAI_MEAN_GS"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_gpp=srcnc["GPP"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)
        timeseries_npp=srcnc["NPP"][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((10))

        # 0 : Maximum value.  
        carray[0]=np.nanmax(timeseries_height)
        # 1 : Minimum value.  
        carray[1]=np.nanmin(timeseries_height)

        # 2 : Fraction of values below a threshold
        temp_array=np.where(timeseries_height < sim_params.height_threshold, True, False)
        carray[2]=float(np.sum(temp_array))/float(len(temp_array))

        ### Extract the longest timeseries
        def extract_longest_timeseries(timeseries_height,timeseries_ind,timeseries_rind):
            sindex=0
            eindex=len(timeseries_ind)
            lbreaks=[]
            lbreaks.append(False)

            # Identify all the breaks
            for i in range(1, len(timeseries_ind)):
                diff=timeseries_ind[i]-timeseries_ind[i-1]
                if diff>0.0 and diff > timeseries_rind[i] and abs(timeseries_height[i]-timeseries_height[i-1]) > 1.0 :
                    lbreaks.append(True)
                else:
                    lbreaks.append(False)
                #endif
            #endif

            # Sometimes there are several breaks in a row.  This is 
            # some behavior I don't understand,
            for i in range(1, len(lbreaks)):
                if lbreaks[i] and lbreaks[i-1]:
                    lbreaks[i-1]=False
                #endif
            #endif

            # If there are no breaks, our task is easy
            nbreaks=np.sum(lbreaks)
            if nbreaks != 0:
                
                sindices=[]
                sindices.append(0)
                eindices=[]
                for ival,lval in enumerate(lbreaks):
                    if lval:
                        sindices.append(ival)
                        eindices.append(ival)
                    #endif
                #endfor
                eindices.append(len(lbreaks))

                #print('start ieoru ',sindices)
                #print('end ieoru ',eindices)

                #for i in range(len(timeseries_ind)):
                
                #    print('fieo ',i,timeseries_height[i],timeseries_ind[i],timeseries_rind[i],lbreaks[i])
                #endfor

                longest_break=-1
                for ibreak in range(len(sindices)):
                    if eindices[ibreak]-sindices[ibreak]>longest_break:
                        longest_break=eindices[ibreak]-sindices[ibreak]
                        sindex=sindices[ibreak]
                        eindex=eindices[ibreak]
                    #endif
                #endif
            #endif

            return sindex,eindex,nbreaks
        #enddef

        sindex,eindex,ndrops=extract_longest_timeseries(timeseries_height,timeseries_ind,timeseries_rind)

        # 3 : Penalizing drops.  I want a function that takes
        # two things into account: closeness of the drop to
        # the threshold value, and the time since the last drop.
        # If the height drops to the threshold value after
        # 200 years, that is not the same thing as if it drops
        # to the threshold after 10 years.
        #last_drop=0
        #penalty=0.0
        #growth=0.0
        #ndrops=0
        #ave_length=0.0
        #longest_stretch=0
        #current_stretch=0
        #for i in range(1, len(timeseries_height)):
        #    if timeseries_height[i] > 10.0 and timeseries_height[i-1]>10.0:
        #        longest_stretch=longest_stretch+1
            #endif
        #    if timeseries_height[i] < 10.0 and timeseries_height[i-1]>10.0:
        #        if current_stretch > longest_stretch:
        #            longest_stretch=current_stretch
                #endif
        #        current_stretch=0
            #endif

        #    value = timeseries_height[i] - timeseries_height[i - 1]
        #    time_since_last_drop=float(i-last_drop)
        #    if value < 0.0:

        #        if time_since_last_drop < 20:
        #            pfactor=3.0
        #        elif time_since_last_drop < 80:
        #            pfactor=2.0
        #        elif time_since_last_drop < 130:
        #            pfactor=1.0
        #        else:
        #            pfactor=0.0
                #endif

                # If we are within 25% of the threshold, reset
                # the counter and apply the penalty.  In theory,
                # a thinning from above can drop the height, but
                # is not a cause for concern.
                #if timeseries_ind[i]-timeseries_ind[i-1]>0.0 and timeseries_rind[i] != (timeseries_ind[i]-timeseries_ind[i-1]):
                #    print("Regrozth? ind ",timeseries_ind[i-1],timeseries_ind[i])
                #    print("Regth? height ",timeseries[i-1],timeseries[i])
                #    print("Regth? rind ",timeseries_rind[i-1],timeseries_rind[i])

        #        closeness_to_threshold=timeseries_height[i]/sim_params.height_threshold
         #       if 1.25 > closeness_to_threshold and abs(value) > 1.0:
       #             penalty=penalty+pfactor*abs(value)
       #             ave_length=ave_length+time_since_last_drop
       #             last_drop=i
       #             ndrops=ndrops+1
       #             if timeseries_height[i-1] > 5.0:
       #                 print("DROP TIMESERIES fd ",timeseries_height)
       #                 print(closeness_to_threshold,i,timeseries_height[i],sim_params.height_threshold,ndrops,time_since_last_drop,ave_length)
                    #endif
                #endif
       #     else:
                # Reward growth in the same manner.  Growth at the
                # beginning of a stand is more critical than growth
                # in an old stand.
       #         growth=growth+1.0*np.exp(-0.03465*time_since_last_drop)*abs(value)
            #endif

        #endfor
        #      print("ifjeow ",penalty,growth)
        
        carray[3]=eindex-sindex

        carray[4]=ndrops

        # Test some correlations.  If there are any NaNs, this doesn't work
        nans_lai=np.isnan(timeseries_lai[sindex:eindex])
        nans_gpp=np.isnan(timeseries_gpp[sindex:eindex])
        nans_npp=np.isnan(timeseries_npp[sindex:eindex])
        nans_height=np.isnan(timeseries_height[sindex:eindex])

        if nans_lai.any():
            print("NaNs in LAI! ",timeseries_lai)
        if nans_gpp.any():
            print("NaNs in GPP! ",timeseries_gpp)
        if nans_npp.any():
            print("NaNs in NPP! ",timeseries_npp)
        if nans_height.any():
            print("NaNs in height! ",timeseries_height)


#        if not nans_lai.any() and not nans_gpp.any():
        if True:
            correlation,dummy_val=pearsonr(timeseries_lai[sindex:eindex],timeseries_gpp[sindex:eindex])
            carray[5]=correlation
        else:
            carray[5]=np.nan
        #endif
        if True:
#        if not nans_npp.any() and not nans_gpp.any():
            correlation,dummy_val=pearsonr(timeseries_gpp[sindex:eindex],timeseries_npp[sindex:eindex])
            carray[6]=correlation
        else:
            carray[6]=np.nan
        #endif

        if True:
#        if not nans_npp.any() and not nans_height.any():
            correlation,dummy_val=pearsonr(timeseries_npp[sindex:eindex],timeseries_height[sindex:eindex])
            carray[7]=correlation
        else:
            carray[7]=np.nan
        #endif

        # If there are no drops, try fitting a logarithm growth curve.
        # This didn't work.  So instead, just figure out how many
        # values are above a certain threshold.

        #for i in range(len(timeseries_height)):
        #        print(timeseries_height[i],end=',')
            #endfor
        #    for i in range(len(timeseries_height)):
        #        print(i,end=',')
            #endfor
        #    print(timeseries_height)
        #    sys.exit(1)

        #endif
        # 8 : Fraction of values above a threshold
        temp_array=np.where(timeseries_height > sim_params.height_threshold*3.0, True, False)
        carray[8]=float(np.sum(temp_array))/float(len(temp_array))
        # 9 : Fraction of values above a lower threshold
        temp_array=np.where(timeseries_height > sim_params.height_threshold*2.0, True, False)
        carray[9]=float(np.sum(temp_array))/float(len(temp_array))


        timeseries=timeseries_height

        

    elif sim_params.timeseries_flag == "TWBR":

        timeseries=srcnc[sim_params.timeseries_variable][:, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((2))

        # 0 : Fraction of points above a certain threshold (this is bad)
        temp_array=np.where(abs(timeseries) > abs(sim_params.twbr_high_threshold), True, False)
        carray[0]=np.sum(temp_array)
        # 1 : Fraction of points below a certain threshold (this is good)
        temp_array=np.where(abs(timeseries) < abs(sim_params.twbr_low_threshold), True, False)
        carray[1]=np.sum(temp_array)

    elif sim_params.timeseries_flag == "N_RESERVES":

        def compute_decennial_means(short_timeseries):
            
            if len(short_timeseries) % 10 != 0:
                print("Wrong size timeseries!")
                print(len(short_timeseries))
                print(short_timeseries)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            dec_means=np.zeros((math.floor(len(short_timeseries)/10)))
            dec_variances=np.zeros((math.floor(len(short_timeseries)/10)))
            for idec in range(0,len(dec_means)):
                jdec=idec*10
                sts=short_timeseries[jdec:jdec+10]
                dec_means[idec]=statistics.mean(sts)
                # The internal python statistics package gives an odd error
                # here, which seems to be encountered by others.  So use numpy.
                dec_variances[idec]=np.std(sts)
            #endif
            return dec_means,dec_variances
        #endif

        timeseries=srcnc['RESERVE_M_n'][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)+srcnc['LABILE_M_n'][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((3))

        # 0 : Presence of a 50-year stretch with all decennial means increasing and outside previous variance (1 True, 0 False)

        nslice_years=sim_params.nreserves_ndecades*10
        lfound=False
        # Guessing this will be expensive, so just try doing it every decade first
        for istep in range(0,len(timeseries),10):
            #print("ISTEP ",istep,lfound,istep+nslice_years,timeseries[istep:istep+nslice_years])
            if (istep+nslice_years > len(timeseries)) or lfound:
                continue
            #endif
            dec_means,dec_variances=compute_decennial_means(timeseries[istep:istep+nslice_years])

            # Check here for the patterns.
            lbad=True
            for idec in range(1,len(dec_means)):
                if dec_means[idec] <= dec_means[idec-1]+dec_variances[idec-1]:
                    lbad=False
             #       print("FOUND ",idec,dec_means[idec],dec_means[idec-1],dec_variances[idec-1])
                #endif
            #endif

            if lbad:
                lfound=True
            #endif
            #print("ISTEP END ",lbad,lfound)

       #endfor

        if lfound:
            carray[0]=1.0
        else:
            carray[0]=0.0
        #endif

        # 1 : Fraction of decennial means that fall within the average standard
        #     deviation of one decade from the whole timeseries mean
        
        # Calculate all the decennial means and standard deviations
        dec_means,dec_variances=compute_decennial_means(timeseries[:])
        # Calculate the mean of the whole timeseries
        ts_mean=np.mean(timeseries[:])
        # What is the mean standard deviation?
        sd_mean=np.mean(dec_variances)
        # What fraction of decennial means are within this value from the
        # full timeseries mean?
        temp_array=np.where(abs(dec_means-ts_mean) < sd_mean, True, False)
        carray[1]=float(np.sum(temp_array))/float(len(temp_array))

        # 2 : Not used.

        ###
        # Replace timeseries values for each decade with their means.
        # This makes plots a little easier to decipher.
        # Don't want to do this before because this will mess up our
        # calculation of the variance.
        dec_means,dec_variances=compute_decennial_means(timeseries[:])
        for idec in range(0,len(dec_means)):
            jdec=idec*10
            timeseries[jdec:jdec+10]=dec_means[idec]
        ###

        #print("END CLASSIFY")
        #sys.exit(1)
    elif sim_params.timeseries_flag == "TOTAL_M_c":

        def compute_decennial_means(short_timeseries):
            
            if len(short_timeseries) % 10 != 0:
                print("Wrong size timeseries!")
                print(len(short_timeseries))
                print(short_timeseries)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            dec_means=np.zeros((math.floor(len(short_timeseries)/10)))
            dec_variances=np.zeros((math.floor(len(short_timeseries)/10)))
            for idec in range(0,len(dec_means)):
                jdec=idec*10
                sts=short_timeseries[jdec:jdec+10]
                dec_means[idec]=statistics.mean(sts)
                # The internal python statistics package gives an odd error
                # here, which seems to be encountered by others.  So use numpy.
                dec_variances[idec]=np.std(sts)
            #endif
            return dec_means,dec_variances
        #endif

        timeseries=srcnc['TOTAL_M_c'][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((3))

        # 0 : Presence of a 300-year stretch with all decennial means increasing and outside previous variance (1 True, 0 False)

        nslice_years=sim_params.totalmc_ndecades*10
        lfound=False
        # Guessing this will be expensive, so just try doing it every decade first
        for istep in range(0,len(timeseries),10):
            #print("ISTEP ",istep,lfound,istep+nslice_years,timeseries[istep:istep+nslice_years])
            if (istep+nslice_years > len(timeseries)) or lfound:
                continue
            #endif
            dec_means,dec_variances=compute_decennial_means(timeseries[istep:istep+nslice_years])

            # Check here for the patterns.
            lbad=True
            for idec in range(1,len(dec_means)):
                if dec_means[idec] <= dec_means[idec-1]+np.mean(dec_variances[0:4]):
                    lbad=False
             #       print("FOUND ",idec,dec_means[idec],dec_means[idec-1],dec_variances[idec-1])
                #endif
            #endif

            if lbad:
                lfound=True
            #endif
            #print("ISTEP END ",lbad,lfound)

       #endfor

        if lfound:
            carray[0]=1.0
        else:
            carray[0]=0.0
        #endif

        # 1 : Not used
        # 2 : Not used.

        ###
        # Replace timeseries values for each decade with their means.
        # This makes plots a little easier to decipher.
        # Don't want to do this before because this will mess up our
        # calculation of the variance.
        dec_means,dec_variances=compute_decennial_means(timeseries[:])
        for idec in range(0,len(dec_means)):
            jdec=idec*10
            timeseries[jdec:jdec+10]=dec_means[idec]
        ###

    else:

        print("I don't know how to create criteria for this flag in compute_metrics_extracted_pixel!")
        print(sim_params.timeseries_flag)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    #endif

    return carray,timeseries
#enddef

#################################################
# Folks have asked for a way to find problematic pixels
# more easily.  So this routine prints a NetCDF file with
# all the variables used in the classification as well as
# the classification of each pixel.  People can then visualize
# it by zooming in with Ferret or something.
def create_netcdf_classification_map(classification_vector,timeseries_lat,timeseries_lon,sim_params):

    
    print(sim_params.required_vars)

    # Open up the original file
    srcnc = NetCDFFile(sim_params.condensed_nc_file_name,"r")

    # Get the names of some particular coordinates
    timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)

    # Open up the new file for writing
    output_file=sim_params.condensed_nc_file_name.replace(".nc","_PFT{}_{}_CLASSIFIED.nc".format(sim_params.pft_selected+1,sim_params.timeseries_flag))
    dstnc = NetCDFFile(output_file,"w")

    # copy all the metadata
    dstnc.setncatts(srcnc.__dict__)
    dstnc.setncatts({"NOTE" : "Created after classifying the pixels to allow for easier examination of individual timeseries.  All the variables listed played a role in the classification of the pixel.  Only a single PFT is present, PFT {}.".format(sim_params.pft_selected+1)})

    # Figure out all the dimensions we need.
    dimensions_needed=[]
    if vegetcoord not in srcnc.dimensions:
        print("Could not find a PFT dimension!")
        print(vegetcoord)
        print(srcnc.dimensions)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    try:
        timebounds_name=srcnc[timecoord].bounds
        copy_variables=sim_params.required_vars + [timebounds_name]
    except:
        copy_variables=sim_params.required_vars
    #endtry

    for varname in copy_variables:
        for dim in srcnc[varname].dimensions:
            if dim not in dimensions_needed and dim != vegetcoord:
                print("Keeping dimension: ",varname,dim)
                dimensions_needed.append(dim)
            #endif
        #endfor
    #endfor

    for name, dimension in srcnc.dimensions.items():
        if name in dimensions_needed:
            dstnc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        #endif
    #endfor

    # Create all the variables, including dimension variables.
    # If there is a PFT dimension, eliminate it.

    for name, variable in srcnc.variables.items():
        if name in dimensions_needed or name in copy_variables:
            if vegetcoord in variable.dimensions:
                dimensions=list(variable.dimensions)
                dimensions.remove(vegetcoord)
                idim=variable.dimensions.index(vegetcoord)
                x = dstnc.createVariable(name, variable.datatype, tuple(dimensions))
                # copy variable attributes all at once via dictionary
                dstnc[name].setncatts(srcnc[name].__dict__)
                if len(variable.dimensions) == 4:
                    if idim == 1:
                        dstnc[name][:,:,:]=srcnc[name][:,sim_params.pft_selected,:,:]
                    else:
                        print("Not yet sure how to treat this case!")
                        print("Number of dimensions: ",len(variable.dimensions))
                        print("Dimension of PFT dimension: ",idim)
                        print(name,variable.dimensions)
                        traceback.print_stack(file=sys.stdout)
                        sys.exit(1)
                    #endif
                else:
                    print("Not yet sure how to treat this case!")
                    print("Number of dimensions: ",len(variable.dimensions))
                    print("Dimension of PFT dimension: ",idim)
                    print(name,variable.dimensions)
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            else:
                x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dstnc[name].setncatts(srcnc[name].__dict__)
                dstnc[name][:]=srcnc[name][:]
            #endif
        #endif
    #endfor

    # Now I need to add the classification map
    varname="classification"
    dimensions=(latcoord,loncoord)
    x = dstnc.createVariable(varname, 'f4', dimensions)
    nlats=len(srcnc[latcoord][:])
    nlons=len(srcnc[loncoord][:])
    map_vals=np.zeros((nlats,nlons))*np.nan
    for ipoint in range(len(classification_vector)):
        ilat=np.where(srcnc[latcoord][:] == timeseries_lat[ipoint])
        ilon=np.where(srcnc[loncoord][:] == timeseries_lon[ipoint])
        map_vals[ilat,ilon]=float(classification_vector[ipoint])
    #endfor
    dstnc[varname][:,:]=map_vals[:,:]


    srcnc.close()
    dstnc.close()

    print("Finished with file: ",output_file)

#enddef
