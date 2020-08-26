# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys
from netcdf_subroutines import find_orchidee_coordinate_names
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from sklearn.cluster import KMeans
import math
import matplotlib.gridspec as gridspec
from textwrap import fill # legend text can be too long
from grid import Grid
import matplotlib as mpl
import re

class simulation_parameters:
    def __init__(self, pft_selected,veget_max_threshold,timeseries_flag,do_test,global_operation,force_annual,fix_time_axis):
        self.pft_selected = pft_selected # Note that this is stored in
        # here with Python indexing!  So PFT6 is stored here as 5.
        self.veget_max_threshold = veget_max_threshold
        self.timeseries_flag=timeseries_flag
        self.do_test=do_test
        self.ntest_points=100
        self.global_operation=global_operation
        self.force_annual=force_annual
        self.fix_time_axis=fix_time_axis

        # Based on the above, set some other flags.

        ######## Better to always extract all variables.  Otherwise, I
        ######## cannot reuse extracted files and have to redo them.
        # This can be tricky, as one variable may have a name in one run
        # and another in a different (e.g., LAI_MEAN in TRUNK 4.0 but LAI in 
        # TRUNK 2.2).  I would like this to work in the same way for both,
        # so the code needs to figure out which variable is present.
        # Try putting both the equivalent names here, and then collapsing it
        # to a single name when the variables are read in.
        self.variables_to_extract=[ ["LAI_MEAN","LAI"], ["LAI_MAX","LAI"],["VEGET_MAX","VEGET_COV_MAX"],"IND","TWBR"]
        #self.variables_to_extract=[ "TWBR"]
        # not sure what to do with time_counter_bounds.  I am trying to always create it myself.



        #####################
        # This is for a different purpose.  Sometimes I just want to extract variables.
#        self.variables_to_extract=[ ["VEGET_MAX","VEGET_COV_MAX"],"SAP_M_AB","SAP_M_BE","HEART_M_AB","HEART_M_BE","TOTAL_SOIL_CARB","TOTAL_BM_LITTER","LITTER_STR_AB","LITTER_MET_AB","LITTER_STR_BE","LITTER_MET_BE","PROD10","PROD100","PROD10_HARVEST","PROD100_HARVEST"]
# I may need the product pools, but for right now I do not know how to deal
# with a fourth non-veget axis in the array processing.
        self.variables_to_extract=[ ["VEGET_MAX","VEGET_COV_MAX"],"SAP_M_AB","SAP_M_BE","HEART_M_AB","HEART_M_BE","TOTAL_SOIL_CARB","TOTAL_BM_LITTER","LITTER_STR_AB","LITTER_MET_AB","LITTER_STR_BE","LITTER_MET_BE","WOOD_HARVEST_PFT"]
        #####################

        self.variables_in_which_file={"LAI_MEAN":"stomate","LAI_MAX":"stomate","LAI" : "stomate","VEGET_MAX":"stomate","VEGET_COV_MAX":"stomate","IND":"stomate","TWBR":"sechiba","SAP_M_AB":"stomate","SAP_M_BE":"stomate","HEART_M_AB":"stomate","HEART_M_BE":"stomate","TOTAL_SOIL_CARB" : "stomate","TOTAL_BM_LITTER":"stomate","LITTER_STR_AB":"stomate","LITTER_MET_AB":"stomate","LITTER_STR_BE":"stomate","LITTER_MET_BE":"stomate","PROD10":"stomate","PROD100":"stomate","PROD10_HARVEST":"stomate","PROD100_HARVEST":"stomate","WOOD_HARVEST_PFT":"stomate"}



        if self.timeseries_flag in ("LAI_MEAN1","LAI_MEAN2","LAI_MEAN_BIMODAL"):
            #self.variables_to_extract=["LAI_MEAN","LAI_MAX","VEGET_MAX","IND","time_counter_bounds"]
            #self.variables_in_which_file=["stomate","stomate","stomate","stomate","stomate"]
            self.timeseries_variable="LAI_MEAN"
        elif self.timeseries_flag in ("TWBR"):
            #self.variables_to_extract=["time_counter_bounds","TWBR"]
            #self.variables_in_which_file=["stomate","sechiba"]
            self.timeseries_variable="TWBR"
        else:
            print("I do not recognize this timeseries flag in init sim param!")
            print(self.timeseries_flag)
            sys.exit(1)
        #endif

        # LAI thresholds suggested by Sebastiaan:
        if pft_selected == 1: # Remember that Python arrays start from 0!
            self.lai_good_threshold=5.0
        elif pft_selected == 2: 
            self.lai_good_threshold=5.0
        elif pft_selected == 3: 
            self.lai_good_threshold=4.0
        elif pft_selected == 4: 
            self.lai_good_threshold=2.0
        elif pft_selected == 5: 
            self.lai_good_threshold=4.0
        elif pft_selected == 6: 
            self.lai_good_threshold=3.0
        elif pft_selected == 7: 
            self.lai_good_threshold=3.0
        elif pft_selected == 8: 
            self.lai_good_threshold=2.0
        elif pft_selected == 9: 
            self.lai_good_threshold=3.0
        elif pft_selected == 10: 
            self.lai_good_threshold=3.0
        elif pft_selected == 11: 
            self.lai_good_threshold=3.0
        elif pft_selected == 12: 
            self.lai_good_threshold=3.0
        elif pft_selected == 13: 
            self.lai_good_threshold=2.0 # may be too high?
        elif pft_selected == 14: 
            self.lai_good_threshold=2.0 # may be too high?
        else:
            print("Not sure what to do with this PFT in sim_params def!")
            print(pft_selected)
            sys.exit(1)
        #endif

        # What fraction of our datapoints need to be above this in order for it to be good?
        self.lai_high_threshold_good_fraction=0.8
        self.lai_high_threshold_bad_fraction=0.2 # if it's never more than this, it's bad


        # A lower limit to the LAI
        if pft_selected in (1,2,3,5,6,7,9,10,11,12): 
            self.lai_bad_threshold=1.0
        elif pft_selected in (4,8,13,14): # a few PFTs with lower LAI
            self.lai_bad_threshold=0.5
        else:
            print("I don't know how to create criteria for this PFT!")
            sys.exit(1)
        #endif

        # What fraction of our datapoints need to be below this in order for it to be bad?
        self.lai_low_threshold_bad_fraction=0.5
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
            sys.exit(1)
        #endif

        # Above which fraction of values are NaN do we flag it as bad?
        self.nan_bad_threshold=0.1
        # Below which fraction of values are NaN can we accept it as good?
        self.nan_good_threshold=0.01

        ## This is for the TWBR
        self.twbr_high_threshold=1e-15
        self.twbr_low_threshold=1e-16
        self.twbr_high_threshold_bad_fraction=0.1
        self.twbr_low_threshold_good_fraction=0.5



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

        # This is where we put the maps of every variable that
        # we extract, in NetCDF format
        if year_increment == 1:
            self.condensed_nc_file_name="extracted_variables_{}{}.{}.nc".format(input_file_name_start,syear,eyear)
        else:
            self.condensed_nc_file_name="extracted_variables_{}{}.{}.nc".format(input_file_name_start,syear,eyear)
        #endif

        # Set some additional information, now that we have the input file
        # names
        self.set_classification_filename_information()

    #enddef

    def set_classification_filename_information(self):


        # These names will depend on the analysis we are doing
        if self.timeseries_flag == "LAI_MEAN1":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="{}\nEach pixel classified according to the LAI timeseries for PFT {}".format(self.input_file_name_start,self.pft_selected+1)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA bad pixel has at least {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of annual LAI_MEAN values above {},\n    OR more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_low_threshold_bad_fraction*100.0,self.lai_bad_threshold,self.lai_high_threshold_bad_fraction*100.0,self.lai_good_threshold,self.nan_bad_threshold*100.0))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA bad pixel has at least {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of annual LAI_MEAN values above {},\n    OR more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_low_threshold_bad_fraction*100.0,self.lai_bad_threshold,self.lai_high_threshold_bad_fraction*100.0,self.lai_good_threshold,self.nan_bad_threshold*100.0))

            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA good pixel has at least {}% of annual LAI_MEAN values above {},\n    AND no more than {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_high_threshold_good_fraction*100.0,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold*100.0))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA good pixel has at least {}% of annual LAI_MEAN values above {},\n    AND no more than {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_high_threshold_good_fraction*100.0,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold*100.0))
            self.plot_subtitle.append("PFT {}, classification based only on LAI_MEAN\nA good pixel has at least {}% of annual LAI_MEAN values above {},\n    AND no more than {}% of annual LAI_MEAN values below {},\n    AND no more than {}% of the annual LAI_MEAN values are NaNs.".format(self.pft_selected+1,self.lai_high_threshold_good_fraction*100.0,self.lai_good_threshold,self.lai_low_threshold_good_fraction*100.0,self.lai_bad_threshold,self.nan_good_threshold*100.0))

        elif self.timeseries_flag == "LAI_MEAN_BIMODAL":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="Each pixel classified according to the LAI timeseries for PFT {}".format(self.pft_selected+1)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN.png".format(self.input_file_name_start,self.pft_selected+1)

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
            self.classified_map_filename="classified_map_{}PFT{}_LAIMEAN_IND.png".format(self.input_file_name_start,self.pft_selected+1)
            # And the title of the map itself
            self.classified_map_title="Each pixel classified according to the LAI timeseries for PFT {}".format(self.pft_selected+1)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}PFT{}_LAIMEAN_IND.png".format(self.input_file_name_start,self.pft_selected+1)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}PFT{}_LAIMEAN_IND.png".format(self.input_file_name_start,self.pft_selected+1)

            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle="PFT {}, classification based on LAI_MEAN and IND".format(self.pft_selected+1)

            ######
        elif self.timeseries_flag == "TWBR":

            # For the name of the file with the map
            self.classified_map_filename="classified_map_{}TWBR.png".format(self.input_file_name_start)
            # And the title of the map itself
            self.classified_map_title="{}\nEach pixel classified according to the TWBR timeseries".format(self.sim_name)

            # For the name of the file with the timeseries
            self.classified_timeseries_filename="classified_timeseriesXXXX_{}TWBR.png".format(self.input_file_name_start)

            # For the name of the file with the histograms of the classifiers
            self.class_histogram_filename="classification_histogram_{}TWBR.png".format(self.input_file_name_start)


            # This is a subtitle that identifies what we've done
            # a little better
            self.plot_subtitle=[]
            self.plot_subtitle.append("Classification based only on TWBR\nA bad pixel has at least {}% of annual TWBR values above a magitude of {},\n    AND less than {}% of annual TWBR values below a magnitude of {}.".format(self.twbr_high_threshold_bad_fraction*100.0,self.twbr_high_threshold,self.twbr_low_threshold_good_fraction*100.0,self.twbr_low_threshold))
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append(self.plot_subtitle[0])
            self.plot_subtitle.append("Classification based only on TWBR\nA good pixel has at least {}% of annual TWBR values below a magnitude of {},\n    AND less than {}% of annual TWBR values above a magnitude of {}.".format(self.twbr_low_threshold_good_fraction*100.0,self.twbr_low_threshold,self.twbr_high_threshold_bad_fraction*100.0,self.twbr_high_threshold))

        else:
            print("Do not recognize timeseries flag in set_classification_filename_information!")
            print(self.timeseries_flag)
            sys.exit(1)
        #endif

    #enddef

    def set_variable_names(self,veget_max_name,lai_mean_name,lai_max_name):
        self.veget_max_name=veget_max_name
        self.lai_mean_name=lai_mean_name
        self.lai_max_name=lai_max_name
        
        # Our timeseries variable depends on this, too.

        if self.timeseries_flag in ["LAI_MEAN1","LAI_MEAN_BIMODAL","LAI_MEAN2"]:
            self.timeseries_variable=lai_mean_name
     

            ######
        elif self.timeseries_flag == "TWBR":
            self.timeseries_variable="TWBR"

        else:
            print("Do not recognize timeseries flag in set_variable_names!")
            print(self.timeseries_flag)
            sys.exit(1)
        #endif

    #enddef

#endclass


# This loops through all our timeseries and evaluates them against
# some criteria.  These criteria depend on the variable we have extracted.
def create_classification_criteria(timeseries_array,sim_params):

    timeseries_variable=sim_params.timeseries_variable
    pft_selected=sim_params.pft_selected

    print("Creating classification criteria for {}.".format(timeseries_variable))

    nsamples=timeseries_array.shape[0]
    
    if sim_params.timeseries_flag in ["LAI_MEAN1"]:

        # How many different criteria?
        # 0 : Maximum value
        # 1 : Number of times below a threshold
        # 2 : Number of NaNs

        classification_array=np.zeros((nsamples,3))

        # Loop through all the timeseries and evaluate them
        for isample in range(nsamples):

            ### not sure these are useful right now
            # 0 : Number of NaN values
            #temp_array=np.isnan(timeseries_array[isample,:])
            #classification_array[isample,0]=np.sum(temp_array)

            # 1 : Number of Zero values (actually below 0.01)
            #temp_array=np.where(timeseries_array[isample,:] < 0.01, True, False)
            #classification_array[isample,1]=np.sum(temp_array)
            #################################

            # 0 : Maximum value.  
            classification_array[isample,0]=np.nanmax(timeseries_array[isample,:])
            # 1 : Number of times below a threshold
            temp_array=np.where(timeseries_array[isample,:] < sim_params.lai_bad_threshold, True, False)
            classification_array[isample,1]=np.sum(temp_array)

            # I want to add something with IND here, too.

            
        #endif

        # Now print out a few summary statistics
        classificationdf=pd.DataFrame(data=classification_array)
        classificationdf.hist(bins=30,figsize=(9,9))
        pl.suptitle("Histogram for each classification variable")
        plt.savefig(sim_params.class_histogram_filename)
        plt.close()

    else:
        print("I don't know how to create criteria for this variable!")
        sys.exit(1)
    #endif

    return classification_array

#enddef

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
            # 0 : Fraction of annual LAI_MEAN values above a threshold
            # 1 : Fraction of annual LAI_MEAN values below a different threshold
            # 2 : Fraction of annual LAI_MEAN values are NaNs.

            # The last criteria is essential.  If there are a lot of NaNs, we automatically classify it as bad, since it can
            # mess up the rest of the calculations.

            if classification_array[isample,0] > lai_high_threshold_good_fraction and classification_array[isample,2] <= sim_params.nan_good_threshold and classification_array[isample,1] < lai_low_threshold_good_fraction:
                # Good
                classification_vector[isample]=5
            elif ((classification_array[isample,0] < lai_high_threshold_bad_fraction) and (classification_array[isample,1] > lai_low_threshold_bad_fraction)) or (classification_array[isample,2] > sim_params.nan_bad_threshold):
                # Bad
                classification_vector[isample]=1
            elif (classification_array[isample,0] > lai_high_threshold_good_fraction or classification_array[isample,2] <= sim_params.nan_good_threshold or classification_array[isample,1] < lai_low_threshold_good_fraction) and (not classification_array[isample,0] < lai_high_threshold_bad_fraction and not classification_array[isample,1] > lai_low_threshold_bad_fraction):
                # Semi-good
                classification_vector[isample]=4
            elif (classification_array[isample,0] < lai_high_threshold_bad_fraction or classification_array[isample,1] > lai_low_threshold_bad_fraction) and (not classification_array[isample,0] > lai_high_threshold_good_fraction and not classification_array[isample,2] <= sim_params.nan_good_threshold and not classification_array[isample,1] < lai_low_threshold_good_fraction):
                # Semi-bad
                classification_vector[isample]=2
            else:
                # Ok?  Unclassifiable
                classification_vector[isample]=3
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

    else:
        print("Cannot yet classify points with this flag in classify_observations!")
        print(sim_params.timeseries_flag)
        sys.exit(1)
    #endif

    return classification_vector
#enddef

# Choose some of the classified points at random and plot them as timeseries
def plot_classified_observations(classification_vector,timeseries_array,timeseries_lat,timeseries_lon,sim_params,syear,eyear):

        

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

            # I need these points to be as distinct as possible
            if npoints >  classified_series.shape[0]:

                # Take all the points
                selected_timeseries=classified_series.copy()
                selected_lat=classified_lat.copy()
                selected_lon=classified_lon.copy()
                #selected_points=np.arange(0,npoints)
                #selected_points=np.where(selected_points >= ntotal, 0, selected_points)
                #selected_points=list(selected_points)
            else:

                # Extract all the timeseries with this classification level
                kmeans = KMeans(n_clusters=npoints,random_state=0).fit(classified_series)
                cluster_vector=kmeans.labels_

                selected_timeseries=np.zeros((npoints,classified_series.shape[1]),dtype=float)
                selected_lat=np.zeros((npoints),dtype=float)
                selected_lon=np.zeros((npoints),dtype=float)

                # Now choose one timeseries for each of the clusters
                for ipoint in range(npoints):

                    cluster_timeseries=classified_series[cluster_vector == ipoint]
                    cluster_lat=classified_lat[cluster_vector == ipoint]
                    cluster_lon=classified_lon[cluster_vector == ipoint]

                    if len(cluster_timeseries.shape) != 2:
                        print("Problem with clusters!")
                        print(cluster_timeseries.shape)
                        sys.exit(1)
                    #endif

                    # Need a random integer between 0 and the number of timeseries
                    # we have
                    random_point=np.random.randint(0,cluster_timeseries.shape[0])

                    selected_timeseries[ipoint,:]=cluster_timeseries[random_point,:]
                    selected_lat[ipoint]=cluster_lat[random_point]
                    selected_lon[ipoint]=cluster_lon[random_point]



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
            plt.ylabel(r"""{} [-]""".format(sim_params.timeseries_variable), fontsize=20)
            
            ax1.set_xlim(syear,eyear)
            #ax2.axis('off')
            ax1.set_facecolor(level_colors[colors_by_level[ilevel]])
            
            ax1.set_title("Randomly-selected examples of the {} timeseries classified as {}".format(ntotal,ilevel),fontsize=16)
            
            ax2=plt.subplot(gs[2])
            ax2.text(0.1,0.02,sim_params.plot_subtitle[ilevel-1],transform=fig.transFigure,fontsize=12)
            ax2.set_axis_off()
            fig.savefig(sim_params.classified_timeseries_filename.replace("XXXX","{}".format(ilevel)),dpi=300)
            plt.close(fig)     
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
    

    #####
    # I want to plot a global timeseries by combining all these pixels
    xmin=syear
    xmax=eyear
    xmin=1
    xmax=241
    global_timeseries=np.zeros((timeseries_array.shape[1]),dtype=np.float32)
    ntimeseries=timeseries_array.shape[0]
    for itimeseries in range(ntimeseries):
        global_timeseries[:]=global_timeseries[:]+timeseries_array[itimeseries,:]
    #endfor
    fig=plt.figure(2,figsize=(13, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[9,1])
    ax1=plt.subplot(gs[0])
    p1=ax1.plot(xvalues[xmin-1:xmax-1],global_timeseries[xmin-1:xmax-1], linestyle="-", marker="o",color="black",label="Global")
    plt.xlabel(r"""Time [yr]""", fontsize=20)
    plt.ylabel(r"""{} [-]""".format(sim_params.timeseries_variable), fontsize=20)
            
    #ax1.set_xlim(syear,eyear)
    #ax2.axis('off')
    #ax1.set_facecolor(level_colors[colors_by_level[ilevel]])
            
    if sim_params.global_operation == "sum":
        ax1.set_title("Global timeseries, where {} classified pixels are summed together".format(ntimeseries),fontsize=16)
    #endif
            
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
            if sim_params.timeseries_flag == "LAI_MEAN1":
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
        print("Scanning latitude for points: ",vlat)
        for ilon,vlon in enumerate(srcnc[loncoord][:]):
    
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
        classification_array=np.zeros((npoints,4))

        # How many different criteria?
        # 0 : Fraction of annual LAI_MEAN values above a threshold
        # 1 : Fraction of annual LAI_MEAN values below a different threshold
        # 2 : Fraction of annual LAI_MEAN values are NaNs
        # 3 : Fraction of timesteps when LAI_MEAN changes by more than 0.2

    elif timeseries_flag == "LAI_MEAN2":
        classification_array=np.zeros((npoints,4))

        # How many different criteria?
        # 0 : Maximum value
        # 1 : Number of times below a threshold
        # 2 : Number of times IND is above a certain threshold
        # 3 : Number of NaNs present in the LAI_MEAN timeseries


    elif sim_params.timeseries_flag == "TWBR":

        classification_array=np.zeros((npoints,2))

        # 0 : Fraction of points above a certain threshold (this is bad)
        # 1 : Fraction of points below a certain threshold (this is good)

    else:

        print("I don't know how to create criteria for this flag in extract_and_calculate_classifiction_metrics!")
        print(timeseries_flag)
        sys.exit(1)

    #endif


    # Now we loop through all the points, pick out our timeseries, and then compute
    # the metrics
    for ilat,ilon,ipoint in zip(lat_list,lon_list,range(npoints)):
  
        if ipoint % 100 == 0:
            print("Extracting and classifying point {}.".format(ipoint))
        #endif
        
        timeseries_lat[ipoint]=srcnc[latcoord][ilat]
        timeseries_lon[ipoint]=srcnc[loncoord][ilon]

        # Now the behavior changes with the flag
        classification_array[ipoint,:],timeseries_array[ipoint,:]=compute_metrics_extracted_pixel(srcnc,sim_params,ilat,ilon)
         


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

    return timeseries_array,timeseries_lat,timeseries_lon, classification_array

#enddef

def keep_grid_point(srcnc,pft_selected,ilat,ilon,veget_max_threshold,timeseries_flag,veget_max_name):
    # Check to see if, based on the timeseries we are looking for and the PFT, we want
    # to keep this gridpoint.

    if timeseries_flag in ("LAI_MEAN1","LAI_MEAN2","LAI_MEAN_BIMODAL"):
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
        sys.exit(1)
    #endif

    return lkeep

#endif


def compute_metrics_extracted_pixel(srcnc,sim_params,ilat,ilon):

    if sim_params.timeseries_flag in ("LAI_MEAN1","LAI_MEAN_BIMODAL"):
        

        timeseries=srcnc[sim_params.timeseries_variable][:, sim_params.pft_selected, ilat, ilon].filled(np.nan)

        # How many different criteria?
        carray=np.zeros((4))

        # 0 : Fraction of LAI_MEAN values above a threshold
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

    else:

        print("I don't know how to create criteria for this flag in compute_metrics_extracted_pixel!")
        print(sim_params.timeseries_flag)
        sys.exit(1)

    #endif

    return carray,timeseries
#enddef
