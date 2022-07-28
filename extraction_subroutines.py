# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys
import numpy as np
import pandas as pd
import re
import sys,traceback
from datetime import datetime
import time

# Modules I wrote
from time_axis_manipulations import time_axis,create_annual_axis,create_monthly_axis,create_daily_axis
from netcdf_subroutines import find_orchidee_coordinate_names,harmonized_netcdf

#####################################################################

def get_debug_flags():
    
    # This is a flag that will print information on timing for various
    # subroutines.
    ldebug_time=False

    return ldebug_time

#enddef

# This is a class to hold variable information that
# we are extracting from many input file
class extracted_variable_class:
    def __init__(self, name, var_dimensions,datatype, timecoord, latcoord, loncoord, vegetcoord, nlats, nlons, nvegets, lats, lons, extract_region=None):
        self.dimensions = var_dimensions
        self.name=name
        self.datatype=datatype
        self.timecoord=timecoord
        self.latcoord=latcoord
        self.loncoord=loncoord
        self.vegetcoord=vegetcoord
        self.nlats=nlats
        self.nlons=nlons
        self.nvegets=nvegets
        self.lats=lats
        self.lons=lons

        # This region is a list of four numbers: lat (southern boundary), lat (northern), lon (western), lon (eastern)
        self.extract_region=extract_region

    #enddef

    # Based on the number of timepoints desired, create an empty 
    # array
    def create_data_array(self,timeaxis,timeunits):

        #### I would like to be able to treat arrays without time coordinates
        ####
        # Do we have a timecoord?  If not, we cannot do this!
        #if self.timecoord not in self.dimensions:
        #    print("Do not have a timecoord for this variable!")
        #    print("Variable: ",self.name)
        #    print("Dimensions: ",self.dimensions)
        #    print("Therefore, I cannot extend the timecoord.")
        #    sys.exit(1)
        #endif

        # Get the extracted region
        self.lats,self.lons,self.extract_lats,self.extract_lons=find_extracted_region(self.lats,self.lons,self.extract_region)
        self.nlats=len(self.lats)
        self.nlons=len(self.lons)

        ntimepoints=len(timeaxis)
        self.original_time_axis=timeaxis
        self.timeunits=timeunits

        # This depends on the dimensions of the variable.  Pick a couple
        # explicit cases and crash if it's anything else.
        if len(self.dimensions) == 4:
            if self.dimensions == (self.timecoord, self.vegetcoord, self.latcoord, self.loncoord):

                self.data_values=np.zeros((ntimepoints, self.nvegets, self.nlats, self.nlons),dtype=self.datatype)
            else:
                print("I cannot yet create an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        elif len(self.dimensions) == 3:
            if self.dimensions == (self.timecoord, self.latcoord, self.loncoord):

                 self.data_values=np.zeros((ntimepoints, self.nlats, self.nlons),dtype=self.datatype)

            else:
                print("I cannot yet create an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        elif len(self.dimensions) == 2:
            if self.dimensions == ('time_counter', 'axis_nbounds'):

                self.data_values=np.zeros((ntimepoints, 2),dtype=self.datatype)

            elif self.dimensions == (self.latcoord, self.loncoord):

                self.data_values=np.zeros((self.nlats, self.nlons),dtype=self.datatype)

            else:
                print("I cannot yet create an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        else:
            print("I cannot yet create an array with this dimension string.")
            print(self.dimensions)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

    # Based on the number of timepoints desired, create an empty 
    # array.  Try passing a slice here instead of an individual time
    # in order to make it faster.  It does!  Order of magnitude or more.
    def fill_data_array(self,incoming_data_array,new_timeslice,old_timeslice):

        ldebug_time=get_debug_flags()

        if ldebug_time:
            start_clock=time.perf_counter()
        #endif

        # This depends on the dimensions of the variable.  Pick a couple
        # explicit cases and crash if it's anything else.

        # Note that the incoming array is masked!  I want to convert masked values
        # to NaN

        # Slicing with lists of the integers leads to strange
        # behavior in the case of 4-D, although it seems to work
        # fine in the others.  So convert into a slice first.
        lat_slice=slice(self.extract_lats[0],self.extract_lats[-1]+1,1)
        lon_slice=slice(self.extract_lons[0],self.extract_lons[-1]+1,1)

        if len(self.dimensions) == 4:
            if self.dimensions == (self.timecoord, self.vegetcoord, self.latcoord, self.loncoord):
                
                self.data_values[new_timeslice,:,:,:]=np.ma.filled(incoming_data_array[old_timeslice,:,lat_slice,lon_slice],fill_value=np.nan)


            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        elif len(self.dimensions) == 3:
            if self.dimensions == (self.timecoord, self.latcoord, self.loncoord):

                self.data_values[new_timeslice,:,:]=np.ma.filled(incoming_data_array[old_timeslice,lat_slice,lon_slice],fill_value=np.nan)


            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        elif len(self.dimensions) == 2:
            if self.dimensions == ('time_counter', 'axis_nbounds'):

                self.data_values[new_timeslice,:]=np.ma.filled(incoming_data_array[old_timeslice,:],fill_value=np.nan)

            elif self.dimensions == (self.latcoord, self.loncoord):
                self.data_values[:,:]=np.ma.filled(incoming_data_array[lat_slice,lon_slice],fill_value=np.nan)

            else:
                print("I cannot yet create an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        else:
            print("I cannot yet fill an array with this dimension string.")
            print(self.dimensions)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if ldebug_time:
            end_clock=time.perf_counter()
            print("Total of {:0.4f} seconds for fill_data_array.".format(end_clock-start_clock))
        #endif

    #enddef

    def regrid_time_axis(self,new_time_axis_values,new_time_axis_units):

        if new_time_axis_units != self.timeunits:
            print("I cannot yet change units while regridding in regrid_time_axis.")
            print("Old time units: ",self.timeunits)
            print("New time units: ",new_time_axis_units)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Create time axes for this, so I get the bounds.
        old_timeaxis=time_axis(self.original_time_axis,self.timeunits)
        regridded_timeaxis=time_axis(new_time_axis_values,new_time_axis_units)

        if old_timeaxis.ntimepoints / regridded_timeaxis.ntimepoints != 12:
            print("Can only do monthly to annual time axis regridding right now.")
            print("Old number of points: ",old_timeaxis.ntimepoints)
            print("New number of points: ",regridded_timeaxis.ntimepoints)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if self.name.lower() in ["twbr","veget_cov_max","lai","ind","lai_mean","lai_max","veget_max","sap_m_ab","sap_m_be","heart_m_be","heart_m_ab","total_soil_carb","total_bm_litter","litter_str_ab","litter_met_ab","litter_str_be","litter_met_be","wood_harvest_pft","labile_m_n","reserve_m_n","npp","gpp","nbp_pool","leaf_age_crit","leaf_age","leaf_turn_c","lai_mean_gs","fruit_m_c","wstress_season","leaf_m_max_c",'leaf_turn_ageing_c',"sap_m_ab_c","sap_m_be_c","labile_m_c","reserve_m_c","total_m_c","height","recruits_ind","swdown","tair","rain","snowf","height_dom","recruits_ind"]:
            self.time_aggregation_operation="ave"
        elif self.name.lower() in ["areas","contfrac"]:
            self.time_aggregation_operation="none" # no time axis
        else:
            print("What kind of operation do you want to do for variable {} when regridding time axis?".format(self.name))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # This depends on the dimensions of the variable.  Pick a couple
        # explicit cases and crash if it's anything else.
        if len(self.dimensions) == 4:
            if self.dimensions == (self.timecoord, self.vegetcoord, self.latcoord, self.loncoord):

                # A nice way to take the average across groups of 12.  Reshape the array, adding a new
                # axis, and then take the mean of that axis.
                self.regridded_data_values=np.reshape(self.data_values[:,:,:,:], (-1, 12, self.data_values.shape[1],self.data_values.shape[2],self.data_values.shape[3]))
                if self.time_aggregation_operation=="ave":
                    self.regridded_data_values=np.mean(self.regridded_data_values,axis=1)
                #endif

            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        elif len(self.dimensions) == 3:
            if self.dimensions == (self.timecoord, self.latcoord, self.loncoord):

                # A nice way to take the average across groups of 12.  Reshape the array, adding a new
                # axis, and then take the mean of that axis.
                self.regridded_data_values=np.reshape(self.data_values[:,:,:], (-1, 12, self.data_values.shape[1],self.data_values.shape[2]))
                if self.time_aggregation_operation=="ave":
                    self.regridded_data_values=np.mean(self.regridded_data_values,axis=1)
                #endif

                # Do a little test.  This showed that the averaging was done properly
                #for ilat in range(self.data_values.shape[1]):
                #    for ilon in range(self.data_values.shape[2]):
                #        if not np.all(np.isnan(self.data_values[:,ilat,ilon])):
                #            print("jifoez ",ilat,ilon)
                #            for iyear in range(self.regridded_data_values.shape[0]):
                #                imonth=iyear*12
                #                jmonth=iyear*12+11
                #                print(iyear,imonth,jmonth)
                #                print(self.data_values[imonth:jmonth+1,ilat,ilon])
                #                print(self.regridded_data_values[iyear,ilat,ilon])
                            #endfor
                        #endif

                    #endfor
                #endfor


            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        elif len(self.dimensions) == 2:
            if self.dimensions == ('time_counter', 'axis_nbounds'):

                # A nice way to take the average across groups of 12.  Reshape the array, adding a new
                # axis, and then take the mean of that axis.
                self.regridded_data_values=np.reshape(self.data_values[:,:], (-1, 12, self.data_values.shape[1]))
                if self.time_aggregation_operation=="ave":
                    self.regridded_data_values=np.mean(self.regridded_data_values,axis=1)
                #endif

            elif self.dimensions == (self.latcoord, self.loncoord):
                # No time dimension...nothing to be done
                self.regridded_data_values=self.data_values[:,:]

            else:
                print("I cannot yet create an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        else:
            print("I cannot yet fill an array with this dimension string.")
            print(self.dimensions)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

    #enddef

#endclass  

def extract_variables_from_file(sim_params):


    ldebug_time=get_debug_flags()

    if ldebug_time:
        start_clock=time.perf_counter()
    #endif

    syear=sim_params.syear
    eyear=sim_params.eyear
    year_increment=sim_params.year_increment
    variables_to_extract=sim_params.variables_to_extract
    input_file_name_start=sim_params.input_file_name_start
    input_file_name_end=sim_params.input_file_name_end
    output_file_name=sim_params.condensed_nc_file_name

    extracted_variables={}
    timeaxis_values=[]
    
    # Extracting variables takes time.  I don't want to always redo it,
    # so I'd like to have one file with all the variables I use.
    # However, there are some classifications that I do with only
    # certain datasets.  To avoid having the change the variable
    # list every time, I skip over a variable if I cannot find it.  Of 
    # course, if you try to use this variable later on, the code will
    # crash.
    skipped_variables=[]

    # We have different behavior for files that have an increment of a single
    # year and those which have an increment of multiple years.
    # In the case of one-year increments, eyear should be one year greater
    # than the last year you have data for.

    for iyear in range(syear,eyear+1,year_increment):
        if year_increment == 1:
            stomate_file_name=input_file_name_start + "{}0101_{}1231".format(iyear,iyear) + input_file_name_end
        else:
            stomate_file_name=input_file_name_start + "{}0101_{}1231".format(iyear,iyear+year_increment-1) + input_file_name_end
        #endif

        # The name should be a stomate filename
        m_sto=re.search("stomate",stomate_file_name,re.IGNORECASE)
        if not m_sto:
            print("I am expecting a series of files with stomate in the filename.")
            print("I then replace stomate by sechiba in the case where you indicate the variable is found in sechiba.")
            print("This follows standard ORCHIDEE output.  I did not find that file format here, so I am stopping.")
            print(stomate_file_name)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        
        # Assign each variable the name of a file where the variable should be located
        sechiba_file_name=re.sub("stomate","sechiba",stomate_file_name,flags=re.IGNORECASE)
        varlocation={}
        for varname in variables_to_extract:

            # At this stage, varname may be a list.  We don't yet know which of the variable names
            # we are looking for.
            if isinstance(varname,list):
                for varname2 in varname:

                    varloc=sim_params.variables_in_which_file[varname2]

                    if varloc == "sechiba":
                        varlocation[varname2]=sechiba_file_name
                    else:
                        varlocation[varname2]=stomate_file_name
                    #endif
                #endif
            else:
                varloc=sim_params.variables_in_which_file[varname]
                if varloc == "sechiba":
                    varlocation[varname]=sechiba_file_name
                else:
                    varlocation[varname]=stomate_file_name
                #endif
            #endif

        #endif

        print("Getting general file information from: ",stomate_file_name)

        try:
            srcnc = NetCDFFile(stomate_file_name,"r")
        except:
            print("Does this file not exist?")
            print("Year increment: ",year_increment,iyear,eyear,syear)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endtry

        # Find the names of some coordinates.  If we are fixing the time axis,
        # we don't care what the time units are here
        timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc,check_units=False)

        try:
            time_units=srcnc[timecoord].units
        except:
            time_units=""
        #endtry

        # For the first run through,
        # I just need to find out how many time points we have.
#        ntimepoints=ntimepoints+len(srcnc[timecoord][:])
        timeaxis_values=np.append(timeaxis_values,srcnc[timecoord][:])

        # Also need to know the latitudes and longitudes, since we may
        # be changing them if we just extract a region.  The new values
        # need to be used in the extracted file.
        old_lats=srcnc[latcoord][:]
        old_lons=srcnc[loncoord][:]

        srcnc.close()

        # If this is the first input file, then we create our output file.
        # We copy over all the dimensions and metadata, as well as
        # any variables with the same name as our dimensions.  Need to do this for BOTH files, just in case!
        # Notice that the sechiba file may not exist, and that's okay...if we have no sechiba variables.
        if iyear == syear:

            # Figure out our new latitude and longitude coordinates.
            new_lats,new_lons,rdum1,rdum2=find_extracted_region(old_lats,old_lons,[sim_params.slat_window,sim_params.nlat_window,sim_params.wlon_window,sim_params.elon_window])
            new_nlats=len(new_lats)
            new_nlons=len(new_lons)

            dstnc = NetCDFFile(output_file_name,"w")

            # Copy over coordinate information
            for input_file_name in [sechiba_file_name,stomate_file_name]:

                print("Getting specific variable information from: ",input_file_name)

                try:
                    srcnc = NetCDFFile(input_file_name,"r")
                except:
                    if input_file_name == sechiba_file_name:
                        if "sechiba" in sim_params.variables_in_which_file.values():
                            print("You have requested a variable in a sechiba history file, but I cannot find")
                            print("   a file for this year!")
                            print("Variables requested: ",variables_to_extract)
                            print("Variable location: ",sim_params.variables_in_which_file)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        else:
                            print("No sechiba file and no sechiba variables requested.  Skipping.")
                        #endif
                    else:
                        if "stomate" in sim_params.variables_in_which_file.values():
                            print("You have requested a variable in a stomate history file, but I cannot find")
                            print("   a file for this year!")
                            print("Variables requested: ",variables_to_extract)
                            print("Variable location: ",sim_params.variables_in_which_file)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        else:
                            print("No stomate file and no stomate variables requested.  Skipping.")
                        #endif 
                    #endif
                #endtry

                for name, dimension in srcnc.dimensions.items():
                    # It's possible the dimension already exists because it 
                    # was in the other file

                    if name not in dstnc.dimensions.keys():
                        if name == latcoord:
                            dstnc.createDimension(name, (new_nlats))
                        elif name == loncoord:
                            dstnc.createDimension(name, (new_nlons))
                        else:
                            dstnc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
                        #endif

                        
                        # If a variable exists for this dimensions, copy it
                        if name in srcnc.variables.keys():
                            variable=srcnc[name]
                            x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
                            # copy variable attributes all at once via dictionary
                            dstnc[name].setncatts(srcnc[name].__dict__)
                            # Copy the values for all variables except the 
                            # time, latitude, and longitude, since they 
                            # may change
                            if name not in [timecoord,latcoord,loncoord]:
                                dstnc[name][:]=srcnc[name][:]
                            elif name == latcoord:
                                dstnc[name][:]=new_lats[:]
                            elif name == loncoord:
                                dstnc[name][:]=new_lons[:]
                            #endif
                        #endtry
                    #endtry
                #endfor

                # copy all the metadata
                dstnc.setncatts(srcnc.__dict__)

                # Now close this file, since the variables may be found in different files
                srcnc.close()
            #endofr

            # Create a time_bounds variable
            timebounds_varname="{}_bounds".format(timecoord)
            x = dstnc.createVariable(timebounds_varname, srcnc[timecoord].datatype, (timecoord, "axis_nbounds"))

            # Now copy over the information for every variable
            # Do not copy the values since we are going to add values along the time axis
            for iname,name in enumerate(variables_to_extract):

                # Here is where we figure out which variables are actually in the files.
                # Note that each element of variables_to_extract could be a list.  We check
                # to see which of those we can actually find, reducing any lists in
                # variables_to_extract to be a single string.
                if isinstance(name,list):

                    found_varname=[]
                    for varname2 in name:
                        # Check to see which input file the variable is in
                        try:
                            srcnc = NetCDFFile(varlocation[varname2],"r")
                        except:
                            print("Does {} not exist?".format(varlocation[name]))
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        #endtry

                        # Does the variable exist?
                        if varname2 in srcnc.variables.keys():
                            found_varname.append(varname2)
                        #endif

                        srcnc.close()

                    #endfor

                    if len(found_varname) != 1:
                        print("*******************************************************************")
                        print("Unable to reduce a list of possible variable names to a single variable!")
                        print("Possible names: ",name)
                        print("Found matches: ",found_varname)
                        print("Location: ",varlocation[name[0]])
                        traceback.print_stack(file=sys.stdout)
                        sys.exit(1)
                    #endif

                    # Replace the list of names with a single name
                    variables_to_extract[iname]=found_varname[0]
                    print("Reducing list of variables to: ",variables_to_extract[iname])

                #endif
                # Don't do anything if it's not a list, since all that is handled normally in
                # the next step

            #endfor

            # It's possible that some of the variables are redundant.  For example, both LAI_MEAN and
            # LAI_MAX are reduced to LAI in ORCHIDEE TAG2.2.  So remove the redundant items from the
            # list we want to extract.
            variables_to_extract=list(set(variables_to_extract))
            
            print("Final list of variables to extract: ",variables_to_extract)

            # Now copy over the information for every variable
            # Do not copy the values since we are going to add values along the time axis
            for name in variables_to_extract:

                # If we haven't found a name before, we skip it.
                if name in skipped_variables:
                    continue
                #endif

                # Check to see which input file the variable is in
                try:
                    srcnc = NetCDFFile(varlocation[name],"r")
                except:
                    print("Does {} not exist?".format(varlocation[name]))
                    #sys.exit(1)
                #endtry

                # Does the variable exist?
                if name not in srcnc.variables.keys():
                    print("*******************************************************************")
                    print("Cannot find variable {} in file {}.".format(name,varlocation[name]))
                    print("Check capitalization and spelling.")
                    print("Variables in the file: ",srcnc.variables.keys())
                    skipped_variables.append(name)
                    continue
                    #sys.exit(1)
                #endif

                variable=srcnc[name]

                # Create a variable instance that I'll fill up with data
                # later

                extracted_variables[name]=extracted_variable_class(name, variable.dimensions,variable.datatype,timecoord, latcoord, loncoord, vegetcoord, len(srcnc[latcoord]),len(srcnc[loncoord]),len(srcnc[vegetcoord]),srcnc[latcoord][:],srcnc[loncoord][:],extract_region=[sim_params.slat_window,sim_params.nlat_window,sim_params.wlon_window,sim_params.elon_window])

                # And create the variable in the file
                x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dstnc[name].setncatts(srcnc[name].__dict__)

                srcnc.close()
                
            #endfor

        #endif


    #endif

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 1.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    ###### Now we need to figure out our time axis.

    #print("In total, we have {} time points.".format(ntimepoints))
    #print(timeaxis)
    # We have a list of values and we have a units string.  Check to see what
    # kind of frequency we are dealing with.  Also do some basic checks to
    # see if the axis makes sense.
    if sim_params.fix_time_axis:
        # This is used when we know we have a problem with the time axis.
        # Look at the starting date, the end date, the number of steps,
        # and the origin date and create a time axis.
        
        # Figure out the units that we want.  seconds don't really work
        # with long FG1 runs of 340 years, so let's try days, since
        # we may be doing some daily simulations.
        harm_nc=harmonized_netcdf()

        if sim_params.fix_time_axis == "annual":
            timeaxis_values=create_annual_axis(sim_params.syear,sim_params.eyear,sim_params.desired_oyear,sim_params.desired_omonth,sim_params.desired_oday,sim_params.desired_ohour,sim_params.desired_omin,sim_params.desired_osec,sim_params.desired_ounits)
        elif sim_params.fix_time_axis == "monthly":
            timeaxis_values=create_monthly_axis(sim_params.syear,sim_params.eyear,sim_params.desired_oyear,sim_params.desired_omonth,sim_params.desired_oday,sim_params.desired_ohour,sim_params.desired_omin,sim_params.desired_osec,sim_params.desired_ounits)

        elif sim_params.fix_time_axis == "daily":
            # Start at noon
            timeaxis_values=create_daily_axis(datetime(sim_params.syear,1,1,12,0,0),datetime(sim_params.eyear,12,31,23,0,0),sim_params.desired_ounits,sim_params.desired_timeunits,calendar_type=sim_params.force_calendar)
            time_units=sim_params.desired_timeunits
        else:
            print("Axis doesn't seem to be annual, monthly, or daily. Not sure what to do.")
            print("Starting year, ending year, total years: ",sim_params.syear,sim_params.eyear,sim_params.ntotal_data_years)
            print("Length of aggregated time axis: ",len(timeaxis_values))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # The calendar attribute and units may be messed up.  
        # Since I've recreated it, reset them.
        if sim_params.force_calendar:
            dstnc[timecoord].setncatts({"calendar" : sim_params.force_calendar})
            combined_timeaxis=time_axis(timeaxis_values,time_units,calendar=sim_params.force_calendar)
        else:
            dstnc[timecoord].setncatts({"calendar" : "GREGORIAN"})
            combined_timeaxis=time_axis(timeaxis_values,time_units)
        #endif
        dstnc[timecoord].setncatts({"units" : sim_params.desired_timeunits})
        dstnc[timecoord].setncatts({"time_origin" : sim_params.desired_timeorigin})


    else:
        combined_timeaxis=time_axis(timeaxis_values,time_units)

    #endif

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 2.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    # Do we have the dates we expect to have?
    if combined_timeaxis.syear != syear or combined_timeaxis.eyear != eyear:
        print("Based on the command line, I expect data from year {} to {}.".format(syear,eyear))
        print("However, after looking through the data files, I found data from year {} to {}.".format(combined_timeaxis.syear,combined_timeaxis.eyear))
        if sim_params.fix_time_axis is None:
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        else:
            print("Continuing, since --fix_time_axis is given a value.")
        #endif
    #endif

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 3.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    # Now we allocate arrays to hold all of our data.
    for name in variables_to_extract:
        if name in skipped_variables:
            continue
        #endif
        extracted_variables[name].create_data_array(combined_timeaxis.values,combined_timeaxis.timeunits)

        new_lats=extracted_variables[name].lats
        new_lons=extracted_variables[name].lons
        
    #endfor

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 4.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    # Start over at the beginning, and take all the data from our
    # variables of interest and the time axis from each file.
    itimestep=0
    itimestep2=0
    for iyear in range(syear,eyear+1,year_increment):
        if year_increment == 1:
            stomate_file_name=input_file_name_start + "{}0101_{}1231".format(iyear,iyear) + input_file_name_end
        else:
            stomate_file_name=input_file_name_start + "{}0101_{}1231".format(iyear,iyear+year_increment-1) + input_file_name_end
        #endif

        # The name should be a stomate filename
        m_sto=re.search("stomate",stomate_file_name,re.IGNORECASE)
        if not m_sto:
            print("I am expecting a series of files with stomate in the filename.")
            print("I then replace stomate by sechiba in the case where you indicate the variable is found in sechiba.")
            print("This follows standard ORCHIDEE output.  I did not find that file format here, so I am stopping.")
            print(stomate_file_name)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        
        # Assign each variable the name of a file where the variable should be located
        sechiba_file_name=re.sub("stomate","sechiba",stomate_file_name,flags=re.IGNORECASE)
        varlocation={}
        for varname in variables_to_extract:
            varloc=sim_params.variables_in_which_file[varname]
            if varloc == "sechiba":
                varlocation[varname]=sechiba_file_name
            else:
                varlocation[varname]=stomate_file_name
            #endif
        #endif

        # Open the file and get some information about coordinate names
        print("Extracting time data from: ",stomate_file_name)
        srcnc = NetCDFFile(stomate_file_name,"r")
        timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc,check_units=False)

        newtimesteprange=np.arange(itimestep,itimestep+len(srcnc[timecoord][:]))

        itimestep=itimestep+len(srcnc[timecoord][:])
#        for jtimestep in range(len(srcnc[timecoord][:])):

            # First, the time axis.
        #new_time_axis[newtimesteprange]=srcnc[timecoord][:]
#            itimestep1=itimestep1+1
        #endfor


        srcnc.close()

        # Now we loop over the variables, since a variable may be in either file
        for varname in variables_to_extract:
            if varname in skipped_variables:
                continue
            #endif

            # Open the file and get some information about coordinate names
            print("Extracting {} data from: ".format(varname),varlocation[varname])
            srcnc = NetCDFFile(varlocation[varname],"r")
            timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc,check_units=False)

            itimestep2=newtimesteprange[0]

            new_timeslice=slice(newtimesteprange[0],newtimesteprange[-1]+1,1)
            old_timeslice=slice(0,len(srcnc[timecoord][:]),1)

            extracted_variables[varname].fill_data_array(srcnc[varname][:],new_timeslice,old_timeslice)

            srcnc.close()
        #endfor


    #endfor

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 5.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    # Fill out the latitude and longitude, which may be different
    # from the original files.
    for name in [latcoord,loncoord]:
        if name == latcoord:
            dstnc[name][:]=new_lats[:]
        else:
            dstnc[name][:]=new_lons[:]
        #endif
    #endfor

    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 6.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    # Sometimes, we may want to change monthly data into annual data.  Let's check that here.
    if sim_params.force_annual:
        if combined_timeaxis.timestep == "1M":
            print("Trying to convert a monthly time axis into an annual time axis for extracted data.")
            annual_timeaxis_values=create_annual_axis(combined_timeaxis.syear,combined_timeaxis.eyear,combined_timeaxis.oyear,combined_timeaxis.omonth,combined_timeaxis.oday,combined_timeaxis.ohour,combined_timeaxis.omin,combined_timeaxis.osec,combined_timeaxis.ounits)
            
            new_timeaxis=time_axis(annual_timeaxis_values,time_units)

            
        else:
            print("Requested to convert a monthly time axis into an annual time axis for extracted data, but data is not monthly!")
            print(combined_timeaxis.timestep)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # What changes here?  Well, we already have the timecoord, timecoord bounds, and all the variables in arrays
        # with the old time axis.  
        for name in variables_to_extract:
            if name in skipped_variables:
                continue
            #endif
            extracted_variables[name].regrid_time_axis(new_timeaxis.values,new_timeaxis.timeunits)
        #endfor
          
        # Write them to the file.  Notice we unmask the variables when we extract, but we
        # want to remask them here.
        for name in variables_to_extract:
            if name in skipped_variables:
                continue
            #endif
            nanmask=np.isnan(extracted_variables[name].regridded_data_values[:])
            masked_array=np.ma.array(extracted_variables[name].regridded_data_values[:],mask=nanmask)
            dstnc[name][:]=masked_array
        #endfor

    else:
            
        print("Creating a new time axis just like the old.")

        if sim_params.force_calendar is not None:
            # We have fixed the calendar here
            new_timeaxis=time_axis(timeaxis_values,time_units,calendar=sim_params.force_calendar)
        else:
            # This is just the same as the combined_timeaxis
            new_timeaxis=time_axis(timeaxis_values,time_units)
        #endif

        # Write the data to a file
        for name in variables_to_extract:
            if name in skipped_variables:
                continue
            #endif
            nanmask=np.isnan(extracted_variables[name].data_values[:])
            masked_array=np.ma.array(extracted_variables[name].data_values[:],mask=nanmask)
            dstnc[name][:]=masked_array
        #endfor

    #endif

    # Now print the time data into the destination file
    dstnc[timecoord][:]=new_timeaxis.values[:]
    dstnc[timebounds_varname][:]=new_timeaxis.timebounds_values[:]
        
    # Due to changing regions, this seems like the best place to do this.
    # I would like the latitudes to always be ascending, and the longitudes
    # to always go from -180 to 180.  Check for this and correct if needed.
    if dstnc[latcoord][1]-dstnc[latcoord][0] < 0.0:
        print("Inverted latitude coordinates.")
        dstnc[latcoord][:]=np.flip(dstnc[latcoord][:])
        for varname in dstnc.variables.keys():
            if varname == latcoord:
                continue
            #endif
            if latcoord in dstnc[varname].dimensions:
                print("Changing latcoord for: ",varname)
                axis_flip=dstnc[varname].dimensions.index(latcoord)
                dstnc[varname][:]=np.flip(dstnc[varname][:],axis=axis_flip)
            #endif
        #endfor
    #endif

    if np.max(dstnc[loncoord][:]) > 180.0:
        print("****************************************************")
        print("-------> Seems like longitudes are not what I want.")
        print("     Need to write this, as I'm not sure how the data may look.")
        print("****************************************************")
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif
    dstnc.close()

#    print("ijofiez ")
#    sys.exit(1)
    if ldebug_time:
        end_clock=time.perf_counter()
        print("Total of {:0.4f} seconds for extract_variables_from_file step 7.".format(end_clock-start_clock))
        start_clock=end_clock
    #endif

    return output_file_name

#enddef


##########################################
####### Not currently used
# This routine takes a while.  So after the extraction I print all the
# data to a new file.
def extract_timeseries(input_file_name,timeseries_variable,pft_selected,veget_max_threshold):

    print("Extracting a timeseries for {} and the variable {}.".format(input_file_name,timeseries_variable))
    timeseries_list=[]
    lat_list=[]
    lon_list=[]

    icounter=0

    # Loop through every pixel in the output file and check to see
    # if this timeseries meets our criteria
    srcnc = NetCDFFile(input_file_name,"r")
    timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)

    # Check to see what the veget_max coord is
    #veget_max_name=find_variable_name_in_netcdf_file(["VEGET_MAX","VEGET_COV_MAX"],srcnc)
                

    for ilat,vlat in enumerate(srcnc[latcoord][:]):
        print("Latitude: ",vlat)
        for ilon,vlon in enumerate(srcnc[loncoord][:]):
    
            ### TESTING
            #if icounter > 10:
            #    continue

            ###################################
            # This is where things get specific for our variables

            if timeseries_variable == "LAI_MEAN":
                if not np.ma.is_masked(srcnc[veget_max_name][0, pft_selected, ilat, ilon]):
                    if srcnc[veget_max_name][0, pft_selected, ilat, ilon] < veget_max_threshold:
                        continue
                    #endif

                    #print("Found a point!",srcnc[veget_max_name][0, pft_selected, ilat, ilon])
                    # My new array is not masked.  So convert the masked
                    # values to NaN.
                    timeseries_list.append(srcnc[timeseries_variable][:, pft_selected, ilat, ilon].filled(np.nan))
                    lat_list.append(vlat)
                    lon_list.append(vlon)

                    # TESTING
                    icounter=icounter+1
                #endif
            elif timeseries_variable in ["TWBR","HEIGHT"]:
                if not np.ma.is_masked(srcnc[timeseries_variable][0, ilat, ilon]):

                    #print("Found a point!",srcnc["VEGET_MAX"][0, pft_selected, ilat, ilon])
                    # My new array is not masked.  So convert the masked
                    # values to NaN.
                    timeseries_list.append(srcnc[timeseries_variable][:, ilat, ilon].filled(np.nan))
                    lat_list.append(vlat)
                    lon_list.append(vlon)

                    # TESTING
                    icounter=icounter+1

                #endif

            elif timeseries_variable == "N_RESERVES":
                if not np.ma.is_masked(srcnc[veget_max_name][0, pft_selected, ilat, ilon]):
                    if srcnc[veget_max_name][0, pft_selected, ilat, ilon] < veget_max_threshold:
                        continue
                    #endif

                    #print("Found a point!",srcnc[veget_max_name][0, pft_selected, ilat, ilon])
                    # My new array is not masked.  So convert the masked
                    # values to NaN.
                    # This is a case where we have two pools
                    temp_series=srcnc[timeseries_variable][:, pft_selected, ilat, ilon].filled(np.nan)+srcnc["LABILE_M_n"][:, pft_selected, ilat, ilon].filled(np.nan)
                    timeseries_list.append(srcnc[timeseries_variable][:, pft_selected, ilat, ilon].filled(np.nan))
                    lat_list.append(vlat)
                    lon_list.append(vlon)

                    # TESTING
                    icounter=icounter+1
                #endif
            else:
                print('Not sure how you want to extract the timeseries in extract_timeseries!')
                print(timeseries_variable)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            ###################################

        #endfor
    #endfor

    # now change these lists into arrays
    npoints=len(lat_list)
    
    print("Found {} timeseries!".format(npoints))
    if npoints == 0:
        print("Stopping.  Nothing to analyze.")
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    timeseries_array=np.zeros((npoints,len(srcnc[timecoord][:])))
    timeseries_lat=np.zeros((npoints))
    timeseries_lon=np.zeros((npoints))

    for ipoint in range(npoints):
        timeseries_array[ipoint,:]=timeseries_list[ipoint]
        timeseries_lat[ipoint]=lat_list[ipoint]
        timeseries_lon[ipoint]=lon_list[ipoint]
    #endfor
                              
    srcnc.close()

    # Since finding timeseries can take a long time, save what we've found
    timeseriesdf=pd.DataFrame(data=timeseries_array)
    timeseriesdf["Latitude"]=timeseries_lat[:]
    timeseriesdf["Longitude"]=timeseries_lon[:]
    timeseriesdf.to_csv(path_or_buf="saved_timeseries.csv",index=False)

    return timeseries_array,timeseries_lat,timeseries_lon

#enddef

# Read in the file created by the above
def read_in_extracted_timeseries():

    df=pd.read_csv("saved_timeseries.csv",sep=",",index_col=None)
    
    timeseries_lat=df["Latitude"].values
    timeseries_lon=df["Longitude"].values

    # Remove the two columns
    df.drop(columns=["Latitude","Longitude"],inplace=True)
    

    # Convert the rest to an array
    timeseries_array=df.to_numpy()

    return timeseries_array,timeseries_lat,timeseries_lon

#enddef

# find the latitudes and longitudes corresponding to a subset of
# a different grid.
def find_extracted_region(old_lats,old_lons,extract_region):

    # I need to get the latitude and longitude indices to extract.
    if extract_region is None:
        new_lats=old_lats.copy()
        new_lons=old_lons.copy()
        extracted_lat_indices=list(range(len(new_lats)))
        extracted_lon_indices=list(range(len(new_lons)))
    else:
        # Figure out the size of our extracted region.  Notice that
        # we won't know the exact latitude/longitude until
        # we actually do the extraction.
        if extract_region[0] == extract_region[1] and extract_region[2] == extract_region[3]:
            print("-- Extracting a single point: ",extract_region[0],extract_region[2])
            
            minind,minval=min(enumerate(old_lats), key=lambda x: abs(x[1]-extract_region[0]))
            new_lats=[minval]
            extracted_lat_indices=[minind]
            minind,minval=min(enumerate(old_lons), key=lambda x: abs(x[1]-extract_region[2]))
            new_lons=[minval]
            extracted_lon_indices=[minind]
        else:
            print("-- Extracting a region: ",extract_region[0],extract_region[1],extract_region[2],extract_region[3])

            # Return all indices between the given bounds
            extracted_lat_indices=(old_lats <= extract_region[1]) & (old_lats >= extract_region[0])
            extracted_lat_indices = [i for i, val in enumerate(extracted_lat_indices) if val]
            new_lats=old_lats[extracted_lat_indices]
            # And for the longitude
            extracted_lon_indices=(old_lons <= extract_region[3]) & (old_lons >= extract_region[2])
            extracted_lon_indices = [i for i, val in enumerate(extracted_lon_indices) if val]
            new_lons=old_lons[extracted_lon_indices]

        #endif
    
    #endif

    return new_lats,new_lons,extracted_lat_indices,extracted_lon_indices
#enddef
