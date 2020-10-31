# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys
from netcdf_subroutines import find_orchidee_coordinate_names
import numpy as np
import pandas as pd
import re
from time_axis_manipulations import time_axis,create_annual_axis
import sys,traceback

# This is a class to hold variable information that
# we are extracting from many input file
class extracted_variable_class:
    def __init__(self, name, var_dimensions,datatype, timecoord, latcoord, loncoord, vegetcoord,nlats,nlons,nvegets):
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
    # array
    def fill_data_array(self,incoming_data_array,itimepoint,jtimepoint):

        # This depends on the dimensions of the variable.  Pick a couple
        # explicit cases and crash if it's anything else.

        # Note that the incoming array is masked!  I want to convert masked values
        # to NaN

        if len(self.dimensions) == 4:
            if self.dimensions == (self.timecoord, self.vegetcoord, self.latcoord, self.loncoord):
                

                self.data_values[itimepoint,:,:,:]=np.ma.filled(incoming_data_array[jtimepoint,:,:,:],fill_value=np.nan)


            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        elif len(self.dimensions) == 3:
            if self.dimensions == (self.timecoord, self.latcoord, self.loncoord):

                
                self.data_values[itimepoint,:,:]=np.ma.filled(incoming_data_array[jtimepoint,:,:],fill_value=np.nan)


            else:
                print("I cannot yet fill an array with this dimension string.")
                print(self.dimensions)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

        elif len(self.dimensions) == 2:
            if self.dimensions == ('time_counter', 'axis_nbounds'):

                self.data_values[itimepoint,:]=np.ma.filled(incoming_data_array[jtimepoint,:],fill_value=np.nan)

            elif self.dimensions == (self.latcoord, self.loncoord):
                self.data_values[:,:]=np.ma.filled(incoming_data_array[:,:],fill_value=np.nan)

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

        if self.name.lower() in ["twbr","veget_cov_max","lai","ind","lai_mean","lai_max","veget_max","sap_m_ab","sap_m_be","heart_m_be","heart_m_ab","total_soil_carb","total_bm_litter","litter_str_ab","litter_met_ab","litter_str_be","litter_met_be","wood_harvest_pft","labile_m_n","reserve_m_n"]:
            self.time_aggregation_operation="ave"
        else:
            print("What kind of operation do you want to do for variable {} when regridding time axis?".format(self.name))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # This depends on the dimensions of the variable.  Pick a couple
        # explicit cases and crash if it's anything else.
        if len(self.dimensions) == 4:
            if self.dimensions == (self.timecoord, self.vegetcoord, self.latcoord, self.loncoord):

                # A nice way to take the average across gruops of 12.  Reshape the array, adding a new
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

                # A nice way to take the average across gruops of 12.  Reshape the array, adding a new
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

                # A nice way to take the average across gruops of 12.  Reshape the array, adding a new
                # axis, and then take the mean of that axis.
                self.regridded_data_values=np.reshape(self.data_values[:,:], (-1, 12, self.data_values.shape[1]))
                if self.time_aggregation_operation=="ave":
                    self.regridded_data_values=np.mean(self.regridded_data_values,axis=1)
                #endif



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

def extract_variables_from_file(sim_parms):

    syear=sim_parms.syear
    eyear=sim_parms.eyear
    year_increment=sim_parms.year_increment
    variables_to_extract=sim_parms.variables_to_extract
    input_file_name_start=sim_parms.input_file_name_start
    input_file_name_end=sim_parms.input_file_name_end
    output_file_name=sim_parms.condensed_nc_file_name

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

                    varloc=sim_parms.variables_in_which_file[varname2]

                    if varloc == "sechiba":
                        varlocation[varname2]=sechiba_file_name
                    else:
                        varlocation[varname2]=stomate_file_name
                    #endif
                #endif
            else:
                varloc=sim_parms.variables_in_which_file[varname]
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
        print("jiofje ",sim_parms.fix_time_axis)
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
        srcnc.close()

        # If this is the first input file, then we create our output file.
        # We copy over all the dimensions and metadata, as well as
        # any variables with the same name as our dimensions.  Need to do this for BOTH files, just in case!
        # Notice that the sechiba file may not exist, and that's okay...if we have no sechiba variables.
        if iyear == syear:


            dstnc = NetCDFFile(output_file_name,"w")

            # Copy over coordinate information
            for input_file_name in [sechiba_file_name,stomate_file_name]:

                print("Getting specific variable information from: ",input_file_name)

                try:
                    srcnc = NetCDFFile(input_file_name,"r")
                except:
                    if input_file_name == sechiba_file_name:
                        if "sechiba" in sim_parms.variables_in_which_file.values():
                            print("You have requested a variable in a sechiba history file, but I cannot find")
                            print("   a file for this year!")
                            print("Variables requested: ",variables_to_extract)
                            print("Variable location: ",sim_parms.variables_in_which_file)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        else:
                            print("No sechiba file and no sechiba variables requested.  Skipping.")
                        #endif
                    else:
                        if "stomate" in sim_parms.variables_in_which_file.values():
                            print("You have requested a variable in a stomate history file, but I cannot find")
                            print("   a file for this year!")
                            print("Variables requested: ",variables_to_extract)
                            print("Variable location: ",sim_parms.variables_in_which_file)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        else:
                            print("No stomate file and no stomate variables requested.  Skipping.")
                        #endif 
                    #endif
                #endtry

                # copy all the dimensions
                for name, dimension in srcnc.dimensions.items():
                    # It's possible the dimension already exists because it was in the other file
                    if name not in dstnc.dimensions.keys():
                        dstnc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
                        
                        # If a variable exists for this dimensions, copy it
                        if name in srcnc.variables.keys():
                            variable=srcnc[name]
                            x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
                            # copy variable attributes all at once via dictionary
                            dstnc[name].setncatts(srcnc[name].__dict__)
                            # Copy the values for all variables except the timecoord,
                            # since that will change
                            if name != timecoord:
                                dstnc[name][:]=srcnc[name][:]
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

                extracted_variables[name]=extracted_variable_class(name, variable.dimensions,variable.datatype,timecoord, latcoord, loncoord, vegetcoord,len(srcnc[latcoord]),len(srcnc[loncoord]),len(srcnc[vegetcoord]))

                # And create the variable in the file
                x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dstnc[name].setncatts(srcnc[name].__dict__)

                srcnc.close()
                
            #endfor

        #endif


    #endif

    ###### Now we need to figure out our time axis.

    #print("In total, we have {} time points.".format(ntimepoints))
    #print(timeaxis)
    # We have a list of values and we have a units string.  Check to see what
    # kind of frequency we are dealing with.  Also do some basic checks to
    # see if the axis makes sense.
    if sim_parms.fix_time_axis:
        # This is used when we know we have a problem with the time axis.
        # Look at the starting date, the end date, the number of steps,
        # and the origin date and create a time axis.
        
        # This timeseries may have messed up values, but by created it,
        # we can parse the time units string and figure out how to make
        # a new one.  Most values will not be trustworthy.
        combined_timeaxis=time_axis(timeaxis_values,time_units,lstop=False)

        if len(timeaxis_values) == sim_parms.ntotal_data_years:
            timeaxis_values=create_annual_axis(sim_parms.syear,sim_parms.eyear,combined_timeaxis.oyear,combined_timeaxis.omonth,combined_timeaxis.oday,combined_timeaxis.ohour,combined_timeaxis.omin,combined_timeaxis.osec,combined_timeaxis.ounits)
#        elif len(timeaxis_values) == sim_parms.ntotal_data_years*12:
#            sys.exit(1)
        else:
            print("Axis doesn't seem to be annual here. Not sure what to do.")
            print("Starting year, ending year, total years: ",sim_parms.syear,sim_parms.eyear,sim_parms.ntotal_data_years)
            print("Length of aggregated time axis: ",len(timeaxis_values))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # The calendar attribute may be messed up.  Since I've recreated it, reset that.
        dstnc[timecoord].setncatts({"calendar" : "GREGORIAN"})

        combined_timeaxis=time_axis(timeaxis_values,time_units)

    else:
        combined_timeaxis=time_axis(timeaxis_values,time_units)

    #endif

    # Do we have the dates we expect to have?
    if combined_timeaxis.syear != syear or combined_timeaxis.eyear != eyear:
        print("Based on the command line, I expect data from year {} to {}.".format(syear,eyear))
        print("However, after looking through the data files, I found data from year {} to {}.".format(combined_timeaxis.syear,combined_timeaxis.eyear))
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    # Now we allocate arrays to hold all of our data.
    for name in variables_to_extract:
        if name in skipped_variables:
            continue
        #endif
        extracted_variables[name].create_data_array(combined_timeaxis.values,combined_timeaxis.timeunits)
        
    #endfor

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
            varloc=sim_parms.variables_in_which_file[varname]
            if varloc == "sechiba":
                varlocation[varname]=sechiba_file_name
            else:
                varlocation[varname]=stomate_file_name
            #endif
        #endif

        # Open the file and get some information about coordinate names
        print("Extracting time data from: ",stomate_file_name)
        srcnc = NetCDFFile(stomate_file_name,"r")
        timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)

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
            timecoord,latcoord,loncoord,vegetcoord=find_orchidee_coordinate_names(srcnc)

            itimestep2=newtimesteprange[0]
            for jtimestep in range(len(srcnc[timecoord][:])):
                extracted_variables[varname].fill_data_array(srcnc[varname][:],itimestep2,jtimestep)
                itimestep2=itimestep2+1
            #endfor

            srcnc.close()
        #endfor


    #endfor


    # Sometimes, we may want to change monthly data into annual data.  Let's check that here.
    if sim_parms.force_annual:
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
        # This is just the same as the combined_timeaxis
        new_timeaxis=time_axis(timeaxis_values,time_units)

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
        
    dstnc.close()

#    print("ijofiez ")
#    sys.exit(1)

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
            elif timeseries_variable == "TWBR":
                if not np.ma.is_masked(srcnc["TWBR"][0, ilat, ilon]):

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

