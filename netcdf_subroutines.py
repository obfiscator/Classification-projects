# I am trying to put here common operations that I need to do
# on NetCDF files
from netCDF4 import Dataset as NetCDFFile 
import netCDF4 
import sys,traceback
from time_axis_manipulations import create_monthly_axis,time_axis
import numpy as np
from datetime import date


__latitude_names__=["latitude","lat","Lat"]
__longitude_names__=["longitude","lon","Lon"]
__time_names__=["time","Time","time_counter","mon","year"]
__veget_names__=["veget"]

###### A class that holds labels we use in the VERIFY files
class harmonized_netcdf:
    def __init__(self):

        # Required dimensions
        self.latcoord="latitude"
        self.loncoord="longitude"
        self.timecoord="time"
        self.nb2_coord="nv"

        # Required variables
        self.timebounds="time_bounds"

        # Dimension information that does not change
        self.nb2_units=''
        self.nb2_values=[1,2]
        self.latitude_units='degrees N'
        self.longitude_units='degrees E'

        # Time axis information
        self.oyear=1900
        self.omonth=1
        self.oday=1
        self.ounits="days"

        # Dimensions which are also variables
        self.full_units_string="{} since {}-{:02d}-{:02d}".format(self.ounits,self.oyear,self.omonth,self.oday)
        self.timecoord_atts={'long_name':self.timecoord,'bounds':self.timebounds,'units': self.full_units_string,'calendar':'standard','axis':'T'}
        self.timeaxis_type="f4"
        self.latcoord_type="f4"
        self.loncoord_type="f4"     

        # time bounds information
        self.timebounds_atts={'units':self.full_units_string,'calendar':'standard'}
        self.timebounds_type="f4"
        self.timebounds_dimensions=(self.timecoord, self.nb2_coord)

        # Preferred units
        self.fluxunits="kg C m-2 yr-1"
    #enddef

#endclass

###### A class to hold dimension information that we will write
# to NetCDF files
class dimensions_class:

    def __init__(self,values,name=None,units=None,atts=None,var_type=None,lcreate_var=False,lspecial_dim=''):
        harm_netcdf=harmonized_netcdf()
        self.lcreate_var=lcreate_var

        # For some dimensions, I want to create a harmonized format.
        # This is the best we I've found to do it and hide much of the
        # nuts and bolts from the calling routine.
        if lspecial_dim == "time":
            self.units=self.check_standard_values(units,harm_netcdf.full_units_string)
            self.atts=self.check_standard_values(atts,harm_netcdf.timecoord_atts)
            self.var_type=self.check_standard_values(var_type,harm_netcdf.timeaxis_type)
            self.name=self.check_standard_values(name,harm_netcdf.timecoord)
            
            self.values=values[:]

            # Check to make sure we have some standard attributes.
            required_atts=["units","bounds","calendar","axis"]
            for att in required_atts:
                if att not in self.atts.keys():
                    print("***************************************")
                    print("Did not find required time axis attribute {} in dimension.")
                    print("Attribute: ",att)
                    print("Requested attributes: ",self.atts.keys())
                    print("***************************************")
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endfor
        elif lspecial_dim == "nb2":
            self.units=self.check_standard_values(units,harm_netcdf.nb2_units)
            self.atts=self.check_standard_values(atts,harm_netcdf.timebounds_atts)
            self.var_type=self.check_standard_values(var_type,harm_netcdf.timebounds_type)
            self.name=self.check_standard_values(name,harm_netcdf.nb2_coord)
            
            self.values=harm_netcdf.nb2_values[:]

            # No required attributes for nb2.  This is only used in the
            # time_bounds variable, since that variable holds a lower bound
            # and an upper bound for every time point.

        elif lspecial_dim == "latitude":
            self.units=self.check_standard_values(units,harm_netcdf.latitude_units)

            self.atts=atts
            self.var_type="f4"
            self.name=self.check_standard_values(name,harm_netcdf.latcoord)
            
            self.values=values[:]

        elif lspecial_dim == "longitude":
            self.units=self.check_standard_values(units,harm_netcdf.longitude_units)
            self.atts=atts
            self.var_type="f4"
            self.name=self.check_standard_values(name,harm_netcdf.loncoord)
            
            self.values=values[:]

        else:
            self.name=name
            self.units=units
            self.atts=atts
            self.var_type=var_type
        #endif

        # Make sure the units are part of the attributes for the automation
        # later.
        if self.atts is None:
            self.atts={}
            self.atts["units"]=self.units
        else:
            if "units" not in self.atts.keys():
                self.atts["units"]=self.units
            else:
                print("Not overwiting unit attribute.")
                print("self.units ",self.units)
                print("atts: ",self.atts)
            #endif
        #endif

    #enddef

    # Print out an error flag if the two values are not equal, and return
    # one of the values depending.  This is something we do a lot in the
    # above routines.
    def check_standard_values(self,req_value,standard_value):
        
        if req_value is not None:
            if req_value != standard_value:
                print("--- Requesting non-standard value.")
                print("    Value requested: ",req_value)
                print("    Standard value: ",standard_value)
                traceback.print_stack(file=sys.stdout)
            #endif
            return req_value
        else:
            return standard_value
        #endif
            
    #enddef

#endclass

# Check to see if we have some things we need for the 2D file.
# Need to have four dimensions, at least: time, lon, lat, nv
def check_2D_file_dimensions(dimensions):
    harm_netcdf=harmonized_netcdf()
    required_dimensions=[harm_netcdf.timecoord,harm_netcdf.latcoord,harm_netcdf.loncoord,harm_netcdf.nb2_coord]
    for dim in required_dimensions:
        if dim not in dimensions.keys():
            print("*********************************************")
            print("I seem to be missing a dimension: ",dim)
            print("Dimensions I have: ",dimensions.keys())
            print("*********************************************")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    #endfor
#endif

###### A class to hold variable information that we will write
# to NetCDF files
class variables_class:
    def __init__(self,name,values,units,dimensions,long_name,atts=None,var_type='f4'):
        self.name=name
        self.units=units
        self.var_type=var_type
        self.dimensions=dimensions
        self.values=values
        self.long_name=long_name

        if atts is None:
            self.atts={}
        else:
            self.atts=atts
        #endif

        self.check_atts("units",units)
        self.check_atts("long_name",long_name)


        # Take on fill value and missing value attributes.
        if self.var_type == 'f4':
            missing_value=netCDF4.default_fillvals['f4']
            fillvalue=missing_value
        else:
            print("***********************************************")
            print("Do not know how to select values for this var_type: ",self.var_type)
            print("***********************************************")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        self.check_atts("missing_value",missing_value)

        # Store this seperately, since it is crashing when I try to 
        # write the attribute dictionary
        self.fillvalue=missing_value
#        self.check_atts("_FillValue",fillvalue)

    #enddef

    # Check to see if an attribute exists.  If not, add it.  If it
    # does, warn that we are overwriting and stop.  I'm not sure what
    # I want to do in this case yet.  This does happen with timebounds.
    def check_atts(self,att_name,att_value):
        if att_value is None:
            return
        #endif

        if att_name in self.atts.keys():
            print("***********************************************")
            print("Trying to overwrite an attribute in a variable.")
            print(att_name)
            print(self.atts.keys())
            #traceback.print_stack(file=sys.stdout)
            #sys.exit(1)
        else:
            self.atts[att_name]=att_value
        #endif

    #enddef

#endclass

# Find the name of the time axis.
def find_time_coordinate_name(srcnc,lcheck_units=False):

    global __time_names__

    timecoord=find_variable(__time_names__,srcnc,True,"seconds since 1901-01-01 00:00:00",lcheck_units=lcheck_units)

    return timecoord
#enddef

# Find the name of the longitude axis.
def find_longitude_coordinate_name(srcnc,lcheck_units=False):

    global __longitude_names__

    loncoord=find_variable(__longitude_names__,srcnc,True,"degrees_east",lcheck_units)

    return loncoord
#enddef

# Find the name of the latitude axis.
def find_latitude_coordinate_name(srcnc,lcheck_units=False):

    global __latitude_names__

    latcoord=find_variable(__latitude_names__,srcnc,True,"degrees_north",lcheck_units)

    return latcoord
#enddef

# Find the name of some standard coordinates that we expect, confirming
# units.
# Return the name of the coordinate
def find_orchidee_coordinate_names(srcnc,check_units=True):

    # for each file time, we need to check different things.
    global __latitude_names__
    global __longitude_names__
    global __time_names__
    global __veget_names__

    timecoord=find_variable(__time_names__,srcnc,True,"seconds since 1901-01-01 00:00:00",lcheck_units=check_units)
    loncoord=find_variable(__longitude_names__,srcnc,True,"degrees_east")
    latcoord=find_variable(__latitude_names__,srcnc,True,"degrees_north")
    vegetcoord=find_variable(__veget_names__,srcnc,True,"1")

    return timecoord,latcoord,loncoord,vegetcoord
#enddef


# with a list of possible variable names and a unit definition, try to find
# what it's actually called
def find_variable(varnames,srcnc,is_dim,desired_units,lcheck_units=True):

    if not varnames:
        print("Not sure how, but you passed an empty string to find_variable!")
        print(varnames)
        print(is_dim)
        print(desired_units)
        print("in file: ",srcnc.filepath())
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    actual_name=""
    lfound_var=False
    lfound_dim=False

    for varname in varnames:

        # first check to see if it's a dimension
        if is_dim:
            for name, dimension in srcnc.dimensions.items():
                if name == varname:
                    lfound_dim=True
                    actual_name=varname
                #endif
            #endfor
        #endif

        # now check to see if it's a variable
        try:
            tempdata=srcnc[varname][:]
            lfound_var=True
            actual_name=varname
            if lfound_dim:
                if actual_name != varname:
                    print("Seems you are looking for a dimension.")
                    print("I found a variable named this, but not a dimension!")
                    print("Not sure how this is possible, so stopping.")
                    print("Variable I found: {}, dimension I found: {}".format(varname,actual_name))
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endif
        except:
            a=1
        #endtry

    #endfor

    if not actual_name:
        print("Do we not have one of these variables in this file: ",varnames)
        print("I could not find it.")
        print("in file: ",srcnc.filepath())
        print(srcnc.variables)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
        
    #endif

    if not lfound_var and not is_dim:
        print("Do we not have one of these variables in this file? ",varnames)
        print("in file: ",srcnc.filepath())
        print(srcnc.variables)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    if is_dim and not lfound_dim:
        print("Do we not have one of these dimensions in this file?",varnames)
        print("in file: ",srcnc.filepath())
        print(srcnc.dimensions)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    # Check the units.  Sometimes we have a dimension that does not
    # have a variable associated with it.
    if lfound_var:
        try:
            unitstring=srcnc[actual_name].units
            lhave_units=True

        except:
            lhave_units=False
        #endtry

        if lhave_units:
            if lcheck_units:
                if unitstring != desired_units:
                    print("Our units for {} don't match!".format(actual_name))
                    print("Desired units: ",desired_units)
                    print("Actual units: ",unitstring)
                    print("in file: ",srcnc.filepath())
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endif
        else:

            print("Do we not have a units attribute for {0}?".format(actual_name))
            print("in file: ",srcnc.filepath())
            if lcheck_units:
                print(srcnc[actual_name])
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        #endif
    #endif

    return actual_name
#enddef

# Take two file identifies, and copy all of the information in one
# to the other, except the time axis.  Use this to regrid the
# time axis.
# This copies all the variable metainformation but does not copy
# data for any variables with a time axis.
def copy_all_except_time_axis(srcnc,dstnc,timecoord):

    # copy global attributes all at once via dictionary
    dstnc.setncatts(srcnc.__dict__)
    
    # copy dimensions
    for name, dimension in srcnc.dimensions.items():
        if name == timecoord:
            dstnc.createDimension(name,None)
        else:
            dstnc.createDimension(name, (len(dimension)))
        #endif
    #endfor

    # Copy variables
    for name, variable in srcnc.variables.items():
        
        x = dstnc.createVariable(name, variable.datatype, variable.dimensions)
        # copy variable attributes all at once via dictionary
        dstnc[name].setncatts(srcnc[name].__dict__)
        
        # if it's not a time variable, copy the data
        if timecoord not in variable.dimensions:
            print("Copying data for variable {}.".format(name))
            dstnc[name][:]=srcnc[name][:]
        else:
            print("Variable {} has a time coordinate.  Will regrid.".format(name))
        #endif
    #endfor

#enddef

############################
# This takes a variable and extracts only the data between certain years, 
# outputing that data.  Works for monthly data with a well-defined time
# axis at the moment.  Outputs the new time coordinate axis.
def select_data_years(input_data,input_timeaxis,desired_syear,desired_eyear,input_mask_2D,plotting_grid=None):
    
    if input_timeaxis.timestep != "1M":
        print("Cannot current select data years for non-monthly data.")
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    output_monthly_axis_values=create_monthly_axis(desired_syear,desired_eyear,input_timeaxis.oyear,input_timeaxis.omonth,input_timeaxis.oday,input_timeaxis.ohour,input_timeaxis.omin,input_timeaxis.osec,input_timeaxis.ounits)
    output_timeaxis=time_axis(output_monthly_axis_values,input_timeaxis.timeunits)

    nmonths=len(output_monthly_axis_values)

    # create the mask from the input.  Take a single year mask, and then
    # add a time axis for the number of months that we have
    # Let's harmonize the fill values.
    try:
        print("Old fill value in select_data_years: ",input_data.fill_value)
        np.ma.set_fill_value(input_data,netCDF4.default_fillvals['f4'])
        print("New fill value in select_data_years: ",input_data.fill_value)
    except:
        print("Is our input data not a masked array?")
        raise
        #sys.exit(1)
    #endtry

    if len(input_data.shape) == 3:
        limits=input_data.shape

        # I assume that the time axis is the first one.
        output_data=np.zeros((nmonths,limits[1],limits[2]))
        output_mask=np.zeros((nmonths,limits[1],limits[2]))
    else:
        print("Cannot deal with this input data shape in convert_annual_variable_to_monthly.")
        print(input_data.shape)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    # Get the indices of the points that we want.
    # I assume the points we want are continuous
    retreive_indices=[]

    # Notice that I constructed my time axis using the same units as the old time axis.
    # And I have already checked for monthly data.  So there should be a faster way to do this.
    test_vals=abs(input_timeaxis.values[:]-output_monthly_axis_values[0])
    def my_min(sequence):
        low = sequence[0] # need to start with some value
        low_index=0
        for i,ival in enumerate(sequence):
            if ival < low:
                low = ival
                low_index=i
            #endif
        #endfor
        return low_index
    #enddef
    starting_index=my_min(test_vals)
    if input_timeaxis.ounits =="seconds":
        if test_vals[starting_index] >= 3600.0*24.0*16.0:
            print("Not sure we found a good starting index!")
            print(output_monthly_axis_values[0])
            print(input_timeaxis.values[:])
            print(test_vals)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    elif input_timeaxis.ounits =="days":
        if test_vals[starting_index] >= 16.0:
            print("Not sure we found a good starting index!")
            print(output_monthly_axis_values[0])
            print(input_timeaxis.values[:])
            print(test_vals)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    else:
        print("Don't understand the time axis units.")
        print(input_timeaxis.ounits)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    retreive_indices=list(range(starting_index,starting_index+nmonths))

    print("Taking indices: ",retreive_indices)
    print(input_data.shape)

    if False:
        ### This loop is really slow!  But completely accurate.
        # Loop
        yy=desired_syear
        mm=1
        for iindex in range(output_data.shape[0]):
            # This keeps track of our month and year for the output data.
        

            # Which means we need to loop through the input data to see if
            # we have a matching datapoint.
            jindex=-1
            for itime,ival in enumerate(input_timeaxis.values[:]):
                jyy=input_timeaxis.get_year(ival)
                if jyy == yy:
                    
                    # Found the year!  But do we have the correct month?
                    startmonthdate=date(yy, mm, 1)-date(input_timeaxis.oyear, input_timeaxis.omonth, input_timeaxis.oday)
                    if mm == 12:
                        endmonthdate=date(yy+1, 1, 1)-date(input_timeaxis.oyear, input_timeaxis.omonth, input_timeaxis.oday)
                    else:
                        endmonthdate=date(yy, mm+1, 1)-date(input_timeaxis.oyear, input_timeaxis.omonth, input_timeaxis.oday)
                    #endif
                    
                    if startmonthdate.days <= ival and endmonthdate.days > ival:
                        jindex=itime
                        ndays_month=endmonthdate.days-startmonthdate.days
                        #print("Found a time!",yy,mm,startmonthdate.days,ival,endmonthdate.days,ndays_month)
                    #endif
               


            # Found a point!
            retreive_indices.append(jindex)
            #output_data[iindex,:,:]=input_data[jindex,:,:].copy().filled(np.nan)

            # Run a test and plot some maps.  T
            if plotting_grid:
                print("Trying some plots")
                plotting_grid.plotmap(input_data[jindex,:,:],filename="epic_input_npp.png")
                plotting_grid.plotmap(output_data[iindex,:,:],filename="epic_output_npp.png")
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            # Copy the mask over directly
            #            print("jifoez ",output_mask.shape,input_data.shape,iindex,jindex)
            #            print("jifoez2 ",input_data.mask.shape)
            #output_mask[iindex,:,:]=input_mask_2D

            mm=mm+1
            if mm == 13:
                yy=yy+1
                mm=1
            #endif
            print("jifoez ",yy,mm)
        #endfor
    #endif

    # Okay, now get the data
    if len(retreive_indices) != nmonths or retreive_indices[-1] > input_data.shape[0]:
        print("Something wrong with index retrival!")
        print("Months found: ",len(retreive_indices))
        print("Supposed to have: ",nmonths)
        print("Indices found: ",retreive_indices)
        print("Input data shape: ",input_data.shape)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

                             
    output_data[:,:,:]=input_data[retreive_indices,:,:].copy().filled(np.nan)
    for ii in range(output_mask.shape[0]):
        output_mask[ii,:,:]=input_mask_2D
    #endif
    
    output_masked_data=np.ma.array(output_data,mask=output_mask)

    return output_masked_data,output_timeaxis
    

#enddef

######
def create_time_axis_from_netcdf(filename,lprint=True):
    srcnc = NetCDFFile(filename)

    timecoord=find_time_coordinate_name(srcnc)

    timeaxis=time_axis(srcnc[timecoord][:],srcnc[timecoord].units,lprint=lprint)

    srcnc.close()

    return timeaxis

#enddef

################################
# Verify some attributes about a NetCDF file for VERIFY.
# This used to be found in explore_epic_subroutines.py
###############################
def verify_file(src,cflag,lcheck_units=True):

    # for each file time, we need to check different things.
    latitude_names=["latitude","lat"]
    longitude_names=["longitude","lon"]
    time_names=["time","Time","time_counter"]
    countrycoords=["country"]
    countrynames=["country_name"]
    countrycodes=["country_code"]
    countrymasks=["country_mask"]

    # Fix the time coord
    if cflag.lower() == "cropland_area":
        areanames=["CROPLAND_AREA"]
        timecoord=find_variable(time_names,src,True,"days since 1900-01-01")
        loncoord=find_variable(longitude_names,src,True,"degrees_east")
        latcoord=find_variable(latitude_names,src,True,"degrees_north")
        areaname=find_variable(areanames,src,False,"m2")

        #surfaceareanames=["SURFACE_AREA"]
        #surfaceareaname=find_variable(surfaceareanames,src,False,"m2")
        surfaceareaname=""

        return timecoord,latcoord,loncoord,areaname,surfaceareaname
     
    elif cflag.lower() == "grassland_area":
        areanames=["GRASSLAND_AREA"]
        timecoord=find_variable(time_names,src,True,"days since 1900-01-01")
        loncoord=find_variable(longitude_names,src,True,"degrees_east")
        latcoord=find_variable(latitude_names,src,True,"degrees_north")
        areaname=find_variable(areanames,src,False,"m2")

        #surfaceareanames=["SURFACE_AREA"]
        #surfaceareaname=find_variable(surfaceareanames,src,False,"m2")
        surfaceareaname=""

        return timecoord,latcoord,loncoord,areaname,surfaceareaname
     
    elif cflag.lower() == "area":
        timecoord=find_variable(time_names,src,True,"days since 1900-01-01")
        loncoord=find_variable(longitude_names,src,True,"degrees_east")
        latcoord=find_variable(latitude_names,src,True,"degrees_north")

        surfaceareanames=["SURFACE_AREA"]
        surfaceareaname=find_variable(surfaceareanames,src,False,"m2")

        return timecoord,latcoord,loncoord,surfaceareaname
     
    elif cflag.lower() == "mask":
        loncoord=find_variable(longitude_names,src,True,"",lcheck_units=lcheck_units)
        latcoord=find_variable(latitude_names,src,True,"",lcheck_units=lcheck_units)
        countrycoord=find_variable(countrycoords,src,True,"",lcheck_units=lcheck_units)
        countryname=find_variable(countrynames,src,False,"",lcheck_units=lcheck_units)
        countrycode=find_variable(countrycodes,src,False,"",lcheck_units=lcheck_units)
        countrymask=find_variable(countrymasks,src,False,"",lcheck_units=lcheck_units)
        return latcoord,loncoord,countrycoord,countryname,countrycode,countrymask

    elif cflag.lower() == "latlontime":
        timecoord=find_variable(time_names,src,True,"days since 1900-01-01")
        loncoord=find_variable(longitude_names,src,True,"degrees_east")
        latcoord=find_variable(latitude_names,src,True,"degrees_north")
        return latcoord,loncoord,timecoord

    elif cflag.lower() == "latlon":        
        loncoord=find_variable(longitude_names,src,True,"degrees_east")
        latcoord=find_variable(latitude_names,src,True,"degrees_north")
        return latcoord,loncoord

    else:
        print("Do not recognize flag in verify_file!")
        print(cflag)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif
#enddef

###########################
# Given a set of variables, monthly data, and metadata, create
# a NetCDF file that conforms to VERIFY standards for country
# total timeseries
def create_countrytot_file(sim_params,mask_file,ldebug=False):

    # This depends on the countries we have.
    if mask_file in ["/home/dods/verify/VERIFY_INPUT/COUNTRY_MASKS/EU_16othercountries.nc"]:
        output_file_name=sim_params.output_filename_base.substitute(filetype="CountryTotWithOutEEZGlobal")
    else:
        output_file_name=sim_params.output_filename_base.substitute(filetype="CountryTotWithOutEEZ")
    #endif

    print("Creating the file {}".format(output_file_name))

    # I need to create a time axis.  I assume the incoming data is annual,
    # since it comes from a spreadsheet.  It needs to be monthly.

    # Open up the output file
    dstCT = NetCDFFile(output_file_name,"w")

    # I have four axes: country, time, strlength, and nb2 (for time_bounds)
    time_coord=sim_params.time_refcoord_name
    time_bounds_coord=sim_params.timebounds_refcoord_name
    country_coord=sim_params.country_refcoord_name
    strlength_coord=sim_params.strlen_refcoord_name
    nb2_coord=sim_params.nb2_refcoord_name
    ncountries=len(sim_params.variables_in_file[0].country_region_codes)
    str_length=sim_params.strlen_value    

    dstCT.createDimension(time_coord, None)
    dstCT.createDimension(country_coord, ncountries)
    dstCT.createDimension(strlength_coord, str_length)
    dstCT.createDimension(nb2_coord, sim_params.nb2_value)

    # Now create the time and time_bounds variables and write them
    # to the file
    x = dstCT.createVariable(time_coord, sim_params.timecoord_type, (time_coord))
    dstCT[time_coord].setncatts(sim_params.timecoord_atts)
    dstCT[time_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.values
    ntimes=len(sim_params.variables_in_file[0].monthly_timeaxis.values)

    x = dstCT.createVariable(sim_params.timebounds_refcoord_name, sim_params.timebounds_type, sim_params.timebounds_dimensions)
    dstCT[time_bounds_coord].setncatts(sim_params.timebounds_atts)
    dstCT[time_bounds_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.timebounds_values

    # I need to output the country codes and the country names as two
    # different variables
    dstCT.createVariable("country_code", "S1", (country_coord, strlength_coord))
    dstCT.createVariable("country_name", "S1", (country_coord, strlength_coord))
    varname_class=sim_params.variables_in_file[0]
    for idx in range(ncountries): 
        ccode=varname_class.country_region_codes[idx]
        cname=varname_class.country_region_data[ccode].long_name
        #print("On ",ccode,cname,len(country_coord))
        for jdx in range(str_length):
            if jdx >= len(ccode):
                if jdx >= len(cname):
                    continue
                else:
                    dstCT.variables["country_name"][idx, jdx] = cname[jdx]
                #endif
                continue
            #endif
            dstCT.variables["country_code"][idx, jdx] = ccode[jdx]
            dstCT.variables["country_name"][idx, jdx] = cname[jdx]
        #endfor
    #endfor

    # Now we write the actual variables.  
    for ivar in range(len(sim_params.variables_in_file)):

        varname_class=sim_params.variables_in_file[ivar]

        # Do a check to make sure the data is what we expect.
        if varname_class.monthly_data.shape[1] != ncountries:
            print("Variable dimensions are not what we expect!")
            print("Variable input name, output name: ",varname_class.input_varname,varname_class.output_varname)
            print("ntimes, ncountries: ",ntimes,ncountries)
            print("variable shape: ",varname_class.monthly_data.shape)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        if varname_class.monthly_data.shape[0] != ntimes:
            print("Variable dimensions are not what we expect!")
            print("Variable input name, output name: ",varname_class.input_varname,varname_class.output_varname)
            print("ntimes, ncountries: ",ntimes,ncountries)
            print("variable shape: ",varname_class.monthly_data.shape)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if ldebug:
            # What is the index of this region?
            ccode_test=sim_params.debug_country
            test_year=2010
            cindex=varname_class.country_region_codes.index(ccode_test)
            print("-- Data: ",varname_class.data[:,cindex])
            years=list(range(varname_class.syear,varname_class.eyear+1))
            yindex=years.index(test_year)
            print("-- Data ({}): ".format(test_year),varname_class.data[yindex,cindex])
        #endif

        dstCT.createVariable(varname_class.output_varname, varname_class.nc_type, (time_coord,country_coord))
        dstCT[varname_class.output_varname].setncatts({"units":varname_class.output_units, "long_name":varname_class.long_name})
        dstCT[varname_class.output_varname][:,:]=varname_class.monthly_data[:,:]
    #endfor

    # Now print some additional meta information
    for iinfo,cinfo in enumerate(sim_params.countrytot_info):
        attname="info{}".format(iinfo+1)
        dstCT.setncatts({attname: cinfo})
    #endfor

    # And print information about missing data for regions
    sim_params.print_missing_region_information(dstCT)

    dstCT.close()

#enddef

###########################
# Given a set of variables, monthly data, and metadata, create
# a NetCDF file that conforms to VERIFY standards for spatially
# explicit timeseries
def create_2D_file(sim_params):
    
    output_file_name=sim_params.output_filename_base.substitute(filetype="2D")
    print("Creating the file {}".format(output_file_name))

    # Open up the output file
    dst2D = NetCDFFile(output_file_name,"w")

    # I have four axes: time, latitude, longitude, and nb2 (for time_bounds)
    time_coord=sim_params.time_refcoord_name
    time_bounds_coord=sim_params.timebounds_refcoord_name
    lat_coord=sim_params.lat_refcoord_name
    lon_coord=sim_params.lon_refcoord_name
    nb2_coord=sim_params.nb2_refcoord_name

    dst2D.createDimension(time_coord, None)
    dst2D.createDimension(lat_coord, sim_params.variables_in_file[0].nlats)
    dst2D.createDimension(lon_coord, sim_params.variables_in_file[0].nlons)
    dst2D.createDimension(nb2_coord, sim_params.nb2_value)

    # Now create the time and time_bounds variables and write them
    # to the file
    x = dst2D.createVariable(time_coord, sim_params.timecoord_type, (time_coord))
    dst2D[time_coord].setncatts(sim_params.timecoord_atts)
    dst2D[time_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.values

    x = dst2D.createVariable(sim_params.timebounds_refcoord_name, sim_params.timebounds_type, sim_params.timebounds_dimensions)
    dst2D[time_bounds_coord].setncatts(sim_params.timebounds_atts)
    dst2D[time_bounds_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.timebounds_values

    # Latitude and longitude values are also pretty easy, with some metadata.
    dst2D.createVariable(lat_coord, "f4", (lat_coord))
    dst2D.createVariable(lon_coord, "f4", (lon_coord))
    dst2D.variables[lat_coord][:]=sim_params.variables_in_file[0].lat
    dst2D.variables[lon_coord][:]=sim_params.variables_in_file[0].lon

    # add some metadata
    

    # Now we write the actual variables.  
    for ivar in range(len(sim_params.variables_in_file)):

        varname_class=sim_params.variables_in_file[ivar]

        
        dst2D.createVariable(varname_class.output_varname, varname_class.nc_type, (time_coord,lat_coord,lon_coord))
        dst2D[varname_class.output_varname].setncatts({"units":varname_class.output_units_2D, "long_name":varname_class.long_name_2D})
        dst2D[varname_class.output_varname][:,:,:]=varname_class.monthly_2D_data[:,:,:]
    #endfor

    # Now print some additional meta information
    for iinfo,cinfo in enumerate(sim_params.spatial_info):
        attname="info{}".format(iinfo+1)
        dst2D.setncatts({attname: cinfo})
    #endfor

    # And print information about missing data for regions
    sim_params.print_missing_region_information(dst2D)

    dst2D.close()

#enddef

###########################
# Given a set of variables, monthly data, and metadata, create
# a NetCDF file that conforms to VERIFY standards for CountryTot
# timeseries.
# The input is a data array for each variable with dimensions: 
# varname_class.monthly_data[imonth,cindex]
#    varname_class.monthly_timeaxis
#    cindex=varname_class.country_region_codes.index(ccode)
def create_2Dmod_file(sim_params):
    
    output_file_name=sim_params.output_filename_base.substitute(filetype="2Dmod")
    print("Creating the file {}".format(output_file_name))

    # Open up the output file
    dst2D = NetCDFFile(output_file_name,"w")

    # I have four axes: time, latitude, longitude, and nb2 (for time_bounds)
    time_coord=sim_params.time_refcoord_name
    time_bounds_coord=sim_params.timebounds_refcoord_name
    nb2_coord=sim_params.nb2_refcoord_name
    # Have all the country information here
    country_name_coord="country_name"
    country_code_coord="country_code"
    strlength=50
    strlength_coord="strlength"
    country_coord="country"
    ncountries=sim_params.variables_in_file[0].monthly_data.shape[1]

    dst2D.createDimension(time_coord, None)
    dst2D.createDimension(nb2_coord, sim_params.nb2_value)
    dst2D.createDimension(country_coord, ncountries)
    dst2D.createDimension(strlength_coord, strlength)

    # Now create the time and time_bounds variables and write them
    # to the file
    x = dst2D.createVariable(time_coord, sim_params.timecoord_type, (time_coord))
    dst2D[time_coord].setncatts(sim_params.timecoord_atts)
    dst2D[time_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.values

    x = dst2D.createVariable(sim_params.timebounds_refcoord_name, sim_params.timebounds_type, sim_params.timebounds_dimensions)
    dst2D[time_bounds_coord].setncatts(sim_params.timebounds_atts)
    dst2D[time_bounds_coord][:]=sim_params.variables_in_file[0].monthly_timeaxis.timebounds_values

    # Not latitude or longitude here.  But we do have country names and codes.
    dst2D.createVariable(country_name_coord, "S1", (country_coord,strlength_coord))
    dst2D.createVariable(country_code_coord, "S1", (country_coord,strlength_coord))

    for idx in range(ncountries): 
        ccode=sim_params.variables_in_file[0].country_region_codes[idx]
        for jdx in range(len(ccode)):
            dst2D.variables[country_code_coord][idx, jdx] = ccode[jdx]
        #endfor
               
        cname=sim_params.variables_in_file[0].country_region_data[ccode].long_name
        for jdx in range(len(cname)):
            dst2D.variables[country_name_coord][idx, jdx] = cname[jdx]
        #endfor

    # add some metadata
    

    # Now we write the actual variables.  
    for ivar in range(len(sim_params.variables_in_file)):

        varname_class=sim_params.variables_in_file[ivar]

        
        dst2D.createVariable(varname_class.output_varname, varname_class.nc_type, (time_coord,country_coord))
        dst2D[varname_class.output_varname].setncatts({"units":varname_class.output_units, "long_name":varname_class.long_name})
        print("jiofe ",ivar,varname_class.monthly_data.shape)
        print(len(sim_params.variables_in_file[0].monthly_timeaxis.values),ncountries)

        dst2D[varname_class.output_varname][:,:]=varname_class.monthly_data[:,:]
    #endfor

    # Now print some additional meta information
    for iinfo,cinfo in enumerate(sim_params.spatial_info):
        attname="info{}".format(iinfo+1)
        dst2D.setncatts({attname: cinfo})
    #endfor

    # And print information about missing data for regions
    sim_params.print_missing_region_information(dst2D)

    dst2D.close()

#enddef

###########################
# Given a set of variables, dimensions, and metadata, create
# a NetCDF file that conforms to the standards passed to the routine.
# var_dim_metadata is something like harmonized_netcdf class
# dimensions is something like dimensions_class
# variables is something like variables_class
# and global_metadata is a dictionary with attribute:string
def create_2D_file_v2(output_file_name,dimensions,variables,global_metadata):

    # Open up the output file
    dst2D = NetCDFFile(output_file_name,"w")

    # Make sure we have some things we need.
    check_2D_file_dimensions(dimensions)
    harm_netcdf=harmonized_netcdf()

    for dim in dimensions.keys():
        print("Creating dimension: ",dim)
        dst2D.createDimension(dimensions[dim].name,len(dimensions[dim].values))
        if dimensions[dim].lcreate_var:
            x = dst2D.createVariable(dimensions[dim].name, dimensions[dim].var_type, (dim))
            dst2D[dim].setncatts(dimensions[dim].atts)
            dst2D[dim][:]=dimensions[dim].values
        #endif

    #endif

    # Now for the variables
    for var in variables.keys():
        print("Creating variable: ",var)

        # _FillValue causes continual problems.
        x = dst2D.createVariable(variables[var].name, variables[var].var_type, variables[var].dimensions,fill_value=variables[var].fillvalue)
        dst2D[var].setncatts(variables[var].atts)
        dst2D[var][:]=variables[var].values

    #endfor
    
    
    dst2D.close()

#enddef
