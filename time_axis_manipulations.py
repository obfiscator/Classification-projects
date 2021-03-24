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
from grid import Grid
import matplotlib as mpl
import re
from datetime import date,datetime,timedelta

class time_axis:
    def __init__(self, timeaxis_values,timeunits,lstop=True):
        self.timeunits = timeunits
        self.values = timeaxis_values
        self.ntimepoints=len(timeaxis_values)
        self.lstop=lstop

        # Check to see if the values make sense.  Calculate
        # the starting and ending year, and the interval
        # of the timesteps.  Check the units to see if they
        # are something we can understand.
        self.check_integrity()
        
        # Calculate the year and month of every timestep.  Not sure if this
        # is an expensive process, but we do use it fairly regularly?
        # I made it a little faster.  Also need to call it before timebounds,
        # since timebounds uses these values and it's much slower to recalculate.
        self.calculate_year_month()

        # Calculate the timebounds.
        print("Starting timebounds")
        self.calculate_timebounds()
        print("Ending timebounds")
        

    #enddef

    # Do some standard checks to see if this axis makes sense.
    def check_integrity(self):
        
        # First, are the values consistantly increasing?
        print("Checking to see if values on this time axis always increase.")
        for itime,time_val in enumerate(self.values):
            if itime == 0:
                continue
            #endif

            if time_val-self.values[itime-1] <= 0.0:
                print("Time axis values do not steadily increase!")
                print("Step {}: {}".format(itime,time_val))
                print("Previous step: ",self.values[itime-1])
                print("Whole timeseries: ",self.values)
                if self.lstop:
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endif
        #endif
        print("Yes, values on this time axis always increase.")

        print("Time axis units: ",self.timeunits)
        self.parse_units()

        print("Calculating the starting year.")
        self.syear=self.get_year(self.values[0])
        print("Starting year is {}".format(self.syear))
        print("Calculating the ending year.")
        self.eyear=self.get_year(self.values[-1])
        print("Ending year is {}".format(self.eyear))

        # Timestep
        print("Calculating the time between data points (1Y, 1M, 1D).")
        self.calculate_timestep()
        print("Timestep is {}.".format(self.timestep))

    #enddef

    # See if we can understand what units we are using.
    def parse_units(self):

        match = re.search(r"since", self.timeunits)
        if not match:
            print("Expecting time units of the form 'XXXX since NNNN' in parse_units.")
            print("Where XXXX can be seconds, days, months, years")
            print("   and NNNN can be a date like 1901, 1901-01, 1901-01-01, or 1901-01-01 00:00:00")
            print("Instead, found: ",self.timeunits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        match = re.search(r"(\w+) since", self.timeunits)
        if match:
           
            # These define the origin date for counting
            self.ounits=match[1]
        else:
            print("Why can I not find units in parse_units?")
            print(self.timeunits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        print("Time axis measured in {}.".format(self.ounits))

        # Now determine the origin date/time.  This can be a variety of formats.
        self.lfound_origin=False
        self.oyear=""
        self.omonth=""
        self.oday=""
        self.ohour=""
        self.omin=""
        self.osec=""

        match = re.search(r"\w+ since (\d\d\d\d)-(\d+)-(\d+)\s*$", self.timeunits)
        if match:
           
            # These define the origin date for counting the number of days
            self.oyear=int(match[1])
            self.omonth=int(match[2])
            self.oday=int(match[3])
            self.lfound_origin=True
        #endif

        match = re.search(r"\w+ since (\d\d\d\d)-(\d+)-(\d+) (\d+):(\d+):(\d+)\s*$", self.timeunits)
        if match:
           
            # These define the origin date for counting the number of days
            self.oyear=int(match[1])
            self.omonth=int(match[2])
            self.oday=int(match[3])
            self.ohour=int(match[4])
            self.omin=int(match[5])
            self.osec=int(match[6])
            self.lfound_origin=True
        #endif

        match = re.search(r"\w+ since (\d\d\d\d)-(\d+)-(\d+) (\d+):(\d+):(\d+).0\s+-0:00\s*$", self.timeunits)
        if match:
           
            # These define the origin date for counting the number of days
            self.oyear=int(match[1])
            self.omonth=int(match[2])
            self.oday=int(match[3])
            self.ohour=int(match[4])
            self.omin=int(match[5])
            self.osec=int(match[6])
            self.lfound_origin=True
        #endif

        if not self.lfound_origin:
            print("Could not determine time origin in parse_units!")
            print(self.timeunits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        
    #enddef

    # Given a value, find out the index where the value falls
    # between the bounds.  If the value falls outside the time 
    # axis, stop with an error.
    def get_index(self,value):
        index=-999
        for itime in range(len(self.values[:])):
            if value >= self.timebounds_values[itime,0] and value <= self.timebounds_values[itime,1]:
                index=itime
            #endif
        #endfor
    
        if index == -999:
            print("Problem finding the index of this value in this time array!  Perhaps the units are not what you are expecting?")
            print(value)
            for itime in range(len(self.values[:])):
                print(itime,self.timebounds_values[itime,0],self.values[itime],self.timebounds_values[itime,1])
            #endfor
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
      
        return index

    #endif

    # Given a year, find out the index.  Only to be used for annual time axis at the moment.
    def get_year_index(self,value):
        
        if self.timestep != "1Y":
            print("**********\nCannot find year index for non-annual data!")
            print("Current timestep: ",self.timestep)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        year_index=-1
        for tindex,tvalue in enumerate(self.values[:]):
            if value == self.get_year(tvalue):
                year_index=tindex
            #endif
        #endfor
        if year_index == -1:
            print("**********\nCould not find year index!")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        return year_index
    #enddef

    # Figure out what year a particular value is in for this time axis
    def get_year(self,value):

        # We should already have passed through the parse_units routine that figures out the origin.
        if self.ounits.lower() == "days":

            # Note that, in the case of day units, we don't care about hours, minutes, seconds
            for iyear in range(100,22000):
                start_year=date(iyear, 1, 1)-date(self.oyear, self.omonth, self.oday)
                end_year=date(iyear, 12, 31)-date(self.oyear, self.omonth, self.oday)
                if value >= start_year.days and value <= end_year.days:
                    return int(iyear)
                #endif
            #endfor

            print("I did not find a year for ",value)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        elif self.ounits.lower() == "seconds":

            # Note that, in the case of second units, we may have hours, minutes, seconds
            for iyear in range(100,22000):
                seconds_per_day=24.0*60.0*60.0
                if self.osec != "":
                    start_year=datetime(iyear, 1, 1,0,0,0)-datetime(self.oyear, self.omonth, self.oday, self.ohour, self.omin, self.osec)
                    end_year=datetime(iyear+1, 1, 1,0,0,0)-datetime(self.oyear, self.omonth, self.oday, self.ohour, self.omin, self.osec)
                else:
                    start_year=date(iyear, 1, 1)-date(self.oyear, self.omonth, self.oday)
                    end_year=date(iyear+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                #endif

                # I cannot use the seconds attribute of datetime here, since that just gives the
                # amount of seconds in the day.
                if value >= start_year.days*seconds_per_day and value <= end_year.days*seconds_per_day:
                    return int(iyear)
                #endif
            #endfor

        else:
            print("Cannot yet deal with these units in get_year!")
            print(self.ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        print("I did not find a year in self.get_year for ",value)
        
        print(self.ounits,self.oyear,self.omonth,self.oday)

        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    #enddef

    # Figure out what month a particular value is in for this time axis
    def get_month(self,value):

        # If we have a yearly time axis, just make the month June.
        if self.timestep == "1Y":
            return 6
        #endif

        # We should already have passed through the parse_units routine that figures out the origin.
        if self.ounits.lower() == "days":

            # Note that, in the case of day units, we don't care about hours, minutes, seconds
            if value > 0.0:
                syear=self.oyear
            else:
                syear=100
            #endif
            for iyear in range(syear,22000):
                for imonth in range(1,13):
                    if imonth != 12:
                        start_date=date(iyear, imonth, 1)-date(self.oyear, self.omonth, self.oday)
                        end_date=date(iyear, imonth+1, 1)-date(self.oyear, self.omonth, self.oday)
                    else:
                        start_date=date(iyear, imonth, 1)-date(self.oyear, self.omonth, self.oday)
                        end_date=date(iyear+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                    #endif
                    #print(value,start_date.days,end_date.days)
                    if value >= start_date.days and value <= end_date.days:
                        return int(imonth)
                    #endif
                #endfor
            #endfor

            print("I did not find a month for ",value)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        else:
            print("Cannot yet deal with these units in get_month!")
            print(self.ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        print("I did not find a month in self.get_month for ",value)
        
        print(self.ounits,self.oyear,self.omonth,self.oday)

        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    #enddef

    # Figure out how much time is between steps
    def calculate_timestep(self):

        total_years=self.eyear-self.syear+1
        # A couple easy ones:
        if self.ntimepoints == total_years:
            self.timestep="1Y"
        elif self.ntimepoints % total_years == 0:
            self.timestep="1M"
        else:
            print("Do not yet have a solution for this time step situation.")
            print("ntimepoints: ",self.ntimepoints)
            print("total years: ",self.syear,self.eyear,total_years)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        

    #enddef

    # Create a new array that holds the time bounds for this axis
    # This is fast if the year_values are available.
    def calculate_timebounds(self):
        self.timebounds_values=np.zeros((self.ntimepoints,2))*np.nan
        self.timebounds_units=self.timeunits

        if self.timestep == "1M":

            mm=1
            for itime,rtime in enumerate(self.values):
                current_year=self.year_values[itime]
                # The bounds are just Jan 1 to Dec 31 of this year
                timestart=date(current_year, mm, 1)-date(self.oyear, self.omonth, self.oday)
                if mm == 12:
                    timeend=date(current_year+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                else:
                    timeend=date(current_year, mm+1, 1)-date(self.oyear, self.omonth, self.oday)
                #endif
                if self.ounits.lower() == "days":
                    self.timebounds_values[itime,0]=timestart.days
                    self.timebounds_values[itime,1]=timeend.days
                elif self.ounits.lower() == "seconds":
                    self.timebounds_values[itime,0]=timestart.days*24.0*60.0*60.0
                    self.timebounds_values[itime,1]=timeend.days*24.0*60.0*60.0
                else:
                    print("Cannot deal with this unit in calculate_timebounds!")
                    print(ounits)
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
                mm=mm+1
                if mm == 13:
                    mm=1
                #endif

            #endfor
            
        elif self.timestep == "1Y":

            for itime,rtime in enumerate(self.values):
                current_year=self.get_year(rtime)
                # The bounds are just Jan 1 to Jan 1 of the next year
                timestart=date(current_year, 1, 1)-date(self.oyear, self.omonth, self.oday)
                timeend=date(current_year+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                if self.ounits.lower() == "days":
                    self.timebounds_values[itime,0]=timestart.days
                    self.timebounds_values[itime,1]=timeend.days
                elif self.ounits.lower() == "seconds":
                    self.timebounds_values[itime,0]=timestart.days*24.0*60.0*60.0
                    self.timebounds_values[itime,1]=timeend.days*24.0*60.0*60.0
                else:
                    print("Cannot deal with this unit in calculate_timebounds!")
                    print(ounits)
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endfor

        else:

            print("Not sure how to deal with this timestep in calculate_timebounds.")
            print(self.timestep)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

    #enddef

    # During some of the checks above, I may not want to stop.  Some
    # of the functions in this routine are useful even if the time series
    # is not bad.  So I give the option to just trip a flag and keep
    # going, so I can use the other subroutines (like parsing units).
    # Calling this routine checks to see if there was a problem.
    def check_if_bad(self,checkpoint_string):
        if self.bad_values:
            print("***************************************************")
            print("Found some bad time series values at {}.".format(checkpoint_string))
            print("Stopping.")
            print("***************************************************")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    #enddef

    # Convert the units of the timeaxis to be a specific string that
    # we use in the VERIFY data.
    def harmonize_units(self):
        harm_oyear=1900
        harm_omonth=1
        harm_oday=1
        harm_ounits="days"
        harmonized_string="{} since {}-{:02d}-{:02d}".format(harm_ounits,harm_oyear,harm_omonth,harm_oday)

        print("Changing time axis units from: ",self.timeunits)
        print("                           to: ",harmonized_string)

        # This we can change, but only from seconds to days.
        if harm_ounits != self.ounits:
            if harm_ounits == "days" and self.ounits == "seconds":

                self.values[:]=self.values[:]/60.0/60.0/24.0
                self.timeunits = self.timeunits.replace(self.ounits,harm_ounits)
                self.parse_units()


            else:
                print("Units not equal in time axis!")
                print("desired units: {}, current units: {}".format(harm_ounits,self.ounits))
                traceback.print_stack(file=sys.stdout)
                sys.exit(1) 
            #endif
        #endif

        # Check, but don't do anything yet.
        if harm_oyear != self.oyear or harm_omonth != self.omonth or harm_oday != self.oday:
            print("Origin date not equal in time axis!")
            print("desired year: {}, current year: {}".format(harm_oyear,self.oyear))
            print("desired month: {}, current month: {}".format(harm_omonth,self.omonth))
            print("desired day: {}, current day: {}".format(harm_oday,self.oday))
            oldorigindate=date(self.oyear,self.omonth,self.oday)
            neworigindate=date(harm_oyear,harm_omonth,harm_oday)
            timediff=neworigindate-oldorigindate

            self.values[:]=self.values[:]-timediff.days

            # Now that the axis has been shifted, recalculate some internal
            # values to make sure it's consistent
            self.timeunits=harmonized_string
            self.parse_units()
            self.calculate_timebounds()

            # Run some checks to make sure the axis has not changed
            test_year=self.get_year(self.values[0])
            if test_year != self.syear:
                print("Error in calculation of harmnoized time units!")
                print("old syear: {}, new syear: {}".format(self.syear,test_year))
                traceback.print_stack(file=sys.stdout)
                sys.exit(1) 
            #endif
            test_year=self.get_year(self.values[-1])
            if test_year != self.eyear:
                print("Error in calculation of harmnoized time units!")
                print("old eyear: {}, new eyear: {}".format(self.eyear,test_year))
                traceback.print_stack(file=sys.stdout)
                sys.exit(1) 
            #endif

        #endif

        
    #enddef

    # Create an annual axis from this axis, and regrid a data array to
    # match
    def regrid_to_annual_axis(self,data_array,data_operation):

        if self.timestep == "1Y":
            print("Axis is already annual.  No regridding needed.")
            return
        #endif

        # Figure out which axis is the time axis, i.e. which axis
        # has the same length as the current time axis.
        time_dim=-1
        for idim,dim_val in enumerate(data_array.shape):
            if dim_val == len(self.values[:]):
                if time_dim == -1:
                    print("Axis {} is our time axis in regridding.".format(idim))
                    time_dim=idim
                else:
                    print("************\nWe seem to have more than one axis of the regridding")
                    print("data array with the same length as the time axis.  Not sure which to pick!")
                    print("data_array.shape :",data_array.shape)
                    print("len of time axis: ",len(self.values[:]))
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endif
        #endfor
        if time_dim == -1:
            print("************\nCould not find a time axis in regridding.")
            print("data_array.shape :",data_array.shape)
            print("len of time axis: ",len(self.values[:]))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Now, depending on the operation requested and the frequency of
        # the data, reshape and recombine the array.
        if self.timestep == "1M":
            nsteps=12
        elif self.timestep == "1D":
            # This is more complicated...What to do with leap years?
            print("************\nNot yet ready for regridding daily data.")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        else:
            print("************\nDo not recognize the timestep {}.".format(self.timestep))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if data_operation not in ("ave","sum"):
            # for sum, nothing additional is needed.  Make sure I recognize this
            # string, though.
            print("************\nDo not recognize the data operation {}.".format(data_operation))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # This is...not pretty.  I cannot think of a general way to do it,
        # though.
        if len(data_array.shape) == 2 and time_dim == 0:

        # A nice way to take the average across groups of 12.  Reshape the array, adding a new
        # axis, and then take the mean of that axis.

            regridded_data_array=np.reshape(data_array[:,:], (-1, 12, data_array.shape[1]))
            if data_operation == "ave":
                regridded_data_array=np.mean(regridded_data_array,axis=1)
            #endif

        elif len(data_array.shape) == 1 and time_dim == 0:
            regridded_data_array=np.reshape(data_array[:], (-1, 12))
            if data_operation == "ave":
                regridded_data_array=np.mean(regridded_data_array,axis=1)
            #endif

        elif len(data_array.shape) == 3 and time_dim == 0:

        # A nice way to take the average across groups of 12.  Reshape the array, adding a new
        # axis, and then take the mean of that axis.

            regridded_data_array=np.reshape(data_array[:,:,:], (-1, 12, data_array.shape[1], data_array.shape[2]))
            if data_operation == "ave":
                regridded_data_array=np.mean(regridded_data_array,axis=1)
            elif data_operation == "sum":
                regridded_data_array=np.nansum(regridded_data_array,axis=1)
            #endif

        else:
            print("************\nCannot yet deal with this shape of data array in regrid time axis.")
            print("data_array.shape :",data_array.shape)
            print("time dimension: ",time_dim)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Also create an annual axis.
        regridded_axis_values=create_annual_axis(self.syear,self.eyear,self.oyear,self.omonth,self.oday,self.ohour,self.omin,self.osec,self.ounits)

        # This shows that the operation works...although there is a difference in the 6th digit
        #print("kjfe ",data_array.shape,regridded_data_array.shape)
        #print("kjfe 1 ",data_array[0:12,16])
        #print("kjfe 1 ",np.sum(data_array[0:12,16])/12.0)
        #print("kjfe 2 ",regridded_data_array[0,16])

        return regridded_data_array,regridded_axis_values

    #enddef

    # Create an monthly axis from this axis, and regrid a data array to
    # match
    def regrid_to_monthly_axis(self,data_array,data_operation):

        if self.timestep == "1M":
            print("Axis is already monthly.  No regridding needed.")
            return
        #endif

        # Figure out which axis is the time axis, i.e. which axis
        # has the same length as the current time axis.
        time_dim=-1
        for idim,dim_val in enumerate(data_array.shape):
            if dim_val == len(self.values[:]):
                if time_dim == -1:
                    print("Axis {} is our time axis in regridding.".format(idim))
                    time_dim=idim
                else:
                    print("************\nWe seem to have more than one axis of the regridding")
                    print("data array with the same length as the time axis.  Not sure which to pick!")
                    print("data_array.shape :",data_array.shape)
                    print("len of time axis: ",len(self.values[:]))
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endif
        #endfor
        if time_dim == -1:
            print("************\nCould not find a time axis in regridding.")
            print("data_array.shape :",data_array.shape)
            print("len of time axis: ",len(self.values[:]))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Now, depending on the operation requested and the frequency of
        # the data, reshape and recombine the array.
        if self.timestep == "1Y":
            nsteps=12
        elif self.timestep == "1D":
            # This is more complicated...What to do with leap years?
            print("************\nNot yet ready for regridding daily data.")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        else:
            print("************\nDo not recognize the timestep {}.".format(self.timestep))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if data_operation not in ("ave","sum"):
            # Make sure I recognize this string, though.
            print("************\nDo not recognize the data operation {}.".format(data_operation))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # This is...not pretty.  I cannot think of a general way to do it, though.
        if data_operation == "ave":
            scale_factor=1.0
        elif data_operation == "sum":
            scale_factor=12.0
        #endif
        if len(data_array.shape) == 3 and time_dim == 0:

            regridded_data_array=np.zeros((data_array.shape[0]*12,data_array.shape[1],data_array.shape[2]))*np.nan
            
            for iyear in range(data_array.shape[0]):
                smonth=iyear*12
                emonth=iyear*12+12
                regridded_data_array[smonth:emonth,:,:]=data_array[iyear,:,:]/scale_factor
            #endif

        else:
            print("************\nCannot yet deal with this shape of data array in regrid time axis.")
            print("data_array.shape :",data_array.shape)
            print("time dimension: ",time_dime)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        # Also create a monthly axis.
        regridded_axis_values=create_monthly_axis(self.syear,self.eyear,self.oyear,self.omonth,self.oday,self.ohour,self.omin,self.osec,self.ounits)

        # This shows that the operation works
        if False:
            print("kjfe ",data_array.shape,regridded_data_array.shape)
            for ilat in range(data_array.shape[1]):
                for ilon in range(data_array.shape[2]):
                    test_nans=np.isnan(regridded_data_array[:,ilat,ilon])
                    if not test_nans.all():
                        for iyear in range(data_array.shape[0]):
                            smonth=iyear*12
                            emonth=iyear*12+12
                            print("kjfe 1 ",data_array[iyear,ilat,ilon])
                            print("kjfe 2 ",regridded_data_array[smonth:emonth,ilat,ilon])
                            
                        #endif
                    #endif
                #endfor
            #endfor
        #endif

        return regridded_data_array,regridded_axis_values

    #enddef

    #####
    def compare_time_axis(self,timeaxis):
        
        if self.timestep != timeaxis.timestep:
            print("Two time axes do not have the same timestep!")
            print("Self: {}, Other axis: {}".format(self.timestep,timeaxis.timestep))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    #enddef

    #### Return the index for this date, stopping if this date is outside
    # the bounds of this time axis.
    def find_date_index(self,target_date):

        origindate=date(self.oyear,self.omonth,self.oday)
        timediff=target_date-origindate
        
        if self.ounits != "days":
            print("Cannot current find a date where the units are not days.")
            print("Considering using the harmonize_units method of this class.")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        if timediff.days < self.timebounds_values[0,0] or timediff.days > self.timebounds_values[-1,1]:
            print("Requested timevalue is not on this axis!")
            print(self.values[:])
            print("Date requested: ",target_date)
            print("timediff: ",timediff.days)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        found_index=-1
        for itime in range(self.timebounds_values.shape[0]):
            if self.timebounds_values[itime,0] <= timediff.days and self.timebounds_values[itime,1] > timediff.days:
                found_index=itime
            #endif
        #endfor

        if found_index == -1:
            print("Somehow did not find time index!")
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        return found_index

    #enddef
    
    # Creates two arrays, one which holds the month and one
    # which holds the year of every timestep, for quick
    # referencing later.
    def calculate_year_month(self):
        print("Calculating year and month values.")
        self.year_values=np.zeros(len(self.values),dtype=np.int32)
        self.month_values=np.zeros(len(self.values),dtype=np.int32)

        # Do this more simply.  Calculate the first value,
        # and then based on our time axis, we can
        # fill in the rest.  This is much faster.
        if self.timestep in ['1M','1Y']:
            self.year_values[0]=self.get_year(self.values[0])
            self.month_values[0]=self.get_month(self.values[0])

            iyear=self.year_values[0]
            imonth=self.month_values[0]

            if self.timestep == '1Y':
                for itime,rtime in enumerate(self.values[:]):
                    if itime == 0:
                        continue
                    #endif
                    iyear=iyear+1
                    self.year_values[itime]=iyear
                    self.month_values[itime]=imonth
                #endfor
            elif self.timestep == '1M':
                for itime,rtime in enumerate(self.values[:]):
                    if itime == 0:
                        continue
                    #endif
                    imonth=imonth+1
                    if imonth > 12:
                        iyear=iyear+1
                        imonth=1
                    #endif
                    self.year_values[itime]=iyear
                    self.month_values[itime]=imonth
                #endfor
            #endif

            # Check to see if the last value is correct.
            test_year=self.get_year(self.values[-1])
            test_month=self.get_month(self.values[-1])

            if test_year != self.year_values[-1] or test_month != self.month_values[-1]:
                print("Messed up doing the fast calculate_year_month!")
                print(self.timestep)
                print("Years: ",self.year_values)
                print("Months: ",self.month_values)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            else:
                print("Successfully created quick month and years.")
            #endif


        else:
            # Do it the slow way.
            for itime,rtime in enumerate(self.values[:]):
                self.year_values[itime]=self.get_year(rtime)
                self.month_values[itime]=self.get_month(rtime)
            #endfor
        #endif

    #endif
    
#enddef    

#endclass

# Create a new time axis, creating values corresponding to the 15th of every month,
# based on an origin year and units given by the caller.
def create_monthly_axis(syear,eyear,oyear,omonth,oday,ohour,omin,osec,ounits):

    nyears=eyear-syear+1
    monthly_timeaxis_values=np.zeros((nyears*12))*np.nan
    seconds_per_day=24.0*60.0*60.0
    
    yy=syear
    icounter=-1
    for iyear in range(nyears):
        for imonth in range(12):
            icounter=icounter+1
            if ounits.lower() == "days":
                timediff=date(yy, imonth+1, 15)-date(oyear, omonth, oday)
                monthly_timeaxis_values[icounter]=timediff.days
            elif ounits.lower() == "seconds":
                if osec != "":
                    timediff=datetime(yy, imonth+1, 15,0,0,0)-datetime(oyear, omonth, oday, ohour, omin, osec)
                    monthly_timeaxis_values[icounter]=timediff.days*seconds_per_day+timediff.seconds
                else:
                    timediff=date(yy, imonth+1, 15)-date(oyear, omonth, oday)
                    monthly_timeaxis_values[icounter]=timediff.days*seconds_per_day
                #endif
                    
            else:
                print("Cannot deal with this unit in create_monthly_axis!")
                print(ounits)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        #endif
        yy=yy+1
    #endif

    return monthly_timeaxis_values

#enddef

# Create a new time axis, creating values corresponding to June 15 of every year, based
# on an origin year and units given by the caller.
def create_annual_axis(syear,eyear,oyear,omonth,oday,ohour,omin,osec,ounits):

    nyears=eyear-syear+1
    annual_timeaxis_values=np.zeros((nyears))*np.nan
    seconds_per_day=24.0*60.0*60.0
    
    yy=syear
    for iyear in range(nyears):
        if ounits.lower() == "days":
            timediff=date(yy, 6, 15)-date(oyear, omonth, oday)
            annual_timeaxis_values[iyear]=timediff.days
        elif ounits.lower() == "seconds":
            if osec != "":
                timediff=datetime(yy, 6, 15,0,0,0)-datetime(oyear, omonth, oday, ohour, omin, osec)
                annual_timeaxis_values[iyear]=timediff.days*seconds_per_day+timediff.seconds
            else:
                timediff=date(yy, 6, 15)-date(oyear, omonth, oday)
                annual_timeaxis_values[iyear]=timediff.days*seconds_per_day
            #endif

        else:
            print("Cannot deal with this unit in create_annual_axis!")
            print(ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
        yy=yy+1
    #endif

    return annual_timeaxis_values

#enddef





