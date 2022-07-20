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
import re,math
from datetime import date,datetime,timedelta

class time_axis:
    # Note that timeunits here is something like 'days since 1900-01-01'
    def __init__(self, timeaxis_values,timeunits,lstop=True,lprint=True):
        self.timeunits = timeunits
        self.values = timeaxis_values
        self.ntimepoints=len(timeaxis_values)
        self.lstop=lstop
        
        # This prints information during the subroutine.  Sometimes
        # it's better to turn this off
        self.lprint=lprint

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
        if self.lprint:
            print("Starting timebounds")
        #endif
        self.calculate_timebounds()
        if self.lprint:
            print("Ending timebounds")
        #endif
        # Some other things we need for creating a NetCDF file for VERIFY
        # with this time axis.
        self.timecoord_name='time'
        self.timecoord_atts={'long_name':self.timecoord_name,'bounds':'time_bounds','units': self.timeunits,'calendar':'standard','axis':'T'}
        self.timebounds_atts={'units':self.timeunits,'calendar':'standard'}
        self.nb2_coord_name="nv"
        self.nb2_dimension_length=self.timebounds_values.shape[1]
        self.timebounds_name="time_bounds"
        self.timecoord_type='f4'
        self.timebounds_type='f4'
        self.timebounds_dimensions=(self.timecoord_name, self.nb2_coord_name)

    #enddef

    # Do some standard checks to see if this axis makes sense.
    def check_integrity(self):
        
        # First, are the values consistantly increasing?
        if self.lprint:
            print("Checking to see if values on this time axis always increase.")
        #endif
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
        if self.lprint:
            print("Yes, values on this time axis always increase.")
            
            print("Time axis units: ",self.timeunits)
        #endif
        self.parse_units()

        if self.lprint:
            print("Calculating the starting year.")
        #endif
        self.syear=self.get_year(self.values[0])
        if self.lprint:
            print("Starting year is {}".format(self.syear))
            print("Calculating the ending year.")
        #endif
        self.eyear=self.get_year(self.values[-1])
        if self.lprint:
            print("Ending year is {}".format(self.eyear))
        #endif

        # Timestep
        if self.lprint:
            print("Calculating the time between data points (1Y, 1M, 1D).")
        #endif
        self.calculate_timestep()
        if self.lprint:
            print("Timestep is {}.".format(self.timestep))
        #endif

        # Sometimes strange things happen if, for example, we are
        # dealing with monthly values and units of days, and the
        # days indicate Jan 1.
        if self.timestep == "1M":
            nmonths={}
            for rvalue in self.values[:]:
                cyear=self.get_year(rvalue)
                if cyear not in nmonths.keys():
                    nmonths[cyear]=1
                else:
                    nmonths[cyear]=nmonths[cyear]+1
                #endif
            #endfor
            for cyear in nmonths.keys():
                if nmonths[cyear] != 12:
                    print("**************** WARNING")
                    print("The number of months for year {} is not equal to twelve on this timeaxis.".format(cyear))
                    print(cyear,nmonths[cyear])
                    print("**************** WARNING")
                #endif
            #endfor
        #endif

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

        if self.lprint:
            print("Time axis measured in {}.".format(self.ounits))
        #endif

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

        # days since 1970-01-01 00:00:00 UTC
        match = re.search(r"\w+ since (\d\d\d\d)-(\d+)-(\d+) (\d+):(\d+):(\d+) UTC\s*$", self.timeunits)
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

    # Given a year, find out the index.  Only to be used for annual time 
    # axis at the moment.  If lstop=True, the code will crash if an
    # index is not found.  Else, an index value of -1 will be returned.
    def get_year_index(self,value,lstop=True):
        
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
            if lstop:
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
        #endif

        return year_index
    #enddef

    # Figure out what year a particular value is in for this time axis
    def get_year(self,value):

        # We should already have passed through the parse_units routine that figures out the origin.
        if self.ounits.lower() == "days":

            # Get an estimate of the starting year.
            if value > 0.0:

                # Use a number of days in a year that is greater than
                # the real number of days in a year.  In this way, we
                # will always underestimate the number of years passed,
                # which is ideal for the starting point.
                syear=self.oyear+int(value/(368))-1

            else:
                
                # Here we do the opposite.  We want to overesimate the
                # number of years passed to overshoot the real year.
                syear=self.oyear-int(value/(360))-1

            #endif

            # Note that, in the case of day units, we don't care about hours, minutes, seconds
            for iyear in range(syear,22000):
                start_year=date(iyear, 1, 1)-date(self.oyear, self.omonth, self.oday)
                end_year=date(iyear, 12, 31)-date(self.oyear, self.omonth, self.oday)

                # This is odd.  For some time axes, we need this to be
                # <= end_year.days, while for the monthly values, we
                # need < end_year.days.  Not sure how to deal with this.
#                print("iureow ",iyear,value,start_year.days,end_year.days)
                if value >= start_year.days and value <= end_year.days+1:
                    return int(iyear)
                #endif
            #endfor

            print("I did not find a year for ",value)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        elif self.ounits.lower() == "seconds":

            # Get an estimate of the starting year.
            if value > 0.0:

                # Use a number of seconds in a year that is greater than
                # the real number of seconds in a year.  In this way, we
                # will always underestimate the number of years passed,
                # which is ideal for the starting point.
                syear=self.oyear+int(value/(368*24*60*60))-1

            else:
                
                # Here we do the opposite.  We want to overesimate the
                # number of years passed to overshoot the real year.
                syear=self.oyear-int(value/(360*24*60*60))-1

            #endif

            # Note that, in the case of second units, we may have hours, minutes, seconds
            for iyear in range(syear,22000):
                seconds_per_day=24.0*60.0*60.0
                if self.osec != "":
                    print("ORIGIN ",iyear,self.oyear, self.omonth, self.oday)
                    start_year=datetime(iyear, 1, 1,0,0,0)-datetime(self.oyear, self.omonth, self.oday, self.ohour, self.omin, self.osec)
                    end_year=datetime(iyear+1, 1, 1,0,0,0)-datetime(self.oyear, self.omonth, self.oday, self.ohour, self.omin, self.osec)
                    print("jifoew2 ||  {} || {}".format(start_year,end_year))
                else:
                    print("jifoew ",iyear,self.oyear, self.omonth, self.oday)
                    start_year=date(iyear, 1, 1)-date(self.oyear, self.omonth, self.oday)
                    end_year=date(iyear+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                    print("jifoew2 ",start_year,end_year)
                #endif

                # I cannot use the seconds attribute of datetime here, since that just gives the
                # amount of seconds in the day.
                print("ijfeow ",value,start_year.days*seconds_per_day)
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

            # Get an estimate of the starting year.
            if value > 0.0:

                # Use a number of days in a year that is greater than
                # the real number of days in a year.  In this way, we
                # will always underestimate the number of years passed,
                # which is ideal for the starting point.
                syear=self.oyear+int(value/(368))-1

            else:
                
                # Here we do the opposite.  We want to overesimate the
                # number of years passed to overshoot the real year.
                syear=self.oyear-int(value/(360))-1

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
                    #print("getting month ",value,imonth,start_date.days,end_date.days)
                    if value >= start_date.days and value < end_date.days:
                        return int(imonth)
                    #endif
                #endfor
            #endfor

            print("I did not find a month for ",value)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        elif self.ounits.lower() == "seconds":

            # Get an estimate of the starting year.
            if value > 0.0:

                # Use a number of seconds in a year that is greater than
                # the real number of seconds in a year.  In this way, we
                # will always underestimate the number of years passed,
                # which is ideal for the starting point.
                syear=self.oyear+int(value/(368*24*60*60))-1

            else:
                
                # Here we do the opposite.  We want to overesimate the
                # number of years passed to overshoot the real year.
                syear=self.oyear-int(value/(360*24*60*60))-1

            #endif

            for iyear in range(syear,22000):
                for imonth in range(1,13):
                    if imonth != 12:
                        start_date=datetime(iyear, imonth, 1)-datetime(self.oyear, self.omonth, self.oday)
                        end_date=datetime(iyear, imonth+1, 1)-datetime(self.oyear, self.omonth, self.oday)
                    else:
                        start_date=datetime(iyear, imonth, 1)-datetime(self.oyear, self.omonth, self.oday)
                        end_date=datetime(iyear+1, 1, 1)-datetime(self.oyear, self.omonth, self.oday)
                    #endif
#                    print(dir(start_date))

                    #print(value,start_date.total_seconds(),end_date.total_seconds())
                    if value >= start_date.total_seconds() and value <= end_date.total_seconds():
                    #    print("Returnning month ",imonth)
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

        seconds_per_day=24.0*60.0*60.0

        total_years=self.eyear-self.syear+1
        # A couple easy ones:
        if self.lprint:
            print("Calculating timestep. ",self.ntimepoints,total_years)
        #endif
        if self.ntimepoints == total_years:
            self.timestep="1Y"
            return
        # This does not always work!
        #elif self.ntimepoints % total_years == 0:
        #    self.timestep="1M"
        else:

            time_diff=self.values[1]-self.values[0]
            
            if self.ounits.lower() == "days":
                if time_diff < 1.0:
                    print("Our time difference is less than our unit!")
                    print(time_diff)
                    print(self.ounits)
                    # This is a known case of ORCHIDEE forcing data
                    if time_diff == 0.25:
                        self.timestep="6H"
                        return
                    #endif
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)

                elif time_diff > 25:
                    # Assuming this is monthly.
                    self.timestep="1M"
                    return
                else:
                    ndays=int(time_diff)
                    self.timestep="{:d}D".format(ndays)
                #endif

            elif self.ounits.lower() == "seconds":

                time_diff=self.values[1]-self.values[0]

                # give a range of values here.
                if time_diff > 0.9*seconds_per_day and time_diff < 1.1*seconds_per_day:
                    self.timestep="1D"
                    return
                elif time_diff > 24.0*seconds_per_day and time_diff < 32.0*seconds_per_day:
                    # Assuming this is monthly.
                    self.timestep="1M"
                    return
                
                elif time_diff > 360.0*seconds_per_day and time_diff < 370.0*seconds_per_day:
                    # Assuming this is yearly.
                    self.timestep="1Y"
                    return
                elif time_diff == 21600:
                    # some ORCHIDEE forcing files
                    self.timestep="6H"
                    return

                else:
                    print("Don't know what this timestep is (seconds).")
                    print(self.ounits)
                    print(self.values[1],self.values[0],time_diff)
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif

            else:
                print("Cannot deal with this unit in calculate_timestep!")
                print(self.ounits)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
            if self.lprint:
                print("###############################################")
#                print("Not an even timestep (not 1Y, 1M, 1D).")
                print("ntimepoints: ",self.ntimepoints)
                print("total years: ",self.syear,self.eyear,total_years)
                print("Setting timestep: ",self.timestep)
                print("###############################################")
            #endif
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

        elif self.timestep == "6H":

            # The bounds will be +/- 3 hours from the current value.  This
            # needs to be seconds, minutes, hours, or days on the units.
            if self.ounits.lower() not in ["hours","seconds","minutes","days"]:
                print("Not sure how to deal with a 6H timestep and these units!")
                print("Units ",self.ounits)
                print(self.timestep)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif
            for itime,rtime in enumerate(self.values):
                if self.ounits.lower() == "days":
                    self.timebounds_values[itime,0]=rtime-1.0/12.0
                    self.timebounds_values[itime,1]=rtime+1.0/12.0
                elif self.ounits.lower() == "seconds":
                    self.timebounds_values[itime,0]=rtime-3.0*60.0*60.0
                    self.timebounds_values[itime,1]=rtime+3.0*60.0*60.0
                else:
                    print("Cannot deal with this unit in calculate_timebounds!")
                    print(ounits)
                    traceback.print_stack(file=sys.stdout)
                    sys.exit(1)
                #endif
            #endfor

        else:

            # Set the timebounds equal to half the distance between points.
            print("Not an even timestep.  Just setting the timebounds to half")
            print("the distance between points for the nearest day.")
            print(self.timestep)
            # Need the value of the day.
            self.get_days()

            match = re.search(r"(\d+)D", self.timestep)
            if match:
           
                # The number of days in a timestep
                ndays=int(match[1])
            else:
                print("Why can I not find the number of days?")
                print(self.timestep)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            if self.ounits.lower() == "days":

                day_difference=math.floor(ndays/2.0)

            else:
                print("Cannot deal with this unit in calculate_timebounds!")
                print(ounits)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            for itime,rtime in enumerate(self.values):

                current_year=self.get_year(rtime)
                current_month=self.get_month(rtime)
                current_day=self.day_values[itime]

                #print("Current values: ",current_year,current_month,current_day)
                

                # All I really need is to figure out the value of the
                # lower bound for the first timestep.
                timecurrent=date(current_year, current_month, current_day)-date(self.oyear, self.omonth, self.oday)
                if itime == 0:
                    self.timebounds_values[itime,0]=timecurrent.days-day_difference
                    
                else:
                    self.timebounds_values[itime,0]=self.timebounds_values[itime-1,1]
                    #endif
                self.timebounds_values[itime,1]=timecurrent.days+day_difference
                    
                #print("itime ",self.timebounds_values[itime,0],self.values[itime],self.timebounds_values[itime,1])


            #endfor
#            traceback.print_stack(file=sys.stdout)
#            sys.exit(1)
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
                print("Error in calculation of harmonized time units!")
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
                regridded_data_array=np.nanmean(regridded_data_array,axis=1)
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

        if len(self.values[:]) != len(timeaxis.values[:]):
            print("Two time axes are not of the same length!")
            print("Self: {}, Other: {}".format(len(self.values[:]),len(timeaxis.values[:])))
            print("Self: ",self.values[:])
            print("Other: ",timeaxis.values[:])
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    #enddef

    #####
    # This creates a list that we can pass to matplotlib for plotting
    def calculate_plotting_axis(self):
        
        dates=[]

        for itimestep in range(len(self.values[:])):
            dates.append(datetime(self.year_values[itimestep],self.month_values[itimestep],15))
        #endfor

        return dates

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
    # Also creates a reverse lookup: an array like index[year][month]
    # which, for a given year and month, returns the position on 
    # the time axis so we can quickly reference it.  This is only use
    # for monthly time axis.
    def calculate_year_month(self):
        if self.lprint:
            print("Calculating year and month values.")
        #endif
        self.year_values=np.zeros(len(self.values),dtype=np.int32)
        self.month_values=np.zeros(len(self.values),dtype=np.int32)

        # Do this more simply.  Calculate the first value,
        # and then based on our time axis, we can
        # fill in the rest.  This is much faster.
        ### I was able to update the get_year and get_month
        # routines to be faster, but giving a better first
        # guess of the year/month.  
#        if self.timestep in ['1M','1Y']:
        if False:
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
#                print("jfiew ",itime,self.year_values[itime],self.month_values[itime])
            #endfor

            # If we have a monthly time axis, add an index.
            class month_index():
                def __init__(self):
                    self.month_index={}
                    for imonth in range(1,13):
                        self.month_index[imonth]=None
                    #endfor
                #enddef
            #endclass

            if self.timestep == '1M':
                self.year_index={}
                for year in set(self.year_values):
                    self.year_index[year]=month_index()
                #endfor
                for itime,rtime in enumerate(self.values[:]):
                    self.year_index[self.year_values[itime]].month_index[self.month_values[itime]]=itime
                #endfor
            #endif

        #endif

    #enddef

    # This takes more time, so don't always do it. 
    def get_days(self):

        try:
            self.year_values[0]
            self.month_values[0]
        except:
            self.calculate_year_month()
        #endtry

        print("Now calculating the day of each timestep.")
        self.day_values=np.zeros(len(self.values),dtype=np.int32)
        for itime,rtime in enumerate(self.values[:]):
            self.day_values[itime]=self.get_day(rtime,itime)
            print("fioew ",itime,self.day_values[itime])
        #endfor

        print("Finished calculating the days.")

    #enddef

    def get_day(self,rtime,itime):

        # If we have a yearly or monthly time axis, just take the 15th
        if self.timestep == "1Y":
            return 15
        #endif

        # If it's equal to days, also not too difficult.
        iyear=self.year_values[itime]
        imonth=self.month_values[itime]
        #print(self.month_values)
        #print(self.year_values)

        # We should already have passed through the parse_units routine that figures out the origin.
        if self.ounits.lower() == "days":

            for iday in range(1,31):
                
                start_date=date(iyear, imonth, iday)-date(self.oyear, self.omonth, self.oday)
                # The ending date might be in the next month or next year
                try:
                    end_date=date(iyear, imonth, iday+1)-date(self.oyear, self.omonth, self.oday)
                except:
                    try:
                        end_date=date(iyear, imonth+1, 1)-date(self.oyear, self.omonth, self.oday)
                    except:
                        try:
                            end_date=date(iyear+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                        except:
                            print("No idea how to deal with this!")
                            print("iyear,imonth,iday: ",iyear,imonth,iday)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        #endtry
                    #endtry
                #endtry

#                print("rtime ",rtime,start_date.days,end_date.days)
#                print("rtime ",iday,iyear,imonth)
                if rtime >= start_date.days and rtime <= end_date.days:
                    return int(iday)
                #endif
            #endfor

            print("I did not find a day for ",rtime)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

        # How does this look for seconds?  Similar, I should think.
        elif self.ounits.lower() == "seconds":
            for iday in range(1,32):
                
                start_date=date(iyear, imonth, iday)-date(self.oyear, self.omonth, self.oday)
                # The ending date might be in the next month or next year
                try:
                    end_date=date(iyear, imonth, iday+1)-date(self.oyear, self.omonth, self.oday)
                except:
                    try:
                        end_date=date(iyear, imonth+1, 1)-date(self.oyear, self.omonth, self.oday)
                    except:
                        try:
                            end_date=date(iyear+1, 1, 1)-date(self.oyear, self.omonth, self.oday)
                        except:
                            print("No idea how to deal with this!")
                            print("iyear,imonth,iday: ",iyear,imonth,iday)
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)
                        #endtry
                    #endtry
                #endtry
                sec_per_day=60.0*60.0*24.0
                start_seconds=start_date.days*sec_per_day
                end_seconds=end_date.days*sec_per_day
#                print("rtime ",rtime,start_seconds,end_seconds)
#                print("rtime ",iday,iyear,imonth)
                if rtime >= start_seconds and rtime <= end_seconds:
                    return int(iday)
                #endif
            #endfor

            print("I did not find a day for ",rtime)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        else:
            print("Cannot yet deal with these units in get_day!")
            print(self.ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        print("I did not find a day in self.get_day for ",rtime)
        
        print(self.ounits,self.oyear,self.omonth,self.oday)

        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

#endclass

# Generate monthly time axis values corresponding to the 15th of every month,
# based on an origin year and units given by the caller.
def create_monthly_axis(syear,eyear,oyear,omonth,oday,ohour,omin,osec,ounits,smonth=1,emonth=12):

    # I don't assume that we have complete years.  But that means
    # I need to figure out how many total months we have.
    yy=syear
    mm=smonth
    nmonths=0
    while yy <= eyear:
        if yy == eyear and mm > emonth:
            break
        #endif
        nmonths=nmonths+1
        mm=mm+1
        if mm > 12:
            mm=1
            yy=yy+1
        #endif
    #endwhile

    monthly_timeaxis_values=np.zeros((nmonths))*np.nan
    seconds_per_day=24.0*60.0*60.0
    
    # Now find the actual data
    yy=syear
    icounter=-1
    yy=syear
    mm=smonth
    icounter=-1
    while yy <= eyear:
        if yy == eyear and mm > emonth:
            break
        #endif
        icounter=icounter+1

        if ounits.lower() == "days":
            timediff=date(yy, mm, 15)-date(oyear, omonth, oday)
            monthly_timeaxis_values[icounter]=timediff.days
        elif ounits.lower() == "seconds":
            if osec != "":
                timediff=datetime(yy, mm, 15,0,0,0)-datetime(oyear, omonth, oday, ohour, omin, osec)
                monthly_timeaxis_values[icounter]=timediff.days*seconds_per_day+timediff.seconds
            else:
                timediff=date(yy, mm, 15)-date(oyear, omonth, oday)
                monthly_timeaxis_values[icounter]=timediff.days*seconds_per_day
            #endif
            
        else:
            print("Cannot deal with this unit in create_monthly_axis!")
            print(ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        mm=mm+1
        if mm > 12:
            mm=1
            yy=yy+1
        #endif
    #endwhile

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

# Create a new time axis at a daily resolution, based
# on an origin year and units given by the caller.  This also depends on
# a calendar time.
# start_date and end_date are datetime objects.
# Right now, I ignore the calendar.  :(
def create_daily_axis(start_date,end_date,oyear,omonth,oday,ohour,omin,osec,ounits,calendar_type='Gregorian'):

    timediff=end_date-start_date
    ndays=timediff.days+1
    print("We have {} days.".format(ndays))
    daily_timeaxis_values=np.zeros((ndays))*np.nan

    seconds_per_day=24.0*60.0*60.0
    current_date=start_date

    for iday in range(ndays):
        if ounits.lower() == "days":
            timediff=current_date-date(oyear, omonth, oday)
            daily_timeaxis_values[iday]=timediff.days
        elif ounits.lower() == "seconds":
            if osec != "":
                timediff=current_date-datetime(oyear, omonth, oday, ohour, omin, osec)
                daily_timeaxis_values[iday]=timediff.days*seconds_per_day+timediff.seconds
            else:
                timediff=current_date-date(oyear, omonth, oday)
                daily_timeaxis_values[iday]=timediff.days*seconds_per_day
            #endif

        else:
            print("Cannot deal with this unit in create_daily_axis!")
            print(ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        current_date=current_date+timedelta(days=1)

    #endfor

    return daily_timeaxis_values

#enddef

# Create a new time axis at subdaily resolution, based
# on an origin year and units given by the caller.  This also depends on
# a calendar time.
# start_date and end_date are datetime objects.
# Right now, I ignore the calendar.  :(
def create_subdaily_axis(start_datetime,end_datetime,oyear,omonth,oday,ohour,omin,osec,ounits,timestep,calendar_type='Gregorian'):

    seconds_per_day=24.0*60.0*60.0
    timediff=end_datetime-start_datetime
    ntotal_seconds=timediff.days*seconds_per_day+timediff.seconds
    # Figure out our timestep, in seconds.
    match = re.search(r"(\d+)(\w+)$", timestep)
    if match:
        number_of_units=int(match[1])
        if match[2].lower() == "h":
            ntimestep_seconds=number_of_units*60.0*60.0
        elif match[2].lower() == "s":
            ntimestep_seconds=number_of_units
        else:
            print("********************************************")
            print("Could not parse this timestep string in create_subdaily_axis.")
            print(timestep)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

    else:
        print("********************************************")
        print("Could not parse this timestep string in create_subdaily_axis.")
        print(timestep)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif
    
    print("nfjiweo ",ntotal_seconds,ntimestep_seconds)
    ntimesteps=int(ntotal_seconds/ntimestep_seconds)

    print("We have {} timesteps.".format(ntimesteps))
    subdaily_timeaxis_values=np.zeros((ntimesteps))*np.nan


    current_date=start_datetime

    for itimestep in range(ntimesteps):
        if ounits.lower() == "seconds":
            timediff=current_date-datetime(oyear, omonth, oday,ohour,omin,osec)
            subdaily_timeaxis_values[itimestep]=timediff.days*seconds_per_day+timediff.seconds
            # These are not going to be integer values
        elif ounits.lower() == "days":
            timediff=current_date-datetime(oyear, omonth, oday,ohour,omin,osec)
            subdaily_timeaxis_values[itimestep]=timediff.days+timediff.seconds/seconds_per_day
#            print("jiefow ",itimestep,subdaily_timeaxis_values[itimestep])
        else:
            print("Cannot deal with this unit in create_subdaily_axis!")
            print(ounits)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        current_date=current_date+timedelta(seconds=ntimestep_seconds)

    #endfor

    return subdaily_timeaxis_values

#enddef

# The above was somewhat complicated to do for simple cases, and didn't
# quite work out.  For example, when I needed 1460 values for noleap years.
# So I made this routine.  However, it's not perfect.  It uses the value
# for Jan 1 based on the Gregorian calendar, but calculates the timestep
# values sometimes using noleap.  So use with caution.
def create_subdaily_annual_axis(start_year,end_year,oyear,omonth,oday,ohour,omin,osec,ounits,timestep,calendar_type='Gregorian'):

    seconds_per_day=24.0*60.0*60.0
    nyears=start_year-end_year+1

    # Do this the stupid way.
    if calendar_type == "noleap":
        ntotal_seconds=nyears*seconds_per_day*365.0
        if timestep == "6H":
            nsteps_per_day=4
            ntimestep_seconds=6.0*60.0*60.0
            current_date=datetime(start_year,1,1,3,0,0)

        else:
            print("********************************************")
            print("Could not parse this timestep string in create_subdaily_annual_axis.")
            print(timestep)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif

        ntimesteps=int(ntotal_seconds/ntimestep_seconds)

        print("We have {} timesteps.".format(ntimesteps))
        subdaily_timeaxis_values=np.zeros((ntimesteps))*np.nan


        for itimestep in range(ntimesteps):
            if ounits.lower() == "seconds":
                timediff=current_date-datetime(oyear, omonth, oday,ohour,omin,osec)
                subdaily_timeaxis_values[itimestep]=timediff.days*seconds_per_day+timediff.seconds
                # These are not going to be integer values
            elif ounits.lower() == "days":
                timediff=current_date-datetime(oyear, omonth, oday,ohour,omin,osec)
                subdaily_timeaxis_values[itimestep]=timediff.days+timediff.seconds/seconds_per_day
                #            print("jiefow ",itimestep,subdaily_timeaxis_values[itimestep])
            else:
                print("Cannot deal with this unit in create_subdaily_axis!")
                print(ounits)
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)
            #endif

            current_date=current_date+timedelta(seconds=ntimestep_seconds)

        #endfor

    else:
        print("********************************************")
        print("Could notdeal with this calendar type in create_subdaily_annual_axis.")
        print(calendar_type)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif

    return subdaily_timeaxis_values

#enddef





