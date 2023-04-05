#########################3
# Open up a NetCDF file with annual values of ORCHIDEE
# output variables, and try some analysis, looking for pixels
# which match some criteria.
######
# Import from standard libraries
from netCDF4 import Dataset as NetCDFFile
import sys,traceback
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# Import from local routines
from netcdf_subroutines import create_time_axis_from_netcdf
from classification_subroutines import create_latlon_string
###############################################

lplot=True
lzero=True
lincreasing=True

ltest_pixel=True
test_lat=35.0
test_lon=63.0

# File to open
# FG2
#input_file="/home/orchidee03/mmcgrath/RUNDIR/STOPLIGHT/extracted_variables_-90Nx90Nx-180Ex180E_FG2.r6792.OD_1901.2010.nc"
# FG1
input_file="/home/orchidee03/mmcgrath/RUNDIR/STOPLIGHT/extracted_variables_-90Nx90Nx-180Ex180E_FG1.r6792.OD_1901.2240.nc"
timeaxis=create_time_axis_from_netcdf(input_file,lprint=True)
allyears=[]
for itime,rtime in enumerate(timeaxis.values):
    allyears.append(timeaxis.year_values[itime])
#endif

ipft=7
nwindow=15 # number of pixels to each side that we consider
scale_factor=1.2

variable_name="TOTAL_M_c"
latcoord="lat"
loncoord="lon"

srcnc = NetCDFFile(input_file,"r")

nlats=len(srcnc[latcoord][:])
nlons=len(srcnc[loncoord][:])
latitudes=srcnc[latcoord][:].copy()
longitudes=srcnc[loncoord][:].copy()

# It takes a long time to check if single pixel timeseries are NaN.
# So try to vectorize it
test_map=srcnc[variable_name][:,ipft,:,:].filled(np.nan)
test_map=np.nanmean(test_map,axis=0)
ltest_map=~np.isnan(test_map)
ltest_map=np.where(test_map ==0.0,False,ltest_map)
units=srcnc[variable_name].units

# Do a simple first test.  Just find the mean value on every pixel
# and plot that distribution.
mean_vals=[]
step_vals=[]
ipixel=0
iplot=0
for ilat in range(nlats):
    for ilon in range(nlons):
        if ltest_pixel:
            if latitudes[ilat] != test_lat or longitudes[ilon] != test_lon:
                continue
            #endif
        #endif
        if ltest_map[ilat,ilon]:
            ipixel=ipixel+1
            if ipixel % 10 == 0:
                print("On pixel: ",ipixel)
            #endif
#            print("Found something: ",ilat,ilon,ipft)
#            print(srcnc[variable_name][:,ipft,ilat,ilon].filled(np.nan))
#            print(np.nanmean(srcnc[variable_name][:,ipft,ilat,ilon].filled(np.nan)))
#            traceback.print_stack(file=sys.stdout)
#            sys.exit(1)
            mean_vals.append(np.nanmean(srcnc[variable_name][:,ipft,ilat,ilon].filled(np.nan)))

            # Do an analysis.
            lfound=False
            timeseries=srcnc[variable_name][:,ipft,ilat,ilon].filled(np.nan)
            ind_timeseries=srcnc["IND"][:,ipft,ilat,ilon].filled(np.nan)

            # Notice this is per day.  Change to per year.
            ind_rec_timeseries=srcnc["RECRUITS_IND"][:,ipft,ilat,ilon].filled(np.nan)*365.0

            for itime in range(nwindow+1,len(timeseries)-nwindow-1):
                if lfound:
                    break
                #endif
                prevmin=np.nanmin(timeseries[itime-nwindow:itime])
                prevmax=np.nanmax(timeseries[itime-nwindow:itime])
                nextmin=np.nanmin(timeseries[itime:itime+nwindow])
                nextmax=np.nanmax(timeseries[itime:itime+nwindow])
                valuestep=abs(timeseries[itime]-timeseries[itime-1])
                # The bars cannot cross.
                if lincreasing:
                    if prevmin >= nextmax:
                        continue
                    #endif
                else:
                    if prevmin <= nextmax:
                        continue
                    #endif
                #endif
                if valuestep > scale_factor*(prevmax-prevmin):
                    if valuestep > scale_factor*(nextmax-nextmin):
#                        print("FOUND")
#                        print(itime,timeseries[itime-1],timeseries[itime],prevmin,prevmax,nextmax,nextmin)
                        lfound=False # don't go to the next timeseries
                        step_vals.append(np.nanmean(srcnc[variable_name][:,ipft,ilat,ilon].filled(np.nan)))

                        if lplot:
                            iplot=iplot+1
                            print("Creating plot: ",iplot)

                            latlon_string=create_latlon_string(latitudes[ilat],longitudes[ilon])

                            fig=plt.figure(3,figsize=(13, 13))
                            gs = gridspec.GridSpec(3, 1, height_ratios=[3,2,2])
                            ax1=plt.subplot(gs[0])
                            lowyear=max(0,itime-20)
                            highyear=min(len(timeseries),itime+21)
                            p1=ax1.plot(allyears[lowyear:highyear],timeseries[lowyear:highyear], linestyle='-', marker='o',color='black')
                            p1=mpl.patches.Rectangle((allyears[itime-nwindow-1],prevmin),nwindow,prevmax-prevmin, color="red", zorder=-100,alpha=0.3)
                            ax1.add_patch(p1)
                            p1=mpl.patches.Rectangle((allyears[itime],nextmin),nwindow,nextmax-nextmin, color="red", zorder=-100,alpha=0.3)
                            ax1.add_patch(p1)

                            ax1.set_title('PFT {}\n{}'.format(ipft,latlon_string),fontdict={'fontsize' : 25,'fontweight' : 'demibold'})

                            ax1.tick_params(labelsize=14)

                            # If I want the y-axis to include zero.
                            if lzero:
                                ymin,ymax=ax1.get_ylim()
                                ax1.set_ylim(0.0,ymax)
                            #endif

                            plt.ylabel(r"""{}""".format(variable_name), fontsize=18)

                            ax2=plt.subplot(gs[1])
                            p1=ax2.plot(allyears[lowyear:highyear],ind_timeseries[lowyear:highyear], linestyle='-', marker='o',color='black')

                            plt.ylabel(r"""IND""", fontsize=18)

                            # Add RECRUITS_IND on an axis to the right
                            ax2_sec=ax2.twinx()
                            color='blue'
                            p1=ax2_sec.plot(allyears[lowyear:highyear],ind_rec_timeseries[lowyear:highyear], linestyle='-', marker='o',color=color)
                            ax2_sec.set_ylabel('RECRUITS_IND', color = color, rotation=270.0,fontsize=18)
                            ax2_sec.yaxis.set_label_coords(1.09,0.5)
                            ax2_sec.tick_params(axis ='y', labelcolor = color)
                            
                            # set the y-ranges the same
                            ymin,ymax=ax2.get_ylim()
                            ax2_sec.set_ylim(0.0,ymax-ymin)
                            ax2.tick_params(labelsize=14)

                            # Now the third plot, with the RDI.
                            ax3=plt.subplot(gs[2])
                            timeseries_1=srcnc["RDI"][:,ipft,ilat,ilon].filled(np.nan)
                            p1=ax3.plot(allyears[lowyear:highyear],timeseries_1[lowyear:highyear], linestyle='--', marker=None,color='black')
                            timeseries_1=srcnc["RDI_TARGET_LOWER"][:,ipft,ilat,ilon].filled(np.nan)
                            p1=ax3.plot(allyears[lowyear:highyear],timeseries_1[lowyear:highyear], linestyle='-', marker=None,color='black')
                            timeseries_1=srcnc["RDI_TARGET_UPPER"][:,ipft,ilat,ilon].filled(np.nan)
                            p1=ax3.plot(allyears[lowyear:highyear],timeseries_1[lowyear:highyear], linestyle='-', marker=None,color='black')
                            plt.ylabel(r"""RDI""", fontsize=18)

                            ax3.tick_params(labelsize=14)

                            # Always at the end
                            plt.xlabel(r"""Time [yr]""", fontsize=18)

                            if lincreasing:
                                plt.savefig("test_timeseries_{}_increasing.png".format(iplot))
                            else:
                                plt.savefig("test_timeseries_{}_decreasing.png".format(iplot))
                            #endif
                            plt.close()
                            
                            #print("EXITING EARLY")
                            #traceback.print_stack(file=sys.stdout)
                            #sys.exit(1)

                        #endif
                    #endif
                #endif
            #endfor
        #endif
    #endfor
    if len(mean_vals) > 100:
        break
    #endif
#endfor

# Now print out a few summary statistics
nbins=30
all_hist,bin_edges=np.histogram(mean_vals,bins=nbins)
prob_hist,bin_edges_2=np.histogram(step_vals,bins=bin_edges)

xvalues=[x+(y-x)/2.0 for x,y in zip(bin_edges[0:-1],bin_edges[1:])]

width=(xvalues[1]-xvalues[0])/2.0

ax1=plt.axes()
plt.bar(xvalues,all_hist,width=width,color="blue",label="All pixels")
plt.bar(xvalues,prob_hist,width=width,color="red",label="Step detected")

ax1.set_xlabel('Total carbon biomass [{}]'.format(units))
ax1.set_ylabel('# of pixels')
ax1.set_title('PFT {}'.format(ipft))
ax1.legend()
plt.savefig("test_{}_hist_pft{}.png".format(variable_name,ipft))
plt.close()


