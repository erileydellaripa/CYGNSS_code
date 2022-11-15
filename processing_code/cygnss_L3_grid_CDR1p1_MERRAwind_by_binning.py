import numpy as np
from netCDF4 import Dataset
import glob
import os
from scipy import stats
#import time

#start = time.time()

def file_open_and_clean(l2_file):
    Data      = Dataset(l2_file)
    var       = np.array(Data.variables['merra2_wind_speed'][:])
    samp_time = np.array(Data.variables['sample_time'][:])
    lats      = np.array(Data.variables['lat'][:])
    lons      = np.array(Data.variables['lon'][:])
    Data.close()
    
    #Eliminate indices with -9999s in var 
    wbad = np.where(var < -999.)
    
    var       = np.delete(var, wbad)
    samp_time = np.delete(samp_time, wbad)
    lats      = np.delete(lats, wbad)
    lons      = np.delete(lons, wbad)
    
    return var, lats, lons, samp_time

def bin_bounds(min_val, max_val, bin_spacing):
    nbins = ((max_val - min_val)/bin_spacing) + 1
    bins  = np.linspace(min_val, max_val, num = int(nbins))
    
    return bins

def file_save(lat_axis, lon_axis, binned_vals, file_date):
    #takes in the gridded L3 data, and the Lat/Lon data
    #saves it into a netcdf for that day
    #This directory is inside the directory where all the raw files are
    #This is the only way I could get it to work on the compute node on Asha.
    os.chdir('cyg_MERRA2wind_1deg_CDR1p1') 
    #need to make netCDF file
    nc_file = Dataset('L3_CDR1p1_MERRA2wind_' + file_date + '_HOURLY.nc', 'w', format='NETCDF4')
    nc_file.close() #I close it here so if there was an issue generating I know when

    #make a group for the variables to exist in
    nc_file = Dataset('L3_CDR1p1_MERRA2wind_' + file_date + '_HOURLY.nc', 'a')
    nc_file.createGroup('data')
    #make dimensions for all the variables
    time = nc_file.createDimension('time',24)
    lat = nc_file.createDimension('lat',len(lat_axis))
    lon = nc_file.createDimension('lon', len(lon_axis))
    #add in the variables
    wind = nc_file.createVariable('wind','f8',('time','lat','lon',), fill_value = -9999, zlib=True,least_significant_digit = 4)
    lats = nc_file.createVariable('latitude','f4',('lat',), fill_value = -9999,zlib=True,least_significant_digit = 1)
    lons = nc_file.createVariable('longitude','f4',('lon',), fill_value = -9999,zlib=True,least_significant_digit = 1)
    #write to the variables
    wind[:]    = binned_vals
    lats[:] = lat_axis
    lons[:] = lon_axis
    #close/save the file
    nc_file.close()

    return 1

def grid_data(vals, lats, lons, time, spacing):
    lat_bins = bin_bounds(-40, 40, spacing) 
    lon_bins = bin_bounds(0, 360, spacing)
    
    lat_axis = lat_bins[0:len(lat_bins)-1]+0.5
    lon_axis = lon_bins[0:len(lon_bins)-1]+0.5
    
    frac_hours = (time/3600.)
    whole_hours = frac_hours.astype(int)
    uniq_hrs, hr_ind = np.unique(whole_hours, return_index = True)
    
    #Add the last indices of the var array, so that I can subset for all the hours appropriately
    hr_ind = np.append(hr_ind, len(var))
    
    #maybe put an if in here to check that min and max for the range are the same
    
    #Create array of -9999s to be filled (see test code on test_2Dbinning.ipynb)
    binned_vals = np.zeros([24, len(lat_bins)-1,len(lon_bins)-1]) - 9999.
    
    #Subset data by hours
    #will need a for loop over each hour, but for now, just test one hour
    
    for ihr in range(len(uniq_hrs)):
        ihr_lats = lats[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_lons = lons[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_vals = vals[hr_ind[ihr]:hr_ind[ihr+1]]
               
        #########################################################################
        #Bin MERRA2 wind according to the lat_bins and lon_bins
        do_bin = stats.binned_statistic_2d(ihr_lats, ihr_lons, ihr_vals, 'mean',bins=[lat_bins, lon_bins])

        binned_vals[ihr, :, :] = do_bin.statistic
    
    return binned_vals, lat_axis, lon_axis

root = os.getcwd()
os.chdir(root + '/cyg_wind_Lev2_CDR1p1/')
fnames  = sorted(glob.glob('*nc'))

for ifile in fnames:
        split_str = ifile.split('.')
        year_str = split_str[2][1:5]
        mon_str = split_str[2][5:7]
        day_str = split_str[2][7:9]
        date_str = year_str + '_' + mon_str + '_' + day_str
        print(date_str)

        var, lats, lons, samp_time = file_open_and_clean(ifile)
        binned_vals, lat_axis, lon_axis = grid_data(var, lats, lons, samp_time, 1)
        file_save(lat_axis, lon_axis, binned_vals, date_str)
        os.chdir(root)
        os.chdir(root + '/cyg_wind_Lev2_CDR1p1/')

#end = time.time()
#print(f"Runtime is {end - start}")
