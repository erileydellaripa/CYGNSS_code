import numpy as np
from netCDF4 import Dataset
import glob
import os
from scipy import stats
import time

def file_open_and_clean_SDR3p1(l2_SDR_file):
    Data      = Dataset(l2_SDR_file)
    var       = np.array(Data.variables['wind_speed'][:])
    var_unc   = np.array(Data.variables['wind_speed_uncertainty'][:])
    samp_time = np.array(Data.variables['sample_time'][:])
    lats      = np.array(Data.variables['lat'][:])
    lons      = np.array(Data.variables['lon'][:])
    qual_flag = np.array(Data.variables['fds_sample_flags'][:])
    Data.close()
    
    #Eliminate indices with fatal flag or with -9999s in var or var uncertainty
    wbad = np.where((qual_flag == 1) |
                        (qual_flag == 16) |
                        (qual_flag == 32) |
                        (qual_flag == 64) |
                        (qual_flag == 128) |
                        (qual_flag == 256) |
                        (qual_flag == 512) |
                        (qual_flag == 131072) |
                        (qual_flag == 262144) |
                        (var < -999.) |
                        (var_unc < -999.))
    
    var       = np.delete(var, wbad)
    var_unc   = np.delete(var_unc, wbad)
    samp_time = np.delete(samp_time, wbad)
    lats      = np.delete(lats, wbad)
    lons      = np.delete(lons, wbad)
    
    return var, var_unc, lats, lons, samp_time

def file_open_and_clean_CDR1p2(l2_CDR_file):
    Data      = Dataset(l2_CDR_file)
    var       = np.array(Data.variables['wind_speed'][:])
    var_unc   = np.array(Data.variables['wind_speed_uncertainty'][:])
    samp_time = np.array(Data.variables['sample_time'][:])
    lats      = np.array(Data.variables['lat'][:])
    lons      = np.array(Data.variables['lon'][:])
    qual_flag = np.array(Data.variables['fds_sample_flags'][:])
    Data.close()
    
    #Eliminate indices with fatal flag or with -9999s in var or var uncertainty
    #These are the same fatal flags as CDR1p1
    wbad = np.where((qual_flag == 1) |
                        (qual_flag == 16) |
                        (qual_flag == 32) |
                        (qual_flag == 64) |
                        (qual_flag == 128) |
                        (qual_flag == 256) |
                        (qual_flag == 512) |
                        (qual_flag == 2048) |
                        (qual_flag == 4096) |
                        (qual_flag == 8192) |
                        (var < -999.) |
                        (var_unc < -999.))
    
    var       = np.delete(var, wbad)
    var_unc   = np.delete(var_unc, wbad)
    samp_time = np.delete(samp_time, wbad)
    lats      = np.delete(lats, wbad)
    lons      = np.delete(lons, wbad)
    
    return var, var_unc, lats, lons, samp_time

def bin_bounds(min_val, max_val, bin_spacing):
    nbins = ((max_val - min_val)/bin_spacing) + 1
    bins  = np.linspace(min_val, max_val, num = int(nbins))
    
    return bins

def ivwa(vals,uncs):
    #does an Inverse Variance Weighted Average as recommended by Ruf 2018
    top_vals = []
    bot_vals = []

    #want to remove the points where there is no data for the values or the uncertainty
    #I'm not sure why this step is needed. Why is zero bad? If there were no data, 
    #woudln't it get the fill data value of -999. As is, for the one file I tested, there
    #actually aren't any points that vals and uncs of zero.
    
    bad_inds = np.where((vals == 0) | (uncs == 0))
    val_cl = np.delete(vals,bad_inds)
    unc_cl = np.delete(uncs,bad_inds)

    for i in range(len(val_cl)):
        top_vals.append(val_cl[i] / (unc_cl[i]**2))
        bot_vals.append(1./(unc_cl[i]**2))

    top = np.sum(top_vals)
    bot = np.sum(bot_vals)

    result = top/bot

    return result

def unc_prop(vals,uncs):
    #propagates the uncertainty through the IVWA
    bad_inds = np.where((vals == 0) | (uncs == 0))
    unc_cl = np.delete(uncs, bad_inds)

    bot = []
    for i in range(len(unc_cl)):
        bot.append(1./unc_cl[i]**2)

    calc_unc = np.sqrt( 1. / np.sum(bot) )

    return calc_unc

def file_save(lat_axis, lon_axis, binned_vals, binned_uncs, file_date, dirname, product_name):
    #takes in the gridded L3 data, and the Lat/Lon data
    #saves it into a netcdf for that day
    #This directory is inside the directory where all the raw files are
    #This is the only way I could get it to work on the compute node on Asha.
    os.chdir(dirname) 
    #need to make netCDF file
    nc_file = Dataset('L3_'+product_name+'_wind_' + file_date + '_HOURLY.nc', 'w', format='NETCDF4') #was originally SDR3p1 instead of product_name
    nc_file.close() #I close it here so if there was an issue generating I know when

    #make a group for the variables to exist in
    nc_file = Dataset('L3_'+product_name+'_wind_' + file_date + '_HOURLY.nc', 'a') #was originally SDR3p1 instead of product_name
    nc_file.createGroup('data')
    #make dimensions for all the variables
    time = nc_file.createDimension('time',24)
    lat = nc_file.createDimension('lat',len(lat_axis))
    lon = nc_file.createDimension('lon', len(lon_axis))
    #add in the variables
    wind = nc_file.createVariable('wind','f8',('time','lat','lon',), fill_value = -9999, zlib=True,least_significant_digit = 4)
    wind_un = nc_file.createVariable('wind_uncertainty','f8',('time','lat','lon',), fill_value = -9999,zlib=True,least_significant_digit = 4)
    lats = nc_file.createVariable('latitude','f4',('lat',), fill_value = -9999,zlib=True,least_significant_digit = 1)
    lons = nc_file.createVariable('longitude','f4',('lon',), fill_value = -9999,zlib=True,least_significant_digit = 1)
    #write to the variables
    wind[:]    = binned_vals
    wind_un[:] = binned_uncs
    lats[:] = lat_axis
    lons[:] = lon_axis
    #close/save the file
    nc_file.close()

    return 1

def grid_data(vals, unc, lats, lons, time, spacing):
    lat_bins = bin_bounds(-40, 40, spacing) 
    lon_bins = bin_bounds(0, 360, spacing)
    
    lat_axis = lat_bins[0:len(lat_bins)-1]+0.5
    lon_axis = lon_bins[0:len(lon_bins)-1]+0.5
    
    frac_hours = (time/3600.)
    whole_hours = frac_hours.astype(int)
    uniq_hrs, hr_ind = np.unique(whole_hours, return_index = True)
    
    #Add the last indices of the vals array, so that I can subset for all the hours appropriately
    hr_ind = np.append(hr_ind, len(vals))
    
    #maybe put an if in here to check that min and max for the range are the same
    
    #Create array of -9999s to be filled (see test code on test_2Dbinning.ipynb)
    binned_vals = np.zeros([24, len(lat_bins)-1,len(lon_bins)-1]) - 9999.
    binned_uncs = np.zeros([24, len(lat_bins)-1,len(lon_bins)-1]) - 9999.
    
    #print('shape bin vals')
    #print(binned_vals.shape)
    #Subset data by hours
    #will need a for loop over each hour, but for now, just test one hour

    for ihr in range(len(uniq_hrs)):

        ihr_lats = lats[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_lons = lons[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_vals = vals[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_uncs = unc[hr_ind[ihr]:hr_ind[ihr+1]]

        #########################################################################
        #Bin the lats and lons according to the lat_bins and lon_bins
        #call binned_statistic_2d twice:
        #once so have the linearized binnumbers and again so have the bin indices for each lat lon bin

        #3/25/2021 - It seems like the keyword None no longer works, so would need to make it ihr_lats or ihr_lons
        #Updated 7/14/2021 to say ihr_lats instead of None & I made sure the 3.0 results were the same as the original code.
        bin_latlons          = stats.binned_statistic_2d(ihr_lats, ihr_lons, ihr_lats, 'count', 
                                                         bins=[lat_bins, lon_bins])
        
        bin_latlons_explicit = stats.binned_statistic_2d(ihr_lats, ihr_lons, ihr_lats, 'count', 
                                                         bins=[lat_bins, lon_bins], expand_binnumbers=True)
        
        #Get the lat and lon bin indices using the expanded bin indices from bin_latlon_explicit. 
        #Had to subtract one b/c the indices started at 1 and not 0 (not sure why is that way)

        lat_bin_inds = bin_latlons_explicit.binnumber[0]-1
        lon_bin_inds = bin_latlons_explicit.binnumber[1]-1

        #find the unique linearized binnumbers. This will tell me how many bins I'll have to loop through
        uniq_bins, bin_ind = np.unique(bin_latlons.binnumber, return_index = True)
        
        #finally time to bin the values according to lat and lons and apply the dumy function to those values
        for ibin in range(len(uniq_bins)):
            wbin = np.where(bin_latlons.binnumber == uniq_bins[ibin])
            
            #Added the below if statement 7/14/2021 b/c if lat or lon value is outside of lat/lon bins, then it won't be counted
            #in the binned_statistics and the binnumber value will be greater than the lat/lon length, since it's out of bounds
            #See CYGNSS_README.txt for 7/14/2021 for a simple example of what happens when values are outside of the bin range.
            if lat_bin_inds[wbin][0] < len(lat_axis) and lon_bin_inds[wbin][0] < len(lon_axis):
                binned_vals[ihr, lat_bin_inds[wbin][0], lon_bin_inds[wbin][0]] = ivwa(ihr_vals[wbin], ihr_uncs[wbin])
                binned_uncs[ihr, lat_bin_inds[wbin][0], lon_bin_inds[wbin][0]] = unc_prop(ihr_vals[wbin], ihr_uncs[wbin])

    return binned_vals, binned_uncs, lat_axis, lon_axis

#code that will need to be run on Asha
root = os.getcwd()
os.chdir(root + '/cyg_wind_Lev2_SDR3p1/')
SDR_fnames    = sorted(glob.glob('*nc'))
#Need to find matching dates b/c CDR 1.2 data is only available for 45 days in July-Aug 2020
SDR_month_ind = SDR_fnames[0].find('2018')
SDR_dates     = [x[SDR_month_ind:SDR_month_ind + 8] for x in SDR_fnames]

os.chdir(root + '/cyg_wind_Lev2_CDR1p2/')
CDR_fnames    = sorted(glob.glob('*nc'))
CDR_month_ind = CDR_fnames[0].find('2020')
CDR_dates     = [x[CDR_month_ind:CDR_month_ind + 8] for x in CDR_fnames]

#Find matching dates
match_dates, sdr_date_bin_ind, cdr_date_bin_ind = np.intersect1d(SDR_dates, CDR_dates,  return_indices=True)
sdr_array     = np.array(SDR_fnames)
new_sdr_array = sdr_array[sdr_date_bin_ind]

cdr_array     = np.array(CDR_fnames)
new_cdr_array = cdr_array[cdr_date_bin_ind]

fnames_len_diff = new_sdr_array.size - new_cdr_array.size

sdr_dir = root + '/cyg_wind_Lev2_SDR3p1/'
cdr_dir = root + '/cyg_wind_Lev2_CDR1p2/'

if fnames_len_diff != 0: print('fnames length different')

for ifile in range(new_sdr_array.size):
        split_str = new_sdr_array[ifile].split('.')
        year_str  = split_str[2][1:5]
        mon_str   = split_str[2][5:7]
        day_str   = split_str[2][7:9]
        date_str  = year_str + '_' + mon_str + '_' + day_str
        print(date_str)
        
        split_cdr_str = new_cdr_array[ifile].split('.')
        cdr_year_str  = split_cdr_str[2][1:5]
        cdr_mon_str   = split_cdr_str[2][5:7]
        cdr_day_str   = split_cdr_str[2][7:9]
        cdr_date_str  = cdr_year_str + '_' + cdr_mon_str + '_' + cdr_day_str
        print(cdr_date_str)

        if date_str != cdr_date_str: print('Dates do not match. Something wrong')
            
        #Get the SDR and CDR data
        sdr_var, sdr_var_unc, sdr_lats, sdr_lons, sdr_times = file_open_and_clean_SDR3p1(sdr_dir + new_sdr_array[ifile])
        cdr_var, cdr_var_unc, cdr_lats, cdr_lons, cdr_times = file_open_and_clean_CDR1p2(cdr_dir + new_cdr_array[ifile])

        #Grid the matched SDR and CDR data
        binned_sdr_vals, binned_sdr_uncs, sdr_lat_axis, sdr_lon_axis = grid_data(sdr_var, sdr_var_unc, 
                                                                                 sdr_lats, sdr_lons, sdr_times, 1)
        binned_cdr_vals, binned_cdr_uncs, cdr_lat_axis, cdr_lon_axis = grid_data(cdr_var, cdr_var_unc, 
                                                                                 cdr_lats, cdr_lons, cdr_times, 1)

        #Save the matched SDR and CDR data to netcdf files
        file_save(sdr_lat_axis, sdr_lon_axis, binned_sdr_vals, binned_sdr_uncs, date_str, 
                  sdr_dir + 'cyg_wind_1deg_SDR3p1_using_CDR1p2_dates/', 'SDR3p1_CDR1p2dates')
        
        file_save(cdr_lat_axis, cdr_lon_axis, binned_cdr_vals, binned_cdr_uncs, date_str, 
                  cdr_dir + 'cyg_wind_1deg_CDR1p2_allData/', 'CDR1p2')
        
        os.chdir(root)

