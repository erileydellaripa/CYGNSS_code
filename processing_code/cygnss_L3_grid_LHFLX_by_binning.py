import numpy as np
from netCDF4 import Dataset
import glob
import os
from scipy import stats
#import time

#start = time.time()

def file_open_and_clean_CDR(l2_CDR_file):
    Data      = Dataset(l2_CDR_file)
    lhf       = np.array(Data.variables['lhf'][:])
    lhf_unc   = np.array(Data.variables['lhf_uncertainty'][:])
    samp_time = np.array(Data.variables['sample_time'][:])
    lats      = np.array(Data.variables['lat'][:])
    lons      = np.array(Data.variables['lon'][:])
    qual_flag = np.array(Data.variables['quality_flags'][:])
    Data.close()

    fatal_flags = np.array([4])
    mask_bad    = np.empty(qual_flag.size)

    #See page 7 of V1.1 NOAA Level 2 CyGNNS Winds Basic Users Guide: 
    #https://podaac-tools.jpl.nasa.gov/drive/files/allData/cygnss/L2/docs/basic_user_guide_noaa_l2_wind_v1.1.pdf
    for ibit in range(fatal_flags.size):
        bitmask_array = np.fix((qual_flag/2.0**fatal_flags[ibit] % 2))
        wfatal        = np.where(bitmask_array == 1)
        nfatal        = np.count_nonzero(bitmask_array == 1)

        if nfatal > 0: mask_bad[wfatal] = 1

    #In the original method, I was looking for where the qual_flag (i.e., fds_sample_flag) was
    #equal to the power of 2s number. This is not correct. However, I don't think it matters that it was
    #wrong because I was checking for -9999s and where the -9999s are is also where the fatal flags are
    #for the qual_flag (i.e., fds_sample_flags).
    wbad = np.where((mask_bad == 1) | (lhf < -999) | (lhf_unc < -999))

    #Old, incorrect way of finding fatal sample flags
    #Eliminate indices with fatal flag or with -9999s in lhf or lhf uncertainty
    #wbad = np.where((qual_flag == 16) | (lhf < -999.) | (lhf_unc < -999.))
    
    lhf       = np.delete(lhf, wbad)
    lhf_unc   = np.delete(lhf_unc, wbad)
    samp_time = np.delete(samp_time, wbad)
    lats      = np.delete(lats, wbad)
    lons      = np.delete(lons, wbad)
    
    return lhf, lhf_unc, lats, lons, samp_time

def file_open_and_clean_SDR(l2_SDR_file):
    Data      = Dataset(l2_SDR_file)
    lhf       = np.array(Data.variables['lhf'][:])
    lhf_unc   = np.array(Data.variables['lhf_uncertainty'][:])
    samp_time = np.array(Data.variables['sample_time'][:])
    lats      = np.array(Data.variables['lat'][:])
    lons      = np.array(Data.variables['lon'][:])
    qual_flag = np.array(Data.variables['quality_flags'][:])
    Data.close()

    #originally had [4, 9] here, but I'm not sure why b/c 9 corresponds to yslf fatal flag and I'm not using YSLF
    fatal_flags = np.array([4]) 

    bitmask_array = np.empty(fatal_flags.size * qual_flag.size).reshape(fatal_flags.size, qual_flag.size)
    mask_bad      = np.empty(qual_flag.size)

    #See page 7 of V1.1 NOAA Level 2 CyGNNS Winds Basic Users Guide: 
    #https://podaac-tools.jpl.nasa.gov/drive/files/allData/cygnss/L2/docs/basic_user_guide_noaa_l2_wind_v1.1.pdf
    for ibit in range(fatal_flags.size):
        bitmask_array = np.fix((qual_flag/2.0**fatal_flags[ibit] % 2))
        wfatal        = np.where(bitmask_array[ibit, :] == 1)
        nfatal        = np.count_nonzero(bitmask_array[ibit,:] == 1)

        if nfatal > 0: mask_bad[wfatal] = 1

    #In the original method, I was looking for where the qual_flag (i.e., fds_sample_flag) was
    #equal to the power of 2s number. This is not correct. However, I don't think it matters that it was
    #wrong because I was checking for -9999s and where the -9999s are is also where the fatal flags are
    #for the qual_flag (i.e., fds_sample_flags).
    wbad = np.where((mask_bad == 1) | (lhf < -999) | (lhf_unc < -999))

    #Old, incorrect way of finding fatal sample flags
    #Eliminate indices with fatal flag or with -9999s in lhf or lhf uncertainty
    #wbad = np.where((qual_flag == 16) | (qual_flag == 512) | (lhf < -999.) | (lhf_unc < -999.))
    
    lhf       = np.delete(lhf, wbad)
    lhf_unc   = np.delete(lhf_unc, wbad)
    samp_time = np.delete(samp_time, wbad)
    lats      = np.delete(lats, wbad)
    lons      = np.delete(lons, wbad)
    
    return lhf, lhf_unc, lats, lons, samp_time

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
    os.chdir(dirname) #('cyg_flux_1deg_CDR1p1') 
    #need to make netCDF file
    nc_file = Dataset('L3_'+product_name+'_LHF_' + file_date + '_HOURLY.nc', 'w', format='NETCDF4') #was originally CDR, but changed to product name
    nc_file.close() #I close it here so if there was an issue generating I know when

    #make a group for the variables to exist in
    nc_file = Dataset('L3_'+product_name+'_LHF_' + file_date + '_HOURLY.nc', 'a')
    nc_file.createGroup('data')
    #make dimensions for all the variables
    time = nc_file.createDimension('time',24)
    lat = nc_file.createDimension('lat',len(lat_axis))
    lon = nc_file.createDimension('lon', len(lon_axis))
    #add in the variables
    lhf = nc_file.createVariable('lhf','f8',('time','lat','lon',), fill_value = -9999, zlib=True,least_significant_digit = 4)
    lhf_un = nc_file.createVariable('lhf_uncertainty','f8',('time','lat','lon',), fill_value = -9999,zlib=True,least_significant_digit = 4)
    lats = nc_file.createVariable('latitude','f4',('lat',),  fill_value = -9999,zlib=True,least_significant_digit = 1)
    lons = nc_file.createVariable('longitude','f4',('lon',), fill_value = -9999,zlib=True,least_significant_digit = 1)
    #write to the variables
    lhf[:] = binned_vals
    lhf_un[:] = binned_uncs
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
    
    #Add the last indices of the lhf array, so that I can subset for all the hours appropriately
    hr_ind = np.append(hr_ind, len(vals))
    
    #maybe put an if in here to check that min and max for the range are the same
    
    #Create array of -9999s to be filled (see test code on test_2Dbinning.ipynb)
    binned_vals = np.zeros([24, len(lat_bins)-1,len(lon_bins)-1]) - 9999.
    binned_uncs = np.zeros([24, len(lat_bins)-1,len(lon_bins)-1]) - 9999.
    
    #Subset data by hours
    #will need a for loop over each hour, but for now, just test one hour
    
    for ihr in range(len(uniq_hrs)):
        
        ihr_lats = lats[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_lons = lons[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_vals = vals[hr_ind[ihr]:hr_ind[ihr+1]]
        ihr_uncs = unc[hr_ind[ihr]:hr_ind[ihr+1]]
        
        #print([hr_ind[ihr], hr_ind[ihr+1]]
        #print(lhf.shape)
        #print(ihr_lons.shape)
        
        #########################################################################
        #Bin the lats and lons according to the lat_bins and lon_bins
        #call binned_statistic_2d twice:
        #once so have the linearized binnumbers and again so have the bin indices for each lat lon bin

        #3/25/2021 - It seems like the keyword None no longer works, so would need to make it ihr_lats or ihr_lons
        #Updated 8/6/2021 to say ihr_lats instead of None 
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

            #Added the below if statement 8/06/2021 b/c if lat or lon value is outside of lat/lon bins, then it won't be counted
            #in the binned_statistics and the binnumber value will be greater than the lat/lon length, since it's out of bounds
            #See CYGNSS_README.txt for 7/14/2021 for a simple example of what happens when values are outside of the bin range.
            if lat_bin_inds[wbin][0] < len(lat_axis) and lon_bin_inds[wbin][0] < len(lon_axis):
                binned_vals[ihr, lat_bin_inds[wbin][0], lon_bin_inds[wbin][0]] = ivwa(ihr_vals[wbin], ihr_uncs[wbin])
                binned_uncs[ihr, lat_bin_inds[wbin][0], lon_bin_inds[wbin][0]] = unc_prop(ihr_vals[wbin], ihr_uncs[wbin])

    return binned_vals, binned_uncs, lat_axis, lon_axis

root = os.getcwd()
os.chdir(root + '/CYGNSS_L2_SURFACE_FLUX_V2p0/') #used to be SDR3p1
fnames  = sorted(glob.glob('*nc'))

for ifile in fnames:
        split_str = ifile.split('.')
        year_str = split_str[2][1:5]
        mon_str = split_str[2][5:7]
        day_str = split_str[2][7:9]
        date_str = year_str + '_' + mon_str + '_' + day_str
        print(date_str)

        lhf, lhf_unc, lats, lons, samp_time          = file_open_and_clean_CDR(ifile)
        binned_vals, binned_uncs, lat_axis, lon_axis = grid_data(lhf, lhf_unc, lats, lons, samp_time, 1)
        file_save(lat_axis, lon_axis, binned_vals, binned_uncs, date_str, 'CYGNSS_L2_LHFLX_V2p0_1deg', 'SDR2p0')
        os.chdir(root)
        os.chdir(root + '/CYGNSS_L2_SURFACE_FLUX_V2p0/')

#end = time.time()
#print(f"Runtime is {end - start}")
