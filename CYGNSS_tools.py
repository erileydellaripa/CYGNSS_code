import numpy as np
from scipy import stats
from netCDF4 import Dataset
import glob
import os
import bin_ndarray as rebin
import matplotlib.pyplot as plt
import proplot as pplt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as cfeature
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.interpolate import griddata
import math
from rotate_grid import rotate

#Find points in rotated box and save them
# Find points in the rotated box
def points_in_box(in_xs = None, in_ys = None, minx = 0, maxx = 0, miny = 0, maxy = 0):
    wbox = np.where((in_xs <= maxx) & (in_xs >= minx) &
                    (in_ys <= maxy) & (in_ys >= miny))

    n_wbox = np.count_nonzero((in_xs <= maxx) & (in_xs >= minx) &
                              (in_ys <= maxy) & (in_ys >= miny))

    return wbox, n_wbox

def finding_a_line(rise = 0, run = 0, y_pt = 0, x_pt = 0, x_vals = None, y_vals = None):
    slope = rise/run
    #print(slope)
    
    b = y_pt - slope * x_pt
    #print(b)
    #print('')

    y_bounds = slope * x_vals + b

    x_bounds = (y_vals - b)/slope

    return y_bounds, x_bounds, slope, b

def rotate_variable(in_var = None, in_lats = None, in_lons = None, in_origin = None, in_angle = 0):
    #Rotate all the points
    ym, xm = np.meshgrid(in_lats, in_lons)
    xm_reshaped = xm.reshape(in_lats.size * in_lons.size)
    ym_reshaped = ym.reshape(in_lats.size * in_lons.size)

    if in_var.ndim == 2: var_reshaped = in_var.reshape(in_var.shape[0]*in_var.shape[1])
    if in_var.ndim == 3: var_reshaped = in_var.reshape(in_var.shape[0], in_var.shape[1]*in_var.shape[2])
    if in_var.ndim == 4: var_reshaped = in_var.reshape(in_var.shape[0], in_var.shape[1], in_var.shape[2]*in_var.shape[3])

    #print(var_reshaped.ndim)
        
    new_x, new_y = rotate(origin = in_origin, xpoints = xm_reshaped, ypoints = ym_reshaped, angle = in_angle)
     
    sorted_x_inds = np.argsort(new_x)
    sorted_xs     = np.array(new_x)[sorted_x_inds]
    sorted_ys     = np.array(new_y)[sorted_x_inds]
    if var_reshaped.ndim == 1: sorted_var = var_reshaped[sorted_x_inds]  #[nx*ny points]
    if var_reshaped.ndim == 2: sorted_var = var_reshaped[:, sorted_x_inds] #[time, nx*nypoints]
    if var_reshaped.ndim == 3: sorted_var = var_reshaped[:, :, sorted_x_inds]  #[time, level, nx*nypoints]
        
    return sorted_xs, sorted_ys, sorted_var

def rotate_a_box(origin = None, angle = None, x_points = None, y_points = None, var = None):
    
    #Find angle
    rot_angle = math.atan(angle) 

    #Sort the points according the ascending longitude
    new_xs = []
    new_ys = []

    for ipoint in range(len(y_points)): 
        temp_point = (x_points[ipoint], y_points[ipoint])
        temp_pt_tup = tuple(temp_point)
        temp_x, temp_y = rotate(origin, temp_pt_tup, rot_angle)
        new_xs.append(temp_x)
        new_ys.append(temp_y)
    
    sorted_x_inds = np.argsort(new_xs)
    sorted_xs     = np.array(new_xs)[sorted_x_inds]
    sorted_ys     = np.array(new_ys)[sorted_x_inds]
    if var.ndim == 2: sorted_var = var[:, sorted_x_inds]
    if var.ndim == 3: sorted_var = var[:, :, sorted_x_inds]

    return sorted_xs, sorted_ys, sorted_var

def regrid_data(xinfo = None, yinfo = None, orig_xs = None, orig_ys = None,
                orig_var = None, method = 'linear'):
    
    xi = np.arange(xinfo[0], xinfo[1], xinfo[2])
    yi = np.arange(yinfo[0], yinfo[1], yinfo[2])
    
    xi_mesh, yi_mesh = np.meshgrid(xi,yi)
    
    zi = griddata((orig_xs, orig_ys), orig_var, (xi_mesh, yi_mesh), method = method)

    return zi, xi, yi

def lat_avg_rotated_box(sorted_xs, sorted_var):
    
    uniq_xs, uniq_xs_ind, xs_inverse, xs_counts = np.unique(sorted_xs, return_index = True,
                                                            return_inverse = True, return_counts = True)

    lat_avg_array = np.zeros(uniq_xs.size*sorted_var.shape[0]).reshape(sorted_var.shape[0], uniq_xs.size)

    sorted_var_nan = np.where(sorted_var > -99, sorted_var, np.nan)
    
    for iuniq in range(uniq_xs.size):
        
        for itime in range(sorted_var_nan.shape[0]):
            
            var_temp = sorted_var_nan[itime, :, :]
            
            if iuniq == uniq_xs.shape[0]-1: 
                lat_avg_array[itime, iuniq] = np.nanmean(var_temp[uniq_xs_ind[iuniq]:, uniq_xs_ind[iuniq]:])
            else:
                lat_avg_array[itime, iuniq] = np.nanmean(var_temp[uniq_xs_ind[iuniq]:uniq_xs_ind[iuniq+1], 
                                                                  uniq_xs_ind[iuniq]:uniq_xs_ind[iuniq+1]])
                
    #Find daily anomaly by removing daily mean
    diurnal_avg_var = np.nanmean(lat_avg_array, axis = 0)
    
    #Loop through each time and subtract off the mean
    lat_avg_daily_anom_var = [lat_avg_array[it, :] - diurnal_avg_var for it in range(sorted_var.shape[0])] 

    #5) Convert from list to numpy array
    lat_avg_daily_anom_var = np.asarray(lat_avg_daily_anom_var)
            
        
    return lat_avg_array, lat_avg_daily_anom_var, uniq_xs

def lat_avg_rot_regrided_box(in_var = None, in_lats = None):

    var_shape_list  = list(in_var.shape)
    
    #average over latitudes
    lat_avg_array = np.nanmean(in_var, axis = var_shape_list.index(in_lats.shape[0])) 
    
    #Find daily anomaly by removing daily mean
    diurnal_avg_var = np.nanmean(lat_avg_array, axis = 0)
    
    #Loop through each time and subtract the mean
    lat_avg_daily_anom_var = [lat_avg_array[it, :] - diurnal_avg_var for it in range(in_var.shape[0])] 
    
    #Convert from list ot numpy array
    lat_avg_daily_anom_var = np.asarray(lat_avg_daily_anom_var)           
        
    return lat_avg_array, lat_avg_daily_anom_var

def rotate_regrid_var(in_var = None, in_lats = None, in_lons = None, in_origin = np.array([-74.5, 5.5]),
                     in_angle = 0.463648, minx = 0, miny = 0, maxx = 0, maxy = 0, file_save_name = '',
                     xinfo = None, yinfo = None):
    
    #Rotate all the points        
    sorted_x, sorted_y, sorted_var = rotate_variable(in_var = in_var, in_lats = in_lats, in_lons = in_lons,
                                                     in_origin = in_origin, in_angle = in_angle)
    
    #Find points in the box
    wbox_pts, n_wbox_pts = points_in_box(in_xs = sorted_x, in_ys = sorted_y, minx = minx, 
                                         maxx = maxx, miny = miny, maxy = maxy)
    
    #print(wbox_pts)
    #print(n_wbox_pts)
        
    x_box_pts   = sorted_x[wbox_pts]
    y_box_pts   = sorted_y[wbox_pts]
    if in_var.ndim == 4: var_in_box = sorted_var[:, :, wbox_pts]  #[time, level, nlat*nlon]
    if in_var.ndim == 3: var_in_box = sorted_var[:, wbox_pts] #[time, nlat*nlon]
    if in_var.ndim == 2: var_in_box = sorted_var[wbox_pts] #[nlat*nlon]
    var_in_box  = var_in_box.squeeze()
    
    #print(var_in_box.ndim)
    
    # Save the points that fall within the box
    np.savez(file_save_name+'_OrigRes_BoxPts', var_in_box = var_in_box, x_in_box = x_box_pts, y_in_box = y_box_pts)

    # Regrid the wind points in the tilted box
    #if var is [time, level, x/ypoints]
    if var_in_box.ndim == 3:
        new_val_array = np.empty(var_in_box.shape[0]*var_in_box.shape[1]*28*9).reshape(var_in_box.shape[0], 
                                                                                       var_in_box.shape[1], 9, 28)
        for itime in range(var_in_box.shape[0]):
            for ilev in range(var_in_box.shape[1]):
                new_vals, new_xs, new_ys = regrid_data(xinfo = xinfo, yinfo = yinfo, orig_xs = x_box_pts, 
                                                       orig_ys = y_box_pts, orig_var = var_in_box[itime, ilev, :], 
                                                       method = 'linear')
                new_val_array[itime, ilev, :, :] = new_vals
  
    #if var is [time, x/ypoints]
    if var_in_box.ndim == 2:
        new_val_array = np.empty(var_in_box.shape[0]*28*9).reshape(var_in_box.shape[0], 9, 28)
        
        for itime in range(var_in_box.shape[0]):
            new_vals, new_xs, new_ys = regrid_data(xinfo = xinfo, yinfo = yinfo, orig_xs = x_box_pts,
                                                   orig_ys = y_box_pts, orig_var = var_in_box[itime], 
                                                   method = 'linear')
            new_val_array[itime] = new_vals
    
    #if var is [x/ypoints]
    if var_in_box.ndim == 1:
        new_val_array, new_xs, new_ys = regrid_data(xinfo = xinfo, yinfo = yinfo, orig_xs = x_box_pts,\
                                                    orig_ys = y_box_pts, orig_var = var_in_box, 
                                                    method = 'linear')
        
    np.savez(file_save_name + '_Regridded_BoxPts', gridded_var = new_val_array, xs = new_xs, ys = new_ys)
    
    return new_val_array, new_xs, new_ys


#Find points in tilted box
def find_tilted_box_pts(var, lats, lons, y1, y2, x3, x4, offset, diff_val1, diff_val2):
    #Create blank arrays to be filled later
    lats2keep = np.zeros(lats.size * lons.size) - 999.
    lons2keep = np.zeros(lats.size * lons.size) - 999.
    var2keep  = np.zeros_like(var) - 999.

    var2keep = var2keep.reshape(var2keep.shape[0], lats.size*lons.size) if var.ndim == 3 \
    else var2keep.reshape(var2keep.shape[0], var2keep.shape[1], lats.size*lons.size)

    j = 0

    for ilon in range(lons.size):
        
        w      = np.where((lats >= y1[ilon + offset]) & (lats <= y2[ilon + offset]))
        count1 = np.count_nonzero((lats >= y1[ilon + offset]) & (lats <= y2[ilon + offset]))
        
        lon_diff1 = lons[ilon] - x3
        lon_diff2 = lons[ilon] - x4
        
        #Want to keep the following points.
        #Originally was lon_diff1 LE 0 and lon_diff GE 0
        #Had to add the lon_diff1[lon_diff1.size-1] GT 0 to prevent a point being included that exceeded the 
        #precip box eastern most boundary
        w_lon_diff = np.where((lon_diff1 <= diff_val1) & (lon_diff2 >= diff_val2))
        count2     = np.count_nonzero((lon_diff1 <= diff_val1) & (lon_diff2 >= diff_val2))
        
        if count2 > 0:
            itterations = np.min([count1, count2])
            #Keep these lats and lons
            for ict in range(itterations):
                
                if w_lon_diff[0][ict] - np.asarray(w).size < 0:
                    lats2keep[j]      = lats[w[0][w_lon_diff[0][ict]]]
                    lons2keep[j]      = lons[ilon]
                    if var2keep.ndim == 2: var2keep[:, j]= var[:, ilon, w[0][w_lon_diff[0][ict]]]
                    if var2keep.ndim == 3: var2keep[:, :, j] = var[:, :, ilon, w[0][w_lon_diff[0][ict]]]                           
                    j += 1

    #Find lats and lons to keep
    wlats     = np.where(lats2keep > -900)
    wlons     = np.where(lons2keep > -900)

    lats2keep = lats2keep[wlats]
    lons2keep = lons2keep[wlons]
    #Need to use wlats, as wvars for winds will not be the same as wlats & wlons b/c some of the 
    #points are over land and there is no cygnss data over land
    if var2keep.ndim == 2: var2keep = var2keep[:, np.min(wlats):np.max(wlats)+1]
    if var2keep.ndim == 3: var2keep = var2keep[:, :, np.min(wlats):np.max(wlats)+1]
    
    return lats2keep, lons2keep, var2keep

#Open file and set any -9999s to nans
def file_open_and_clean(l3_file, var_name):
    Data = Dataset(l3_file)
    var  = np.array(Data.variables[var_name][:]) 
    lats = np.array(Data.variables['latitude'][:])
    lons = np.array(Data.variables['longitude'][:])
    Data.close()
    
    #Set -9999.9s in precip and HQprecip to nans
    var  = np.where(var < -999., np.nan, var)

    return var, lats, lons

def isolate_region(all_lons, min_lon, max_lon, all_lats, min_lat, max_lat, all_var):
    w_lon = np.where((all_lons >= min_lon) & (all_lons <= max_lon))
    w_lat = np.where((all_lats >= min_lat) & (all_lats <= max_lat))

    #Want the min and max lon and lat indices to encompase all the box, so check where the bounds are
    #and adjust as needed
    lon_min_ind = np.amin(w_lon) - 1 if all_lons[np.amin(w_lon)] > min_lon else np.amin(w_lon)
    lon_max_ind = np.amax(w_lon) + 1 if all_lons[np.amax(w_lon)] < max_lon else np.amax(w_lon)
 
    lat_min_ind = np.amin(w_lat) - 1 if all_lats[np.amin(w_lat)] > min_lat else np.amin(w_lat)
    lat_max_ind = np.amax(w_lat) + 1 if all_lats[np.amax(w_lat)] < max_lat else np.amax(w_lat)
    
    #isolate the lats, lons, and variable for the given region
    if all_var.ndim == 3: region_avg_var = all_var[:, lon_min_ind:lon_max_ind+1, lat_min_ind:lat_max_ind+1]
    if all_var.ndim == 4: region_avg_var = all_var[:, :, lon_min_ind:lon_max_ind+1, lat_min_ind:lat_max_ind+1]
    region_lats    = all_lats[lat_min_ind:lat_max_ind+1]
    region_lons    = all_lons[lon_min_ind:lon_max_ind+1]

    return region_avg_var, region_lats, region_lons


def find_time_avg_var(dir_name, var_name):
    fnames   = sorted(glob.glob(dir_name +'*nc'))
    print(len(fnames))
    #Create just a dummy array to use as a check below and then overwrite 
    var_total = np.zeros(10) 
    
    for i in range(len(fnames)):

        ifile = fnames[i]
        
        if i == 0: month_ind = ifile.find('08') #will need to change this if first file month isn't '08'
            
        month = float(ifile[month_ind:month_ind+2])
        
        #Only continue for months May-Oct
        if month >= 5. and month <= 10.:
            var, lats, lons = file_open_and_clean(ifile, var_name) #options are lhf, wind, precip
            
            if len(var_total) == 10:
                print('inside if to make var_total')
                var_total   = np.zeros_like(var) #remake var_total to fill below. Should only be done once!
                var_tot_cnt = np.zeros_like(var)
            
            #Find where var is finite. where the array is finite will be flagged as TRUE otherwise FALSE
            ind_finite = np.isfinite(var)
            
            #Create a masked array of 1s and 0s for finite and nan values, respectively
            #itteratively sum the 1s and 0s over each file to keep track of how many "true" values there are for each bin
            var_tot_cnt += ind_finite.astype(np.int)*1. #assigns TRUES 1 and FALSE 0
            
            #Keep a running sum of the var values that fall into each bin, while avoiding nan values.
            var_total    = np.nansum(np.stack((var_total,var)), axis=0)

    #Average the var over all the files (i.e., days since files represent 1 day)
    avg_var = var_total/var_tot_cnt

    #print(np.nanmax(avg_var))
    #print(np.nanmin(avg_var))
    
    return avg_var, lats, lons

def find_time_avg_var_certain_days(dir_name = '~/', var_name = 'precip', date_name_array = None, st_month = '08'):
    fnames   = sorted(glob.glob(dir_name +'*nc'))
    #print(len(fnames))
    #Create just a dummy array to use as a check below and then overwrite 
    var_total = np.zeros(10) 
    
    for i in range(len(fnames)):

        ifile = fnames[i]
        
        if i == 0: month_ind = ifile.find(st_month) #will need to change this if first file month isn't '08'
            
        month     = float(ifile[month_ind:month_ind+2])
        date_name = ifile[month_ind-5:month_ind+5]
        
        #Only continue for months May-Oct
        if month >= 5. and month <= 10.:
            
            if date_name_array is not None: 
                #Only open and use matching date_list
                count_date = np.count_nonzero(date_name_array == date_name)

                if count_date == 1: 
                    var, lats, lons = file_open_and_clean(ifile, var_name)
                    
                    #print(date_name)
                    
                    if len(var_total) == 10:
                        #print('inside if to make var_total')
                        var_total   = np.zeros_like(var) #remake var_total to fill below. Should only be done once!
                        var_tot_cnt = np.zeros_like(var)
                        
                    #Find where var is finite. where the array is finite will be flagged as TRUE otherwise FALSE
                    ind_finite = np.isfinite(var)
                    
                    #Create a masked array of 1s and 0s for finite and nan values, respectively
                    #itteratively sum the 1s and 0s over each file to keep track of how many "true" values there are for each bin
                    var_tot_cnt += ind_finite.astype(np.int)*1. #assigns TRUES 1 and FALSE 0
                    
                    #Keep a running sum of the var values that fall into each bin, while avoiding nan values.
                    var_total    = np.nansum(np.stack((var_total,var)), axis=0)

                    
            else: 
                
                var, lats, lons = file_open_and_clean(ifile, var_name)
               
                if len(var_total) == 10:
                    print('inside if to make var_total')
                    var_total   = np.zeros_like(var) #remake var_total to fill below. Should only be done once!
                    var_tot_cnt = np.zeros_like(var)
                        
                #Find where var is finite. where the array is finite will be flagged as TRUE otherwise FALSE
                ind_finite = np.isfinite(var)
                
                #Create a masked array of 1s and 0s for finite and nan values, respectively
                #itteratively sum the 1s and 0s over each file to keep track of how many "true" values there are for each bin
                var_tot_cnt += ind_finite.astype(np.int)*1. #assigns TRUES 1 and FALSE 0
                
                #Keep a running sum of the var values that fall into each bin, while avoiding nan values.
                var_total    = np.nansum(np.stack((var_total,var)), axis=0)

    #Average the var over all the files (i.e., days since files represent 1 day)
    avg_var = var_total/var_tot_cnt

    #print(np.nanmax(avg_var))
    #print(np.nanmin(avg_var))
    
    return avg_var, lats, lons

#Should maybe add a check to make sure that the dimensions are correct
def calc_diurnal_anom(input_var, input_lats, ntimes, time_factor):

    #1) Average over the latitudes. Array is [time, lon, lat]
    #make the shape of the input variable a list
    var_shape_list  = list(input_var.shape)
    lat_avg_avg_var = np.nanmean(input_var, axis = var_shape_list.index(input_lats.shape[0])) #axis should be equal to 2
    
    #2) Average over time to get daily mean
    #would need to change what the axis is equal to, if the time dimension changes position in the array
    diurnal_avg_var = np.nanmean(lat_avg_avg_var, axis = 0)
    
    print(np.nanmax(lat_avg_avg_var))
    print(np.nanmin(lat_avg_avg_var))
    print(diurnal_avg_var)
    
    #Find diurnal anomalies
    #3) Create time array (will need for plotting later); dont' think this is necessary
    #times_per_day = np.arange(ntimes)*time_factor + 0.5
    
    #4) Loop through each time and subtract off the mean
    lat_avg_daily_anom_var = [lat_avg_avg_var[itime, :] - diurnal_avg_var for itime in range(input_var.shape[0])] 

    #5) Convert from list to numpy array
    lat_avg_daily_anom_var = np.asarray(lat_avg_daily_anom_var)

    print(np.nanmin(lat_avg_daily_anom_var))
    print(np.nanmax(lat_avg_daily_anom_var))
    #print(lat_avg_daily_anom_var.shape)
    
    return lat_avg_daily_anom_var

#need to average over all the latitudes at each unique longitude for a tilted box
def lat_avg_tilted_box(lons2keep, var2keep):
    
    uniq_lons = np.unique(lons2keep)

    #Create array to be filled
    lat_avg_array = np.zeros(uniq_lons.size*var2keep.shape[0]).reshape(var2keep.shape[0], uniq_lons.size)

    #loop through uniq_lons and find the latitudinal average for each longitude
    for iuniq in range(uniq_lons.size):
        w2avg   = np.where(lons2keep == uniq_lons[iuniq])
        
        for itime in range(var2keep.shape[0]):
            var_temp = var2keep[itime, :, :]
            lat_avg_array[itime, iuniq] = np.nanmean(var_temp[w2avg[0], w2avg[0]])
    
    #Find daily anomaly by removing daily mean
    diurnal_avg_var = np.nanmean(lat_avg_array, axis = 0)
    
    #Loop through each time and subtract off the mean
    lat_avg_daily_anom_var = [lat_avg_array[itime, :] - diurnal_avg_var for itime in range(var2keep.shape[0])] 

    #5) Convert from list to numpy array
    lat_avg_daily_anom_var = np.asarray(lat_avg_daily_anom_var)
    
    return lat_avg_array, lat_avg_daily_anom_var, uniq_lons

def make_daily_anom_Hovmoller(in_var1 = None, in_lons1 = None, in_var2 = None, in_lons2 = None, 
                              ntimes = 8, factor = 3, time_offset = 1.5, title = '',
                              cbar_title = '', save_name = '', clevs1 = None, clevs2 = None,
                              cmap1 = 'bwr', cmap2 = 'bwr', zero_ind = 10, zero_thick = 2,
                              yaxis_name = 'Time (UTC)', xaxis_name = 'Longitude',
                             do_invert_xaxis = False, do_invert_yaxis = False,
                            yminor_ticks = 0.5, xminor_ticks = 0.25, line_widths = np.zeros(21)+1,
                            extend_option = 'neither', vmin_val = 0, vmax_val = 0):
    
    #Make time array
    times_per_day = np.arange(ntimes)*factor + time_offset
    if ntimes == 8:  ntimes_str = ["0130", "0430", "0730", "1030", "1330", "1630", "1930", "2230"]
    if ntimes == 24: ntimes_str = ["0030", "0130", "0230", "0330", "0430", "0530", "0630", "0730", 
                                   "0830", "0930", "1030", "1130", "1230", "1330", "1430", "1530",
                                   "1630", "1730", "1830", "1930", "2030", "2130", "2230", "2330"]
        
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(5.5, 4))

    cf = ax.contourf(in_lons1, times_per_day, in_var1,levels = clevs1, cmap = plt.get_cmap(cmap1),
                         extend = extend_option, vmin = vmin_val, vmax = vmax_val)

    #overlay contour lines. By default if one color is used, the negative values are dashed, but 
    #if you use a color table,  you need to set the linestyle
    if in_var2 is not None:
        cs = ax.contour(in_lons2, times_per_day, in_var2, levels= clevs2,
                        linestyles = np.where(clevs2 >= 0, "-", "--"),linewidths = line_widths,
                        cmap= plt.get_cmap(cmap2)) #'RdBu_r')
        #ax.clabel(cs, inline=1, fontsize=20, fmt='%1.1f') #This adds extra inline labels
#                    colors = 'black')

    if do_invert_xaxis is True: ax.invert_xaxis()
    if do_invert_yaxis is True: ax.invert_yaxis()
        
    #set y-axis ticks
    yticks = times_per_day
    ax.set_yticks(yticks)
    ax.set_yticklabels(ntimes_str)#, size = 18) 

    #Add title and axes titles
    ax.set_title(title, fontsize=14) #+ ", May-Oct; 2°N-7°N"
    ax.set_xlabel(xaxis_name, fontsize = 12)
    ax.set_ylabel(yaxis_name, fontsize = 12)
    if in_var2 is not None: ax.clabel(cs, cs.levels, inline=True, fontsize=10, fmt = f"%.1f")

    #Want to make axis tick marks bigger, but can't figure out how
    #ax.tick_params(axis='both', which='major', labelsize=50)

    # Set 0 level contour line to a thicker linewidth
    if in_var2 is not None: cs.collections[zero_ind].set_linewidth(zero_thick)

    # Label the contour levels -4, 0, and 4
    #cl = ax.clabel(cs, fmt='%d', levels=[-.3, 0, .3])

    #Add a color bar
    cbar = fig.colorbar(cf)
    cbar.set_label(cbar_title, size=10, weight = 'bold')
    cbar.ax.tick_params(labelsize=12)
    plt.tick_params(labelsize=12)

    #Add minor tick axis
    ax.yaxis.set_minor_locator(MultipleLocator(yminor_ticks)) #0.5
    ax.xaxis.set_minor_locator(MultipleLocator(xminor_ticks)) #0.25

    #plt.savefig('GPM_prec_hourly.png')
    plt.savefig(save_name + '.png')
    
    return cf

def make_3hrly_spatial_plots(in_var = None, in_lons = None, in_lats = None, in_var2 = None, in_lons2 = None,
                                 in_lats2 = None, clevs = None, cclevs = None, var2_colors = 'black',cmap = 'viridis',
                                 topo_levs = [1000], topo_color = 'black', topo_widths = 1,
                                 topo_lons = None, topo_lats = None,topo_data = None, fig_title = '',
                                 cbar_title = '', save_name = '', do_daily_anom = 'False', do_box = True,
                                 coast_color = 'white', quiverx = None, quivery = None, skip = 1, scale_length = 2,
                                 do_transpose = 'True', keyval = 8, quiver_lons = None, quiver_lats = None, width = 0.005,
                                 xloc = 0, yloc = 0,x_line1 = None, y_line1 = None, x_line2 = None, y_line2 = None,
                                 x_line3 = None, y_line3 = None, x_line4 = None, y_line4 = None, box_color = 'white',
                                 cwidths = 1, var2_widths = 1, linewidth = 2, extend_option = 'neither',
                                 vmax_val = None, vmin_val = None, do_topo = False):

    ntimes_str   = ["0130 UTC (2030 LT)", "0430 UTC (2330 LT)", "0730 UTC (0230 LT)", 
                    "1030 UTC (0530 LT)", "1330 UTC (0830 LT)", "1630 UTC (1130 LT)", 
                    "1930 UTC (1430 LT)", "2230 UTC (1730 LT)"]
    title_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    #Find daily mean to remove later if desired
    diurnal_avg         = np.nanmean(in_var,  axis = 0)
    if in_var2 is not None: diurnal_avg2 = np.nanmean(in_var2, axis = 0)
        
    if quiverx is not None and quivery is not None: 
        quiverx_diurnal_avg = np.nanmean(quiverx, axis = 0)
        quivery_diurnal_avg = np.nanmean(quivery, axis = 0)
    
    if do_transpose == 'True': 
        diurnal_avg = np.transpose(diurnal_avg)

        if quiverx is not None and quivery is not None:
            quiverx_diurnal_avg = np.transpose(quiverx_diurnal_avg)
            quivery_diurnal_avg = np.transpose(quivery_diurnal_avg)
    
    #transform input lat/lons into projection coordinate points
    #Sean's example code said that even if you use a different projection above, use PlateCarree here.
    #this doesn't work for me b/c my x and y are not the same size. Maybe I don't have to use this?
    #xformed_pts = proj.transform_points(src_crs = ccrs.PlateCarree(), x = prec_lons, y = prec_lats)
    #print(xformed_pts.shape)

    #create matplotlib figure and give it a size/aspect ratio
    fig, axs = plt.subplots(2, 4, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(20,8)) #was originally (20,8)

    #flatten the axs
    ax = axs.flatten()
    
    #define x and y tick marks
    xticks = [-95, -105, -100, -95, -90, -85, -80, -75] #changed to -110 for some plots
    yticks = [0, 5, 10, 15]
    
    quiver_slices = (slice(None, None, skip), slice(None, None, skip))
    quiver_kwargs = {'headlength': 4, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy',
                     'scale': scale_length, 'width': width}

    X, Y = np.meshgrid(in_lons, in_lats)
    if in_var2 is not None: X2, Y2 = np.meshgrid(in_lons2, in_lats2)
    if do_topo is True: Xtopo, Ytopo = np.meshgrid(topo_lons, topo_lats)
    
    for i in range(8):

        #set lon/lat extent
        ax[i].set_extent([-95, -74.5, -0.5, 15.5]) #was -95, -110
        
        #draw coastlines
        ax[i].coastlines(resolution = '10m', color = coast_color, linewidth = linewidth)

        #draw gridlines and write ticks
        gl = ax[i].gridlines(xlocs=xticks, ylocs=yticks, linestyle = ':', draw_labels = True)

        #remove the labels from the top and right axes
        gl.top_labels   = False
        gl.right_labels = False

        # tell matplotlib that the ticks are lats/lons so that they look nicer
        ax[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
        ax[i].yaxis.set_major_formatter(LATITUDE_FORMATTER)
        gl.xlabel_style = {'size': 14} 
        gl.ylabel_style = {'size':14}
        
        #draw a filled contour plot
        var_temp = in_var[i, :, :]
        if in_var2 is not None: var2_temp = in_var2[i, :, :]
            
        if do_transpose == 'True': var_temp = np.transpose(var_temp)
            
        #Add if statement to remove daily mean if desired
        if do_daily_anom == 'True':
                var_temp = var_temp - diurnal_avg
                if in_var2 is not None:
                    var2_temp = var2_temp - diurnal_avg2
        
        cf = ax[i].contourf(X, Y, var_temp, levels = clevs, cmap = cmap, extend = extend_option,
                                vmax = vmax_val, vmin = vmin_val)

        if do_topo is True:
                cs = ax[i].contour(Xtopo, Ytopo, topo_data, levels = topo_levs, colors = topo_color,
                                   linewidths = topo_widths, linestyles = '-')
        if in_var2 is not None:
            cs = ax[i].contour(X2, Y2, var2_temp, levels = cclevs, colors = var2_colors,
                                   linestyles = '-', linewidths = var2_widths)

        #Add a title above each subplot
        ax[i].set_title(title_labels[i] + ") " + ntimes_str[i], fontsize=18, horizontalalignment = 'center')
        
        #Add box to plot where Hovmollers come from
        if do_box is True:
            if y_line1 is None:
                ax[i].hlines(2.5, -86.5, -78.5, colors = box_color, linestyle = 'solid')
                ax[i].hlines(6.5, -86.5, -78.5, colors = box_color, linestyle = 'solid')
                ax[i].vlines(-86.5, 2.5, 6.5,   colors = box_color, linestyle = 'solid')
                ax[i].vlines(-78.5, 2.5, 6.5,   colors = box_color, linestyle = 'solid')
            else:
                ax[i].plot(x_line1, y_line1, color = box_color, linewidth = 3)
                ax[i].plot(x_line2, y_line2, color = box_color, linewidth = 3)
                ax[i].plot(x_line3, y_line3, color = box_color, linewidth = 3)
                ax[i].plot(x_line4, y_line4, color = box_color, linewidth = 3)
        
        if quiverx is not None and quivery is not None:
            quiverx_temp = quiverx[i, :, :]
            quivery_temp = quivery[i, :, :]
    
            if do_transpose == 'True': 
                quiverx_temp = np.transpose(quiverx_temp)
                quivery_temp = np.transpose(quivery_temp)
                
            if do_daily_anom == 'True':
                quiverx_temp = quiverx_temp - quiverx_diurnal_avg
                quivery_temp = quivery_temp - quivery_diurnal_avg
                
            q = ax[i].quiver(quiver_lons[::skip], quiver_lats[::skip], quiverx_temp[quiver_slices], 
                          quivery_temp[quiver_slices], **quiver_kwargs)
            qk = ax[i].quiverkey(q, xloc, yloc, keyval, str(keyval) + r' m $s^-$$^1$', labelpos='E',
                   coordinates='data')
            
    #Add main title
    fig.suptitle(fig_title, fontsize = 20)

    #create the color bar for the filled contour plot
    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) #arguments for add_axes are [left, bottom, width, height]
    cb = plt.colorbar(cf, cax = cb_ax)

    #set font sizes
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_title, size=16)
    

    plt.savefig(save_name + '.png')


def make_3hrly_spatial_plots_with_overlay(in_var = None, in_var2 = None, in_lons = None, in_lats = None, in_lons2 = None, in_lats2 = None, 
                                          clevs = None, clevs2 = None, cmap = 'viridis',cmap2 = 'binary', line_widths = None,
                                          topo_levs = [1000], topo_color = 'black', topo_widths = 1,
                                          topo_lons = None, topo_lats = None,topo_data = None, do_topo = False, do_box = True,
                                          fig_title = '', cbar_title = '', save_name = '', do_daily_anom = 'False', coast_color = 'white',
                                          quiverx = None, quivery = None, skip = 1, scale_length = 2, do_transpose = 'True',
                                          keyval = 8, quiver_lons = None, quiver_lats = None, width = 0.005, xloc = 0, yloc = 0,
                                          x_line1 = None, y_line1 = None, x_line2 = None, y_line2 = None, x_line3 = None,
                                          y_line3 = None, x_line4 = None, y_line4 = None, box_color = 'white',
                                          extend_option = 'neither', vmax_val = None, vmin_val = None, linewidth = 2, 
                                          zero_ind = -99, zero_thick = 3):

    ntimes_str   = ["0130 UTC (2030 LT)", "0430 UTC (2330 LT)", "0730 UTC (0230 LT)", 
                    "1030 UTC (0530 LT)", "1330 UTC (0830 LT)", "1630 UTC (1130 LT)", 
                    "1930 UTC (1430 LT)", "2230 UTC (1730 LT)"]
    title_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    #Find daily mean to remove later if desired
    diurnal_avg  = np.nanmean(in_var,  axis = 0)
    if in_var2 is not None: diurnal_avg2 = np.nanmean(in_var2, axis = 0)
        
    if quiverx is not None and quivery is not None: 
        quiverx_diurnal_avg = np.nanmean(quiverx, axis = 0)
        quivery_diurnal_avg = np.nanmean(quivery, axis = 0)
    
    if do_transpose == 'True': 
        diurnal_avg = np.transpose(diurnal_avg)
        if in_var2 is not None: diurnal_avg2 = np.transpose(diurnal_avg2)

        if quiverx is not None and quivery is not None:
            quiverx_diurnal_avg = np.transpose(quiverx_diurnal_avg)
            quivery_diurnal_avg = np.transpose(quivery_diurnal_avg)
    
    #transform input lat/lons into projection coordinate points
    #Sean's example code said that even if you use a different projection above, use PlateCarree here.
    #this doesn't work for me b/c my x and y are not the same size. Maybe I don't have to use this?
    #xformed_pts = proj.transform_points(src_crs = ccrs.PlateCarree(), x = prec_lons, y = prec_lats)
    #print(xformed_pts.shape)

    #create matplotlib figure and give it a size/aspect ratio
    fig, axs = plt.subplots(2, 4, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(20,8)) #was originally (20,8)

    #flatten the axs
    ax = axs.flatten()
    
    #define x and y tick marks
    xticks = [-95, -105, -100, -95, -90, -85, -80, -75] #changed to -110 for some plots
    yticks = [0, 5, 10, 15]
    
    quiver_slices = (slice(None, None, skip), slice(None, None, skip))
    quiver_kwargs = {'headlength': 4, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy',
                     'scale': scale_length, 'width': width}

    X, Y = np.meshgrid(in_lons, in_lats)
    if in_var2 is not None: X2, Y2 = np.meshgrid(in_lons2, in_lats2)
    if do_topo is True: Xtopo, Ytopo = np.meshgrid(topo_lons, topo_lats)
        
    for i in range(8):

        #set lon/lat extent
        ax[i].set_extent([-95, -74.5, -0.5, 15.5]) #was -95, -110
        
        #draw coastlines
        ax[i].coastlines(resolution = '10m', color = coast_color, linewidth = linewidth)

        #draw gridlines and write ticks
        gl = ax[i].gridlines(xlocs=xticks, ylocs=yticks, linestyle = ':', draw_labels = True)

        #remove the labels from the top and right axes
        gl.top_labels   = False
        gl.right_labels = False

        # tell matplotlib that the ticks are lats/lons so that they look nicer
        ax[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
        ax[i].yaxis.set_major_formatter(LATITUDE_FORMATTER)
        gl.xlabel_style = {'size': 14} 
        gl.ylabel_style = {'size':14}
        
        #draw a filled contour plot
        var_temp  = in_var[i, :, :]
        if in_var2 is not None: var_temp2 = in_var2[i, :, :] 
            
        if do_transpose == 'True': 
            var_temp = np.transpose(var_temp)
            if in_var2 is not None: var_temp2 = np.transpose(var_temp2)
            
        #Add if statement to remove daily mean if desired
        if do_daily_anom == 'True': 
            var_temp = var_temp - diurnal_avg
            if in_var2 is not None: var_temp2 = var_temp2 - diurnal_avg2
        
        cf = ax[i].contourf(X, Y, var_temp, levels = clevs, cmap = plt.get_cmap(cmap), extend = extend_option,
                                vmax = vmax_val, vmin = vmin_val)
        
        if in_var2 is not None:
            ct = ax[i].contour(X2, Y2, var_temp2, levels = clevs2, linestyles = np.where(clevs2 >= 0, "-", "--"),
                               linewidths = line_widths, colors = 'black')#,  cmap= plt.get_cmap(cmap2))
            
            # Set 0 level contour line to a thicker linewidth
            if zero_ind > -99: ct.collections[zero_ind].set_linewidth(zero_thick)

        if do_topo is True:
                cs = ax[i].contour(Xtopo, Ytopo, topo_data, levels = topo_levs, colors = topo_color,
                                   linewidths = topo_widths, linestyles = '-')

        #Add a title above each subplot
        ax[i].set_title(title_labels[i] + ") " + ntimes_str[i], fontsize=18, horizontalalignment = 'center') 
        
        #Add box to plot where Hovmollers come from
        if do_box is True:
            if y_line1 is None:
                ax[i].hlines(2.5, -86.5, -78.5, colors = box_color, linestyle = 'solid')
                ax[i].hlines(6.5, -86.5, -78.5, colors = box_color, linestyle = 'solid')
                ax[i].vlines(-86.5, 2.5, 6.5,   colors = box_color, linestyle = 'solid')
                ax[i].vlines(-78.5, 2.5, 6.5,   colors = box_color, linestyle = 'solid')
            else:
                ax[i].plot(x_line1, y_line1, color = box_color, linewidth = 3)
                ax[i].plot(x_line2, y_line2, color = box_color, linewidth = 3)
                ax[i].plot(x_line3, y_line3, color = box_color, linewidth = 3)
                ax[i].plot(x_line4, y_line4, color = box_color, linewidth = 3)
        
        if quiverx is not None and quivery is not None:
            quiverx_temp = quiverx[i, :, :]
            quivery_temp = quivery[i, :, :]
    
            if do_transpose == 'True': 
                quiverx_temp = np.transpose(quiverx_temp)
                quivery_temp = np.transpose(quivery_temp)
                
            if do_daily_anom == 'True':
                quiverx_temp = quiverx_temp - quiverx_diurnal_avg
                quivery_temp = quivery_temp - quivery_diurnal_avg
                
            q = ax[i].quiver(quiver_lons[::skip], quiver_lats[::skip], quiverx_temp[quiver_slices], 
                          quivery_temp[quiver_slices], **quiver_kwargs)
            qk = ax[i].quiverkey(q, xloc, yloc, keyval, str(keyval) + r' m $s^-$$^1$', labelpos='E',
                   coordinates='data')
            
    #Add main title
    fig.suptitle(fig_title, fontsize = 20)

    #create the color bar for the filled contour plot
    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) #arguments for add_axes are [left, bottom, width, height]
    cb = plt.colorbar(cf, cax = cb_ax)

    #set font sizes
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cbar_title, size=14)
    

    plt.savefig(save_name + '.png')

def spatial_plot_of_mean_var(in_var = None, in_var2 = None, in_uwind = None, in_vwind = None, 
                             in_lons = None, in_lats = None, in_lons2 = None, in_lats2 = None,
                             topo_levs = [1000], topo_color = 'grey', topo_widths = 2,
                             topo_lons = None, topo_lats = None,topo_data = None, do_topo = True,
                             box_color = 'black', box_width = 2, skip = 2, scale_length = 0.4, keyval = 1, 
                             width = 0.01, xloc = -74, yloc = 15.7, save_name = '', levs = None, 
                             clevs = None, ccolor = 'black', color_choice = 'bwr', extend_option = 'neither',
                             plot_title = '',coast_color = 'black', xline1 = None, xline2 = None, xline3 = None, 
                             xline4 = None, yline1 = None, yline2 = None, yline3 = None, line_widths = None,
                             yline4 = None, colorbar_name = '$ms^{-1}$', do_box = False):
    
    if do_topo is True: Xtopo, Ytopo = np.meshgrid(topo_lons, topo_lats)

    fig  = pplt.figure(refwidth = 3)
    axs  = fig.subplots(111, proj='cyl')

    cf = axs.contourf(in_lons, in_lats, np.transpose(in_var), levels = levs, cmap = color_choice, 
                      extend = extend_option)
    
    if in_var2 is not None:
        cs = axs.contour(in_lons2, in_lats2, in_var2, levels = clevs, colors = ccolor,
                        linewidths = line_widths, linestyles = '-') #'RdBu_r') #linestyles = np.where(clevs >= 0, "-", "--")
        
    if do_topo is True:
        ct = axs.contour(topo_lons, topo_lats, topo_data, levels = topo_levs, colors = topo_color,
                         linewidths = topo_widths, linestyles = '-')
    
    axs.format(title= plot_title, grid=True, labels = True, lonlim=(-95, -74.5), latlines = 2,
               lonlines = 5, latlim=(0, 15), land = False, coast = True, borders = False, 
               coastlinewidth=2, coastcolor = coast_color)
    
    if do_box is True:
        axs.plot(xline1, yline1, color = box_color, linewidth = box_width)
        axs.plot(xline2, yline2, color = box_color, linewidth = box_width)
        axs.plot(xline3, yline3, color = box_color, linewidth = box_width)
        axs.plot(xline4, yline4, color = box_color, linewidth = box_width)

    if in_uwind is not None:
        quiver_slices = (slice(None, None, skip), slice(None, None, skip))
        quiver_kwargs = {'headlength': 4, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy',
                             'scale': scale_length, 'width': width}

        quiverx_temp = np.transpose(in_uwind)
        quivery_temp = np.transpose(in_vwind)

        q = axs.quiver(in_lons[::skip], in_lats[::skip], quiverx_temp[quiver_slices], 
                       quivery_temp[quiver_slices], **quiver_kwargs)

        qk = axs.quiverkey(q, xloc, yloc, keyval, str(keyval) + r' m $s^{-1}$', labelpos='E',
                           coordinates='data')

    fig.colorbar(cf, label= colorbar_name, length=0.9)
    
    plt.savefig(save_name + '.png')

    return cf

##############################################################################################################################
###Functions for processing ERA5
##############################################################################################################################
def avg_ERA5wind_certain_days(CYGdays_in = None, uwind_in = None, vwind_in = None, speed_in = None, ERA5_times = None):
    
    #Find the intersection of the arrays
    #PC12_norm_prop_days
    event_dates, CYGdays_ind_with_ERA5, ERA5_ind_with_CYGdays = np.intersect1d(CYGdays_in, ERA5_times, return_indices = True)

    #Isolate days for event (i.e., prop vs. non-prop) of interest

    #Make blank arrays to be filled later
    time_dim = len(ERA5_ind_with_CYGdays)*24
    lat_dim = uwind_in.shape[1]
    lon_dim = vwind_in.shape[2]

    event_uwind = np.zeros(time_dim * lat_dim * lon_dim).reshape(time_dim, lat_dim, lon_dim)
    event_vwind = np.zeros(time_dim * lat_dim * lon_dim).reshape(time_dim, lat_dim, lon_dim)
    event_speed = np.zeros(time_dim * lat_dim * lon_dim).reshape(time_dim, lat_dim, lon_dim)
    event_times = np.zeros(time_dim).reshape(time_dim)
    
    j = 0
    
    for iday in range(len(ERA5_ind_with_CYGdays)):
        start_ind  = ERA5_ind_with_CYGdays[iday]
        
        temp_uwind = uwind_in[start_ind:start_ind+24, :, :]
        temp_vwind = vwind_in[start_ind:start_ind+24, :, :]
        temp_speed = speed_in[start_ind:start_ind+24,  :, :]
        
        #Used temp_times to make sure that the days were being selected correctly
        temp_times = ERA5_times[start_ind:start_ind+24]
        
        #print(temp_times[0], temp_times[-1])
        
        #Fill a blank array that will consist of all the days that qualify for a certain criteria
        event_uwind[j:j+24, :, :] = temp_uwind
        event_vwind[j:j+24, :, :] = temp_vwind
        event_speed[j:j+24, :, :] = temp_speed
        event_times[j:j+24]       = temp_times
        j += 24

    #Then average over every 24th array element to get average diurnal cycle for certain days that qualify as pre-defined event
    mean_daily_speed = np.zeros(24 * uwind_in.shape[2] * uwind_in.shape[1]).reshape(24, uwind_in.shape[2], uwind_in.shape[1])
    mean_daily_uwind = np.zeros(24 * uwind_in.shape[2] * uwind_in.shape[1]).reshape(24, uwind_in.shape[2], uwind_in.shape[1])
    mean_daily_vwind = np.zeros(24 * uwind_in.shape[2] * uwind_in.shape[1]).reshape(24, uwind_in.shape[2], uwind_in.shape[1])
    
    #Find average for each hour of the day
    for ihr in (range(24)):
        
        speed_ihr = event_speed[ihr::24,:,:] #every 24th point is the same hour
        uwind_ihr = event_uwind[ihr::24,:,:] 
        vwind_ihr = event_vwind[ihr::24,:,:]
        test_time = event_times[ihr::24]
        
        #print(speed_ihr.shape)
        #print(test_time[0], test_time[-1])
        
        #checked this on 2/15/2022 to make sure that the selection of every 24th element was being averaged as I expected
        #print(test_time)
        #print(np.squeeze(uwind_ihr[:, 0, 0]))
        
        #find the average of that hour over all days
        mean_hr   = np.mean(speed_ihr, axis = 0)
        mean_u    = np.mean(uwind_ihr, axis = 0)
        mean_v    = np.mean(vwind_ihr, axis = 0)
        
        #Rotate the array so that it's lon x lat instead of lat x lon and make a daily array
        mean_daily_speed[ihr,:,:] = np.transpose(mean_hr)
        mean_daily_uwind[ihr,:,:] = np.transpose(mean_u)
        mean_daily_vwind[ihr,:,:] = np.transpose(mean_v)
        
    return mean_daily_speed, mean_daily_uwind, mean_daily_vwind

def avg_ERA5var_certain_days(CYGdays_in = None, var_in = None, ERA5_times = None):
    
    #Find the intersection of the arrays
    #PC12_norm_prop_days
    event_dates, CYGdays_ind_with_ERA5, ERA5_ind_with_CYGdays = np.intersect1d(CYGdays_in, ERA5_times, return_indices = True)

    #Isolate days for event (i.e., prop vs. non-prop) of interest

    #Make blank arrays to be filled later
    time_dim = len(ERA5_ind_with_CYGdays)*24
    lat_dim = var_in.shape[1]
    lon_dim = var_in.shape[2]

    event_var   = np.zeros(time_dim * lat_dim * lon_dim).reshape(time_dim, lat_dim, lon_dim)
    event_times = np.zeros(time_dim).reshape(time_dim)
    
    j = 0
    
    for iday in range(len(ERA5_ind_with_CYGdays)):
        start_ind  = ERA5_ind_with_CYGdays[iday]
        
        temp_var = var_in[start_ind:start_ind+24, :, :]
        
        #Used temp_times to make sure that the days were being selected correctly
        temp_times = ERA5_times[start_ind:start_ind+24]
        
        #print(temp_times[0], temp_times[-1])
        
        #Fill a blank array that will consist of all the days that qualify for a certain criteria
        event_var[j:j+24, :, :] = temp_var
        event_times[j:j+24]     = temp_times
        j += 24

    #Then average over every 24th array element to get average diurnal cycle for certain days that qualify as pre-defined event
    mean_daily_var = np.zeros(24 * var_in.shape[2] * var_in.shape[1]).reshape(24, var_in.shape[2], var_in.shape[1])
    
    #Find average for each hour of the day
    for ihr in (range(24)):

        var_ihr   = event_var[ihr::24,:,:]
        test_time = event_times[ihr::24]
        
        #find the average of that hour over all days
        mean_var  = np.mean(var_ihr, axis = 0)
        
        #Rotate the array so that it's lon x lat instead of lat x lon and make a daily array
        mean_daily_var[ihr,:,:] = np.transpose(mean_var)
        
    return mean_daily_var

