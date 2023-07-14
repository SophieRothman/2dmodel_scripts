# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:41:44 2022

@author: gjg882
"""

#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from matplotlib.gridspec import  GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import xarray as xr

import pandas as pd

from os import chdir
from pathlib import Path

import numpy as np
from landlab import RasterModelGrid
from landlab import load_params
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)

#%%

#Label for output filenames - model type, runtime, grid size, bedrock erdobility ratio, misc. 

file_id = 'Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers'
ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers_ds_final.nc'

#file_id = 'dtch_1200kyr_nx50_Kr5'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/dtch_driver_master/dtch_1200kyr_nx50_Kr5_fsc_ero/dtch_1200kyr_nx50_Kr5_fsc_ero_ds_final.nc'


#file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs1'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs1/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs1_ds_final.nc'

#file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs3'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_ds_final.nc'

#file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs5'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs5/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs5_ds_final.nc'



#load in model output

ds = xr.open_dataset(ds_file) 

#load in parameters
ds_attrs = ds.attrs

#Select model time (in years) to plot
plot_time = 1200000
#plot_time = ds.attrs['space_runtime']

rock = ds.rock_type__id.sel(time=plot_time)


#%%

#Load parameters from attributes dictionary into variables

model_name = ds_attrs['model_name']

K_soft = ds_attrs['K_soft']
K_ratio = ds_attrs['K_ratio']

#Erodibility
K_hard = K_soft / K_ratio

K_hard_label = "{:.2e}".format(K_hard)

layer_thickness = ds_attrs['layer_thickness']

#if model_name == 'Detachment Limited':
   # uplift = ds_attrs['fsc_uplift']
   # fsc_runtime = ds_attrs['fsc_runtime']
   # fsc_runtime_kyr = int(fsc_runtime / 1000) #for labeling output files, plots

#if model_name == 'mixed':
    

nx = ds_attrs['nx']
ny = ds_attrs['ny']
dx = ds_attrs['dx']

#m and n are the same for both fastscape and space
m_sp = ds_attrs['m_sp']
n_sp = ds_attrs['n_sp']

#dt = ds_attrs['fsc_dt']

theta = m_sp / n_sp

#minimum drainage area for steepness and chi finder
sf_min_DA = 500
cf_min_DA = dx**2 #ratio used in Nicole's tutorial notebook

#define colormap for plotting
lith_cmap=plt.cm.get_cmap('Paired', 10)


#%%
#Define function to run channel profiler, calculate steepness, and calculate chi for a given time in xarray dataset

def channel_calcs(ds, plot_time, sf_min_DA, cf_min_DA):
    
    ds_attrs = ds.attrs
    
    nx = ds_attrs['nx']
    ny = ds_attrs['ny']
    dx = ds_attrs['dx']

    #m and n are the same for both fastscape and space
    m_sp = ds_attrs['m_sp']
    n_sp = ds_attrs['n_sp']
    
    theta = m_sp / n_sp
    
    
    #make model grid using topographic elevation at desired time
    mg = RasterModelGrid((nx, ny), dx)
    z = ds.topographic__elevation.sel(time=plot_time)
    z = mg.add_field("topographic__elevation", z, at="node")
    
    #Setting Node 0 (bottom left corner) as outlet node 
    mg.set_watershed_boundary_condition_outlet_id(0, mg.at_node['topographic__elevation'],-9999.)
    
    #Add rock type at each node to model grid
    rock = ds.rock_type__id.sel(time=plot_time)
    mg.has_field('node', 'rock_type__id')
    rock is mg.add_field('node', 'rock_type__id', rock, dtype=int)
    
    var_list = list(ds.keys())
    
    if 'bedrock__erosion' in var_list:
    
        #Add bedrock erosion term
        Er = ds.bedrock__erosion.sel(time=plot_time)
        mg.has_field('node', 'bedrock__erosion')
        Er is mg.add_field('node', 'bedrock__erosion', Er, dtype=float)
    
    if model_name == 'Mixed':
        
        #Add soil depth to model grid
        sed_depth = ds.soil__depth.sel(time=plot_time)
        mg.has_field('node', 'soil__depth')
        sed_depth is mg.add_field('node', 'soil__depth', sed_depth, dtype=float)
        
        #Add sediment erosion term
        Es = ds.sediment__erosion.sel(time=plot_time)
        mg.has_field('node', 'sediment__erosion')
        Es is mg.add_field('node', 'sediment__erosion', Es, dtype=float)

    
    #Run flow accumulator
    fa = FlowAccumulator(mg, flow_director='D8')
    fa.run_one_step()

    #Run channel profiler to find main channel nodes
    prf = ChannelProfiler(mg, main_channel_only=True,minimum_channel_threshold=sf_min_DA)
    prf.run_one_step()
    
    #Calculate Channel Steepness
    sf = SteepnessFinder(mg, reference_concavity=theta, min_drainage_area=sf_min_DA)
    sf.calculate_steepnesses()    

    #Calculate Chi
    cf = ChiFinder(mg, min_drainage_area=cf_min_DA,reference_concavity=theta,use_true_dx=True)
    cf.calculate_chi()
    
    #Get nodes that define main channel
    prf_keys = list(prf.data_structure[0])
    
    

    channel_dist = prf.data_structure[0][prf_keys[0]]['distances']
    channel_dist_ids = prf.data_structure[0][prf_keys[0]]['ids']
    channel_elev = mg.at_node["topographic__elevation"][channel_dist_ids]
    channel_chi = mg.at_node["channel__chi_index"][channel_dist_ids]
    channel_ksn = mg.at_node["channel__steepness_index"][channel_dist_ids]

    channel_x = []
    channel_y = []
    
    for node in channel_dist_ids:
    
        channel_x.append(mg.x_of_node[node])
        channel_y.append(mg.y_of_node[node])

    channel_rock_ids = mg.at_node['rock_type__id'][channel_dist_ids]
    channel_rock_ids = channel_rock_ids.astype(int)


    df = pd.DataFrame({'channel_dist': channel_dist,
                   'channel_elev': channel_elev,
                   'channel_rock_id': channel_rock_ids, 
                   'channel_ksn' : channel_ksn,
                   'channel_chi' : channel_chi,
                   'channel_x' : channel_x,
                   'channel_y' : channel_y})
    
    df["uplift"] = .001
    
    if 'bedrock__erosion' in var_list:
        channel_Er = mg.at_node['bedrock__erosion'][channel_dist_ids] 
        df['channel_Er'] = channel_Er
    
    if model_name == 'Mixed':
        
        #add sediment data for SPACE model runs
        channel_sed_depth = mg.at_node['soil__depth'][channel_dist_ids]
        df['channel_sed_depth'] = channel_sed_depth
        
        channel_Es = mg.at_node['sediment__erosion'][channel_dist_ids]  
        df['channel_Es'] = channel_Es
            
        
    return mg, df

    
#%%
#Plot the channel profile, chi, and ksn

def channel_plots(df, plot_time, model_name, save_fig):
    
    #colormap for plotting
    lith_cmap=plt.cm.get_cmap('Paired', 10)
    
    #Convert time to kyr for labels
    plot_time_kyr = int(plot_time / 1000)
    plot_time_myr = plot_time / 1000000
    
    #group dataframe by rock type
    groups = df.groupby('channel_rock_id')
    
    #make the figure
    fig = plt.figure(constrained_layout=True, figsize=(10, 8), dpi=300)

    gs = GridSpec(2, 2, figure=fig)
    
    #plot channel profile in top row
    ax1 = fig.add_subplot(gs[0, :])

    for name, group in groups:
        
        if name %2 == 0:
        
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        
    legend_elements_dtch = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10)]
    
    legend_elements_mixed = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='dimgrey', lw=2, label='Sediment Depth')]        
    

    anno1 = 'Odd Layer ID = Hard Rock'
    plt.annotate(anno1, xy=(0.1, 0.94), xycoords='axes fraction')

    ax1.legend(title="Layer ID")
    ax1.set_xlabel('Distance Upstream (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(top=600)
    

    if model_name == 'Mixed':
        #Add sediment depth to channel profile w/ secondary y-axis
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr"
        ax=ax1.twinx()
        ax.plot(df['channel_dist'], df['channel_sed_depth'], '-', color = 'dimgrey', label = 'Sediment Depth')
        ax.set_ylabel("Sediment Depth (m)")
        ax.set_ylim(top=2.25)
        #ax.legend(loc='best')
    else:
        #title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_kyr} kyr"
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr"
    
    ax1.set_title(title_string)
    
    #Plot Chi in lower left corner
    ax2 = fig.add_subplot(gs[1, :-1])
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax2.plot(group.channel_chi, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax2.plot(group.channel_chi, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
            

    title_string = f"Main Channel Chi Index"
    #ax2.legend(title="Layer ID", loc=4)
    ax2.set_xlabel('Chi Index')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title(title_string)
    #ax2.set_ylim(top=600)
    
    #Plot Ksn in lower right corner
    ax3 = fig.add_subplot(gs[1:, -1])
    
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax3.plot(group.channel_dist, group.channel_ksn, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax3.plot(group.channel_dist, group.channel_ksn, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
   

    title_string = "Main Channel Steepness Index"
    #ax3.legend(title="Layer ID", loc=1, ncol=2)
    ax3.set_xlabel('Distance Upstream (m)')
    ax3.set_ylabel('Ksn')
    ax3.set_title(title_string)
    ax3.set_ylim(top=400)
    
    if save_fig == True: 
        file_string = "MainChannelCalcs" + file_id + ".svg"
        fig.savefig(file_string);
        

#%%
#plot erosion/entrainment rates along channel profile

def plot_channel_ero_rate (df, plot_time, model_name, save_fig, file_id):
    
    
    #colormap for plotting
    lith_cmap=plt.cm.get_cmap('Paired', 10)
    
    #Convert time to kyr for labels
    plot_time_kyr = int(plot_time / 1000)
    plot_time_myr = plot_time / 1000000
    
    #group dataframe by rock type
    groups = df.groupby('channel_rock_id')
    
    #make the figure
    fig = plt.figure(constrained_layout=True, figsize=(10, 4), dpi=300)

    
    #plot channel profile in top row
    ax1 = fig.add_subplot()
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        
    legend_elements_dtch = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='b', lw=2, label='Bedrock Erosion Rate'),
                   Line2D([0], [0], color='dimgrey', linestyle='dashed', lw=2, label='Uplift Rate')]
    
    legend_elements_mixed = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='b', lw=2, label='Bedrock Erosion Rate'),
                   Line2D([0], [0], color='r', lw=2, label='Sediment Entrainment Rate'),
                   Line2D([0], [0], color='dimgrey', linestyle='dashed', lw=2, label='Uplift Rate')]

    #ax1.legend(title="Layer ID")

    ax1.set_xlabel('Distance Upstream (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(top=600)
    
    ax=ax1.twinx()
    ax.set_ylim(top=.013)
    
    if model_name == 'Detachment Limited':
        
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr"
        
        ax1.legend(handles=legend_elements_dtch, loc='best')
        
       
        ax.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Erosion Rate')
        ax.set_ylabel("Erosion/Entrainment/Uplift Rate (m/yr)")
        
        
        ax.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')

    
    if model_name == 'Mixed':
        
        v_s_round = np.round(ds.attrs['v_s'])
        
        #title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr, V={v_s_round} m/yr"
        title_string = file_id
        
        ax1.legend(handles=legend_elements_mixed, loc='best')
        
        #Add sediment depth to channel profile w/ secondary y-axis
        #ax=ax1.twinx()
        ax.plot(df['channel_dist'], df['channel_Es'], '-', color = 'red', label = 'Sediment Entrainment Rate')
        
        ax.set_ylabel("Erosion Rate (m/yr)")
        #ax.set_ylim(top=2.25)
        #ax.legend(loc='best')
        
        ax.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')
        
        ax.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Erosion Rate')
        ax.set_ylabel("Erosion Rate (m/yr)")

    ax1.set_title(title_string)
    
    if save_fig == True:
        file_string = 'PRF_ero_rates' + file_id + '.svg'
        fig.savefig(file_string);

#%%


def plot_channel_sed (df, plot_time, model_name, save_fig, file_id):
    
    #colormap for plotting
    lith_cmap=plt.cm.get_cmap('Paired', 10)
    
    #Convert time to kyr for labels
    plot_time_kyr = int(plot_time / 1000)
    plot_time_myr = plot_time / 1000000
    
    #group dataframe by rock type
    groups = df.groupby('channel_rock_id')
    
    #make the figure
    fig = plt.figure(constrained_layout=True, figsize=(10, 4), dpi=300)

    
    #plot channel profile in top row
    ax1 = fig.add_subplot()
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        
    legend_elements_dtch = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10)]
    
    legend_elements_mixed = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='dimgrey', lw=2, label='Sediment Depth')]

    #ax1.legend(title="Layer ID")

    ax1.set_xlabel('Distance Upstream (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(top=450)
    
    
    if model_name == 'Detachment Limited':
        
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr"
        
        
        ax1.legend(handles=legend_elements_dtch, loc='best')
        
        '''
        
        ax=ax1.twinx()
        
       
        ax.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Erosion Rate')
        ax.set_ylabel("Erosion Rate (m/yr)")
        
        ax.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')
        '''
    
    if model_name == 'Mixed':
        
        v_s_round = np.around(ds.attrs['v_s'])
        
        #title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_kyr} kyr, V={v_s_round} m/yr"
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_kyr} kyr, V={v_s_round} m/yr"
        
        ax1.legend(handles=legend_elements_mixed, loc='best')
        
        #Add sediment depth to channel profile w/ secondary y-axis
        ax=ax1.twinx()
        ax.plot(df['channel_dist'], df['channel_sed_depth'], '-', color = 'dimgrey', label = 'Sediment Depth')
        ax.set_ylabel("Sediment Depth (m)")
        ax.set_ylim(top=2.25)
        #ax.legend(loc='best')
        
    
    ax1.set_title(file_id)
    
    if save_fig == True:
        file_string = 'PRF_sed_depth' + file_id + '.svg'
        fig.savefig(file_string);



#%%
plot_time = 1200000


mg, df = channel_calcs(ds, plot_time, sf_min_DA, cf_min_DA)

#%%

#channel_plots(df, plot_time, model_name, save_fig=False)



#%%


#plot_channel_ero_rate(df, plot_time, model_name, save_fig=False, file_id=file_id)