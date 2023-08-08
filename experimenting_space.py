# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:59:31 2023

@author: srothman
"""

## Import Numpy and Matplotlib packages
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt  # For plotting results; optional

## Import Landlab components
# Flow routing and depression handling
from landlab.components import PriorityFloodFlowRouter, FastscapeEroder, ExponentialWeathererIntegrated, TaylorNonLinearDiffuser # BedrockLandslider

# SPACE model
from landlab.components import SpaceLargeScaleEroder, ChiFinder # SPACE model

# BedrockLandslider model
from landlab.components import BedrockLandslider, ChannelProfiler, ExponentialWeathererIntegrated  # BedrockLandslider model

## Import Landlab utilities

from landlab import RasterModelGrid # Grid utility
from landlab import imshowhs_grid, imshow_grid  # For plotting results; optional
#%%
# Set grid parameters
nr = 10
nc = 10
dx = 50

timestep = 500  # years

# Set elapsed time to zero
elapsed_time = 0.0  # years

# Set timestep count to zero
count = 0

# Set model run time
run_time = 1.8e7 # years

# Array to save sediment flux values
sed_flux = np.zeros(int(run_time // timestep))

# Uplift rate in m/yr
U = (4e-4)#-4)
cmap = copy.copy(mpl.cm.get_cmap("terrain"))



max_rate=3e-5  #max soil production rate
dec_depth=.34   #depth of soil production
D=(1e-3)  #Transport rate
scrit=.6


Ks_min=float(1.25e-5) 
Ks_max=float(5e-5)
Ks_mid=float(1.5e-4)   #K for sediment for baseline model

K_min = float(0.5e-6)
K_max = float(2e-6)
K_mid = float(1e-6)    #K for rock for baseline model
max_slp = 0.05
#%%
# track sediment flux at the node adjacent to the outlet at lower-left
node_next_to_outlet = nc + 1

# Instantiate model grid
mg1 = RasterModelGrid((nr, nc),dx)

# add field ’topographic elevation’ to the grid
z1=mg1.add_zeros("node", "topographic__elevation")
K=mg1.add_zeros("node", "bedrock_erodibility")
mg1.at_node["bedrock_erodibility"][:]=K_mid

# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# Create initial model topography:

# add topographic roughness
random_noise = (
    np.random.rand(len(mg1.node_y)) / 1000.0
)  # impose topography values on model grid
mg1["node"]["topographic__elevation"] += random_noise



#%% make it a one node exit condition
mg1.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
mg1.set_watershed_boundary_condition_outlet_id(0, mg1.at_node['topographic__elevation'], -9999.0)

#%% instantiate components
fr=PriorityFloodFlowRouter(mg1, surface="topographic__elevation", flow_metric="D8", runoff_rate=None)
fsc1=FastscapeEroder(mg1, K_sp=K, m_sp=.45, n_sp=1 )


#%% burn in initial topography

for i in range(0 , 15000):#int(run_time/timestep)):
    z1[mg1.core_nodes] += U * timestep #uplfit
    fr.run_one_step()          #route flow
    fsc1.run_one_step(timestep) #erode
    #sp1.run_one_step(timestep)
    #mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
    #ex1.run_one_step(timestep) #weather 
    #mg1.at_node['bedrock__elevation']=-mg1.at_node['soil_production__dt_weathered_depth'] #change bedrock elevation from weathering
    #mg1.at_node['soil__depth']=+mg1.at_node['soil_production__dt_produced_depth'] #change soil depth from weathering
#%%
# add field 'soil__depth' to the grid
mg1.add_zeros("node", "soil__depth")


# Set 2 m of initial soil depth at core nodes
mg1.at_node["soil__depth"][mg1.core_nodes] = 0  # meters


# Add field 'bedrock__elevation' to the grid
bdr_z=mg1.add_zeros("bedrock__elevation", at="node")

mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]

mg1.at_node["topographic__elevation"][:] += mg1.at_node["soil__depth"]
ex1=ExponentialWeathererIntegrated(mg1, soil_production__maximum_rate=max_rate, 
                                   soil_production__decay_depth=dec_depth,
                                   soil_production__expansion_factor=1.5)
sp1=SpaceLargeScaleEroder(mg1, K_sed=Ks_mid, K_br=K_mid, F_f=0, phi=.6, 
                          H_star=.1, v_s=1, m_sp=.45, n_sp=1 )
tnld1=TaylorNonLinearDiffuser(mg1, linear_diffusivity=D, slope_crit=scrit, nterms=2)
#%%check it out
#imshowhs_grid(mg1, 'topographic__elevation')
imshow_grid(mg1, "topographic__elevation")
#

#%%
da1=mg1.at_node["drainage_area"].copy()
slp1 = mg1.at_node["topographic__steepest_slope"].copy()
stpow1=da1**0.45 * slp1*mg1.at_node["bedrock_erodibility"]

#%% print some figs
plt.figure()
imshow_grid(mg1, 'topographic__elevation', plot_name="elev no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
plt.show()
plt.figure()
imshow_grid(mg1, 'soil__depth', plot_name="soil depth",  cmap = cmap, colorbar_label="Soil Depth (m)")
plt.show()
plt.figure()
imshow_grid(mg1,
    stpow1,
    plot_name="stream power",
    #vmin=1e-5,
    #vmax=120,
    cmap='winter') 
plt.show()
plt.figure()
imshow_grid(mg1,
    slp1,
    plot_name="slope",
    #vmin=1e-5,
    #vmax=120,
    cmap='winter') 
plt.show()
#%%
cp1 = ChannelProfiler(mg1, number_of_watersheds=1 ,main_channel_only=True)
cp1.run_one_step()
cp1.plot_profiles()
cf1 = ChiFinder(
     mg1,
     use_true_dx=True,
     reference_concavity=0.45,
     reference_area=mg1.at_node['drainage_area'].max(),
     clobber=True)
cf1.calculate_chi()
#%%
nowfprf={}
for i, outlet_id in enumerate(cp1.data_structure):
    nowfprf['dist'+str(i)]=np.zeros(0)
    #slowelev1=np.zeros(0)
    for j, segment_id in enumerate(cp1.data_structure[outlet_id]):
        totprof = cp1.data_structure[outlet_id][segment_id]
        totprofid=totprof["ids"]
        nowfprf['chi'+str(i)]=mg1.at_node['channel__chi_index'][totprofid]
        nowfprf['elev'+str(i)]=mg1.at_node['topographic__elevation'][totprofid]
        nowfprf['br_elev'+str(i)]=mg1.at_node['bedrock__elevation'][totprofid]
        nowfprf['soil_d'+str(i)]=mg1.at_node['soil__depth'][totprofid]
        dist=cp1.data_structure[outlet_id][segment_id]["distances"]
        nowfprf['da'+str(i)]=mg1.at_node['drainage_area'][totprofid]
        nowfprf['dist'+str(i)] = np.concatenate((nowfprf['dist'+str(i)], dist))
    Nnodes=len(nowfprf['elev'+str(i)])
    nowfprf['slope'+str(i)]=np.zeros(Nnodes)
    nowfprf['slope'+str(i)][1:(Nnodes)] = (1/dx)*(nowfprf['elev'+str(i)][1:(Nnodes)] - nowfprf['elev'+str(i)][0:(Nnodes-1)]) 
#%% print out a bunch of metrics
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

plt.axes(ax[0, 0])

ax[0, 0].set_title("topographic elevation")
plt.plot(nowfprf['dist0'], nowfprf['elev0'])
#ax[0, 0].set_ylim([0, 800])
plt.ylabel('Elevation (m)')
#plt.ylim([0,175])
plt.xlabel(' Distance Along Profile (m)')

plt.axes(ax[0,1])
plt.plot(nowfprf['dist0'], nowfprf['slope0'])
ax[0,1].set_title("slope")
plt.xlabel(' Distance Along Profile (m)')
plt.ylabel('Slope')

plt.axes(ax[1,1])
plt.plot(nowfprf['dist0'], nowfprf['br_elev0'])
ax[1,1].set_title("bedrock elevation")
plt.xlabel(' Distance Along Profile (m)')

#plt.ylim([0,175])
plt.ylabel('Bedrock_elev')

plt.axes(ax[1,0])
plt.plot(nowfprf['dist0'], nowfprf['soil_d0'])
ax[1,0].set_title("soil depth")
plt.xlabel(' Distance Along Profile (m)')
plt.ylabel('Soil Depth')
#%% try out new component
#%% burn in initial topography

for i in range(0 , 500):#int(run_time/timestep)):
    z1[mg1.core_nodes] += U * timestep #uplfit
    fr.run_one_step()          #route flow
    #fsc1.run_one_step(timestep) #erode
    sp1.run_one_step(timestep)
    #mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
    #ex1.run_one_step(timestep) #weather 
    #mg1.at_node['bedrock__elevation']=mg1.at_node['bedrock__elevation']-mg1.at_node['soil_production__dt_weathered_depth'] #change bedrock elevation from weathering
    #mg1.at_node['soil__depth']=mg1.at_node['soil__depth']+mg1.at_node['soil_production__dt_produced_depth'] #change soil depth from weathering
    tnld1.run_one_step(timestep)
    dqdx=mg1.calc_flux_div_at_node('soil__flux')
    mg1.at_node['topographic__elevation'][mg1.core_nodes]+=dqdx[mg1.core_nodes]*timestep
    mg1.at_node['soil__depth'][mg1.core_nodes]-=dqdx[mg1.core_nodes]*timestep


#%%