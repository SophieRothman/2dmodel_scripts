# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:00:56 2023

@author: srothman
"""

#single model run
#Imports
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt  # For plotting results; optional

## Import Landlab components
# Flow routing and depression handling
from landlab.components import PriorityFloodFlowRouter 

#add bedrock weathering
from landlab.components import ExponentialWeathererIntegrated

# SPACE model
from landlab.components import SpaceLargeScaleEroder  # SPACE model

# BedrockLandslider model
from landlab.components import BedrockLandslider, ChannelProfiler, TaylorNonLinearDiffuser # BedrockLandslider model

## Import Landlab utilities

from landlab import RasterModelGrid # Grid utility
from landlab import imshowhs_grid, imshow_grid  # For plotting results; optional
from landlab.components import ChiFinder, FastscapeEroder
#parameter setup
# Set grid parameters
nr = 90
nc = 70
dx = 50
timestep = 1000  # years

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
Usetup=4e-7
cmap = copy.copy(mpl.cm.get_cmap("terrain"))



max_rate=3e-3  #max soil production rate
dec_depth=.34   #depth of soil production
D=(4e-3)  #Transport rate
scrit=.6


Ks_min=float(1.25e-5) 
Ks_max=float(5e-5)
Ks_mid=float(1e-4)   #K for sediment for baseline model


K_min = float(0.5e-6)
K_max = float(2e-6)
K_mid = float(2e-5)#6)    #K for rock for baseline model
m=0.45
n=1
max_slp = 0.05  #critical slope
#%%
#set up the grid
node_next_to_outlet = nc + 1

# Instantiate model grid
mg1 = RasterModelGrid((nr, nc),dx)

# add field ’topographic elevation’ to the grid
z1=mg1.add_zeros("node", "topographic__elevation")

# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# Create initial model topography:

# add topographic roughness
random_noise = (
    np.random.rand(len(mg1.node_y)) / 1000.0
)  # impose topography values on model grid
mg1["node"]["topographic__elevation"] += random_noise

K1 = mg1.add_ones("bedrock_erodibility", at="node")
K1[:] = K_mid

Ks=mg1.add_ones("sediment_erodibility", at="node")
Ks[:]=Ks_mid
#%%
# Open all model boundary edges
mg1.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
mg1.set_watershed_boundary_condition_outlet_id(0, mg1.at_node['topographic__elevation'], -9999.0)



#%% Instantiate the first two components
fr1=PriorityFloodFlowRouter(mg1, surface="topographic__elevation", flow_metric="D8", runoff_rate=None)
fsc1=FastscapeEroder(mg1, K_sp=K_mid, m_sp=.45, n_sp=1 )

#%% burn in an initial topography
for i in range(0 , 1000):#int(run_time/timestep)):
    z1[mg1.core_nodes] += U * timestep #uplfit
    fr1.run_one_step()          #route flow
    fsc1.run_one_step(timestep) #erode
    #sp1.run_one_step(timestep)
    #mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
    #ex1.run_one_step(timestep) #weather 
    #mg1.at_node['bedrock__elevation']=mg1.at_node['bedrock__elevation']-mg1.at_node['soil_production__dt_weathered_depth'] #change bedrock elevation from weathering
    #mg1.at_node['soil__depth']=mg1.at_node['soil__depth']+mg1.at_node['soil_production__dt_produced_depth'] #change soil depth from weathering
    #tnld1.run_one_step(timestep)
    #dqdx=mg1.calc_flux_div_at_node('soil__flux')
    #mg1.at_node['topographic__elevation'][mg1.core_nodes]+=dqdx[mg1.core_nodes]*timestep
    #mg1.at_node['soil__depth'][mg1.core_nodes]-=dqdx[mg1.core_nodes]*timestep
#%%

#np.savetxt("fastscape_ss__u4e4k1e5.txt", z1)
zmaybe=np.loadtxt("fastscape_ss__u4e4k1e5.txt")
#to open it


#%%
# Sum 'soil__depth' and 'bedrock__elevation'
bdr_z=mg1.add_zeros("bedrock__elevation", at="node", clobber=True)

# to yield 'topographic elevation'
mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
# add field 'soil__depth' to the grid
mg1.add_zeros("node", "soil__depth", clobber=True)

mg1.at_node["topographic__elevation"][:] += mg1.at_node["soil__depth"]




# Set 2 m of initial soil depth at core nodes


# Add field 'bedrock__elevation' to the grid


# ex1=ExponentialWeathererIntegrated(mg1, soil_production__maximum_rate=max_rate, 
#                                    soil_production__decay_depth=dec_depth,
#                                    soil_production__expansion_factor=1.5)
sp1=SpaceLargeScaleEroder(mg1, K_sed=Ks_mid, K_br=K_mid, phi=.6, 
                          H_star=.1, v_s=0.5, m_sp=.45, n_sp=1 )
stpow1=mg1.add_zeros("node", "stream_power", clobber=True)

da1=mg1.at_node["drainage_area"].copy()

slp1 = mg1.at_node["topographic__steepest_slope"].copy()


stpow1=da1**0.5 * slp1**n*mg1.at_node["bedrock_erodibility"]
mg1.set_watershed_boundary_condition_outlet_id(0, mg1.at_node['topographic__elevation'], 0)
mg1.set_watershed_boundary_condition_outlet_id(0, mg1.at_node['bedrock__elevation'], 0)

#%%
mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]

#%%
plt.figure()
imshow_grid(mg1, 'topographic__elevation', plot_name="elev no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
plt.show()
plt.figure()
imshow_grid(mg1, 'bedrock__elevation', plot_name="elev no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
plt.show()

#%%
tnld1=TaylorNonLinearDiffuser(mg1, linear_diffusivity=D, slope_crit=scrit, nterms=2)

elapsed_time=0
timestep=500
imshow_grid(mg1, "topographic__elevation")
#while elapsed_time<int(run_time/3):#int(run_time/timestep)):#int(run_time/timestep)):
for i in range(0,1000):
    elapsed_time+=timestep
    mg1.at_node['bedrock__elevation'][mg1.core_nodes] += U * timestep #uplfit

    mg1.at_node["topographic__elevation"][mg1.core_nodes][:] = (
        mg1.at_node["bedrock__elevation"][mg1.core_nodes] + mg1.at_node["soil__depth"][mg1.core_nodes]
    )
    fr1.run_one_step()          #route flow
    #fsc1.run_one_step(timestep) #erode

    #mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
    #ex1.run_one_step(timestep) #weather 
    #mg1.at_node['bedrock__elevation']=mg1.at_node['bedrock__elevation']-mg1.at_node['soil_production__dt_weathered_depth'] #change bedrock elevation from weathering
    #mg1.at_node['soil__depth']=mg1.at_node['soil__depth']+mg1.at_node['soil_production__dt_produced_depth'] #change soil depth from weathering
    
    tnld1.run_one_step(timestep)
    dqdx=mg1.calc_flux_div_at_node('soil__flux')
    mg1.at_node['topographic__elevation'][mg1.core_nodes]+=dqdx[mg1.core_nodes]*timestep
    #mg1.at_node['soil__depth'][dqdx<0]-=dqdx[dqdx<0]*timestep
    mg1.at_node['soil__depth'][mg1.core_nodes]-=dqdx[mg1.core_nodes]*timestep
    mask=mg1.at_node['soil__depth']<0
    mg1.at_node['bedrock__elevation'][mask]+=np.copy(mg1.at_node['soil__depth'][mask])
    mg1.at_node['soil__depth'][mask]=0
    sp1.run_one_step(timestep)

    

    if np.mod(elapsed_time, 5e5)==0:
        da1=mg1.at_node["drainage_area"].copy()
        slp1 = mg1.at_node["topographic__steepest_slope"].copy()
        stpow1=da1**0.5 * slp1**n*mg1.at_node["bedrock_erodibility"]
        thresh=0
        print('%.2f of model run completed' %(elapsed_time/run_time))
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
print(mg1.at_node['bedrock__elevation'][300:305])
print(mg1.at_node['topographic__elevation'][300:305])
print(mg1.at_node['soil__depth'][300:305])

#%%
#Producing images of the rasters
elev_max = z1.max() #max(z1.max(), z2.max(), z3.max())
da1=mg1.at_node["drainage_area"].copy()
slp1 = mg1.at_node["topographic__steepest_slope"].copy()
stpow1=da1**0.5 * slp1**n*mg1.at_node["bedrock_erodibility"]

plt.figure()
imshow_grid(mg1,
            z1,
            plot_name="baseline",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')
plt.scatter(mg1.x_of_node[(slp1>max_slp) & (da1>500000)], mg1.y_of_node[(slp1>max_slp) & (da1>500000)], c='red',marker='.')

plt.figure()
imshow_grid(mg1, 'bedrock__elevation', plot_name="elev no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
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

#%% #Channel profiles

numchannel=1;

cp1 = ChannelProfiler(mg1, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
cp1.run_one_step()

# cp2 = ChannelProfiler(mg2, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
# cp2.run_one_step()

# cp3 = ChannelProfiler(mg3, number_of_watersheds=numchannel,  minimum_channel_threshold=500000 , main_channel_only=True)
# cp3.run_one_step()
# slp3 = mg3.at_node["topographic__steepest_slope"].copy()
slp1 = mg1.at_node["topographic__steepest_slope"].copy()
# slp2 = mg2.at_node["topographic__steepest_slope"].copy()

# da3 = mg3.at_node["drainage_area"].copy()
da1 = mg1.at_node["drainage_area"].copy()
# da2 = mg2.at_node["drainage_area"].copy()


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8))

plt.axes(ax[0, 0])
cp1.plot_profiles()
ax[0, 0].set_title("baseline")
ax[0, 0].set_ylim([0, 800])
plt.ylabel('Elevation (m)')
plt.xlabel(' Distance Along Profile (m)')

# plt.axes(ax[0, 1])
# cp2.plot_profiles()
# ax[0, 1].set_title("with increase in K with slope")
# ax[0, 1].set_ylim([0, 800])
# plt.ylabel('Elevation (m)')
# plt.xlabel(' Distance Along Profile (m)')

# plt.axes(ax[0, 2])
# cp3.plot_profiles()
# ax[0, 2].set_title("with decrease in K with slope")
# ax[0, 2].set_ylim([0, 800])
# plt.ylabel('Elevation (m)')
# plt.xlabel(' Distance Along Profile (m)')
plt.axes(ax[1, 0])
cp1.plot_profiles_in_map_view()
plt.scatter(mg1.x_of_node[(slp1>max_slp) & (da1>500000)], mg1.y_of_node[(slp1>max_slp) & (da1>500000)],c='red', marker='.')

plt.tight_layout()

# plt.axes(ax[1, 1])
# cp2.plot_profiles_in_map_view()
# plt.scatter(mg1.x_of_node[(slp2>max_slp) & (da2>500000)], mg1.y_of_node[(slp2>max_slp) & (da2>500000)], c='red',marker='.')


# plt.axes(ax[1, 2])
# cp3.plot_profiles_in_map_view()
# plt.scatter(mg1.x_of_node[(slp3>max_slp) & (da3>500000)], mg1.y_of_node[(slp3>max_slp) & (da3>500000)], c='red', marker='.')
#%% Chi Calaculations
cf1 = ChiFinder(
     mg1,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=mg1.at_node['drainage_area'].max(),
     clobber=True)

# cf2 = ChiFinder(
#      mg2,
#      min_drainage_area=500000.,
#      use_true_dx=True,
#      reference_concavity=0.5,
#      reference_area=mg2.at_node['drainage_area'].max(),
#      clobber=True)

# cf3 = ChiFinder(
#      mg3,
#      min_drainage_area=500000.,
#      use_true_dx=True,
#      reference_concavity=0.5,
#      reference_area=mg3.at_node['drainage_area'].max(),
#      clobber=True)

cf1.calculate_chi()
# cf2.calculate_chi()
# cf3.calculate_chi()

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

#%% Slope area total

# plt.loglog(mg3.at_node["drainage_area"],
#           mg3.at_node["topographic__steepest_slope"],
#           ".",
#           color='blue',
#           label="with decrease in K with slope")
plt.loglog(mg1.at_node["drainage_area"],
           mg1.at_node["topographic__steepest_slope"],
           ".",
           color="grey",
           label="baseline")
# plt.loglog(mg2.at_node["drainage_area"],
#            mg2.at_node["topographic__steepest_slope"],
#            ".",
#            color='orange',
#            label="with increase in K with slope")

#plt.xlim([500000, 2e7])
plt.legend()
plt.ylabel("Slope")
plt.xlabel("Area")