#make new file
## Import Numpy and Matplotlib packages
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt  # For plotting results; optional

## Import Landlab components
# Flow routing and depression handling
from landlab.components import PriorityFloodFlowRouter 

# SPACE model
from landlab.components import SpaceLargeScaleEroder  # SPACE model

# BedrockLandslider model
from landlab.components import BedrockLandslider, ChannelProfiler  # BedrockLandslider model

## Import Landlab utilities

from landlab import RasterModelGrid # Grid utility
from landlab import imshowhs_grid, imshow_grid  # For plotting results; optional
from landlab.components import ChiFinder
#%%
# Set grid parameters
nr = 60
vnc = 50
dx = 75

# track sediment flux at the node adjacent to the outlet at lower-left
node_next_to_outlet = nc + 1

# Instantiate model grid
mg1 = RasterModelGrid((nr, nc),dx)
mg2 = RasterModelGrid((nr, nc),dx)
mg3 = RasterModelGrid((nr, nc),dx)
# add field ’topographic elevation’ to the grid
z1=mg1.add_zeros("node", "topographic__elevation")
z2=mg2.add_zeros("node", "topographic__elevation")
z3=mg3.add_zeros("node", "topographic__elevation")
# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# Create initial model topography:

# add topographic roughness
random_noise = (
    np.random.rand(len(mg1.node_y)) / 1000.0
)  # impose topography values on model grid
mg1["node"]["topographic__elevation"] += random_noise
mg2["node"]["topographic__elevation"] += random_noise
mg3["node"]["topographic__elevation"] += random_noise

# add field 'soil__depth' to the grid
mg1.add_zeros("node", "soil__depth")
mg2.add_zeros("node", "soil__depth")
mg3.add_zeros("node", "soil__depth")

# Set 2 m of initial soil depth at core nodes
mg1.at_node["soil__depth"][mg1.core_nodes] = 1.0  # meters
mg2.at_node["soil__depth"][mg2.core_nodes] = 1.0  # meters
mg3.at_node["soil__depth"][mg3.core_nodes] = 1.0  # meters

# Add field 'bedrock__elevation' to the grid
mg1.add_zeros("bedrock__elevation", at="node")
mg2.add_zeros("bedrock__elevation", at="node")

mg3.add_zeros("bedrock__elevation", at="node")
#%%

# Open all model boundary edges
mg1.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
mg1.set_watershed_boundary_condition_outlet_id(0, mg1.at_node['topographic__elevation'], -9999.0)

mg3.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
mg3.set_watershed_boundary_condition_outlet_id(0, mg3.at_node['topographic__elevation'], -9999.0)


mg2.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
mg2.set_watershed_boundary_condition_outlet_id(0, mg2.at_node['topographic__elevation'], -9999.0)



# Sum 'soil__depth' and 'bedrock__elevation'
# to yield 'topographic elevation'
mg1.at_node["bedrock__elevation"][:] = mg1.at_node["topographic__elevation"]
mg2.at_node["bedrock__elevation"][:] = mg2.at_node["topographic__elevation"]
mg3.at_node["bedrock__elevation"][:] = mg3.at_node["topographic__elevation"]

mg1.at_node["topographic__elevation"][:] += mg1.at_node["soil__depth"]
mg2.at_node["topographic__elevation"][:] += mg2.at_node["soil__depth"]
mg3.at_node["topographic__elevation"][:] += mg3.at_node["soil__depth"]

#%%
# Instantiate flow router
fr1 = PriorityFloodFlowRouter(mg1, flow_metric="D8", suppress_out= True)
fr2 = PriorityFloodFlowRouter(mg2, flow_metric="D8", suppress_out= True)
fr3 = PriorityFloodFlowRouter(mg3, flow_metric="D8", suppress_out= True)
#Space variables for fast waterfalls
# range of K and max slope for slp-K linear scaling
Ks_min=float(0.5e-4)*10
Ks_max=float(5e-4)*10
Ks_mid=float(2e-4)*10

K_min = float(0.5e-5)
K_max = float(2e-5)
K_mid = float(1e-5)
max_slp = 0.05
min_slp = 0.02
# Instantiate SPACE model with chosen parameters
K1 = mg1.add_ones("bedrock_erodibility", at="node", clobber=True)
K2 = mg2.add_ones("bedrock_erodibility", at="node", clobber=True)
K3 = mg3.add_ones("bedrock_erodibility", at="node", clobber=True)
K1[:] = K_mid
K2[:] = K_mid
K3[:] = K_mid

K_sed1 = mg1.add_ones("sed_erodibility", at="node", clobber=True)
K_sed2 = mg2.add_ones("sed_erodibility", at="node", clobber=True)
K_sed3 = mg3.add_ones("sed_erodibility", at="node", clobber=True)

K_sed1[:] = Ks_mid
K_sed2[:] = Ks_mid
K_sed3[:] = Ks_mid


sp1 = SpaceLargeScaleEroder(
    mg1,
    K_sed=K_sed1, #7.5e-5,
    K_br=K1,#"bedrock_erodibility",
    F_f=0.0,
    phi=0.0,
    H_star=1.0,
    v_s=0,#1,
    m_sp=0.5,
    n_sp=1.0,
    sp_crit_sed=0,
    sp_crit_br=0,
)

sp2 = SpaceLargeScaleEroder(
    mg2,
    K_sed=K_sed2,
    K_br=K2,
    F_f=0.0,
    phi=0.0,
    H_star=1.0,
    v_s=0,#1,
    m_sp=0.5,
    n_sp=1.0,
    sp_crit_sed=0,
    sp_crit_br=0,
)

sp3 = SpaceLargeScaleEroder(
    mg3,
    K_sed=K_sed3,
    K_br=K3,
    F_f=0.0,
    phi=0.0,
    H_star=1.0,
    v_s=0,#1,
    m_sp=0.5,
    n_sp=1.0,
    sp_crit_sed=0,
    sp_crit_br=0,
)


#%%%
m=0.5
n=1
da1=mg1.at_node["drainage_area"].copy()
da2=mg2.at_node["drainage_area"].copy()
da3=mg3.at_node["drainage_area"].copy()

slp1 = mg1.at_node["topographic__steepest_slope"].copy()
slp2 = mg2.at_node["topographic__steepest_slope"].copy()
slp3 = mg3.at_node["topographic__steepest_slope"].copy()

stpow1=mg1.add_zeros("node", "stream_power")
stpow2=mg2.add_zeros("node", "stream_power")
stpow3=mg3.add_zeros("node", "stream_power")

stpow1=da1**0.5 * slp1**n
stpow2=da2**0.5 * slp2**n
stpow3=da3**0.5 * slp3**n

plt.figure()
imshow_grid(mg1,
            da1,
            plot_name="with slow waterfall rule",
            #vmin=60,
            #vmax=120,
            cmap='winter') 


#%%
# Set model timestep
timestep = 1e3  # years

# Set elapsed time to zero
elapsed_time = 0.0  # years

# Set timestep count to zero
count = 0

# Set model run time
run_time = 1.8e6 # years

# Array to save sediment flux values
sed_flux = np.zeros(int(run_time // timestep))

# Uplift rate in m/yr
U = 1e-5
cmap = copy.copy(mpl.cm.get_cmap("terrain"))
        
while elapsed_time < run_time:  # time units of years

    # Insert uplift at core nodes
    mg1.at_node["bedrock__elevation"][mg1.core_nodes] += U * timestep
    mg1.at_node["topographic__elevation"][:] = (
        mg1.at_node["bedrock__elevation"] + mg1.at_node["soil__depth"]
    )
    mg2.at_node["bedrock__elevation"][mg2.core_nodes] += U * timestep
    mg2.at_node["topographic__elevation"][:] = (
        mg2.at_node["bedrock__elevation"] + mg2.at_node["soil__depth"]
    )
    mg3.at_node["bedrock__elevation"][mg3.core_nodes] += U * timestep
    mg3.at_node["topographic__elevation"][:] = (
        mg3.at_node["bedrock__elevation"] + mg3.at_node["soil__depth"]
    )

    # Run the flow router
    fr1.run_one_step()
    fr2.run_one_step()
    fr3.run_one_step()
    
    slp1 = mg1.at_node["topographic__steepest_slope"].copy()
    slp2 = mg2.at_node["topographic__steepest_slope"].copy()
    slp3 = mg3.at_node["topographic__steepest_slope"].copy()

    da1=mg1.at_node["drainage_area"].copy()
    da2=mg2.at_node["drainage_area"].copy()
    da3=mg3.at_node["drainage_area"].copy()
    

    mg2.at_node["bedrock_erodibility"]
    mg2.at_node["bedrock_erodibility"][:] = K_mid    
    mg2.at_node["bedrock_erodibility"][(slp2>max_slp) & (da2>500000)]  = K_max


    mg3.at_node["bedrock_erodibility"][:] = K_mid
    mg3.at_node["bedrock_erodibility"][(slp3>max_slp) & (da3>500000)] = K_min
    
    mg2.at_node["sed_erodibility"][:] = Ks_mid
    mg2.at_node["sed_erodibility"][(slp2>max_slp) & (da2>500000)]  = Ks_max


    mg3.at_node["sed_erodibility"][:]= Ks_mid
    mg3.at_node["sed_erodibility"][(slp3>max_slp) & (da3>500000)] = Ks_min
    

    # Run SPACE for one time step
    sp1.run_one_step(dt=timestep)
    sp2.run_one_step(dt=timestep)
    sp3.run_one_step(dt=timestep)

    # Add to value of elapsed time
    elapsed_time += timestep

    #mg.at_node["water_erodibility"][slp2>max_slp] = K_max
    #mg.at_node["water_erodibility"][slp2<=max_slp] = K_mid


    if np.mod(elapsed_time, 2e5)==0:
        print('%.2f of model run completed' %(elapsed_time/run_time))
        plt.figure()
        imshow_grid(mg1, 'topographic__elevation', plot_name="no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
        plt.show()
        plt.figure()
        imshow_grid(mg2, 'topographic__elevation', plot_name="fast waterfalls", cmap = cmap, colorbar_label="Elevation (m)")

        plt.show()
                    
        plt.figure()
        imshow_grid(mg3, 'topographic__elevation', plot_name="slow waterfalls", cmap = cmap, colorbar_label="Elevation (m)")

        plt.show()
#%%
#Producing images of the rasters
elev_max = max(z1.max(), z2.max(), z3.max())

plt.figure()
imshow_grid(mg1,
            z1,
            plot_name="baseline",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')
plt.scatter(mg1.x_of_node[(slp1>max_slp) & (da1>500000)], mg1.y_of_node[(slp1>max_slp) & (da1>500000)], c='red',marker='.')

plt.figure()
imshow_grid(mg2,
            z2,
            plot_name="with fast waterfall rule",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')
plt.scatter(mg2.x_of_node[(slp2>max_slp) & (da2>500000)], mg1.y_of_node[(slp2>max_slp) & (da2>500000)], c='red',marker='.')


plt.figure()
imshow_grid(mg3,
            z3,
            plot_name="with slow waterfall rule",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth') 
plt.scatter(mg3.x_of_node[(slp3>max_slp) & (da3>500000)], mg1.y_of_node[(slp3>max_slp) & (da3>500000)], c='red',marker='.')

        
 

#%%
#Channel profiles

numchannel=1;

cp1 = ChannelProfiler(mg1, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
cp1.run_one_step()

cp2 = ChannelProfiler(mg2, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
cp2.run_one_step()

cp3 = ChannelProfiler(mg3, number_of_watersheds=numchannel,  minimum_channel_threshold=500000 , main_channel_only=True)
cp3.run_one_step()
slp3 = mg3.at_node["topographic__steepest_slope"].copy()
slp1 = mg1.at_node["topographic__steepest_slope"].copy()
slp2 = mg2.at_node["topographic__steepest_slope"].copy()

da3 = mg3.at_node["drainage_area"].copy()
da1 = mg1.at_node["drainage_area"].copy()
da2 = mg2.at_node["drainage_area"].copy()


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8))

plt.axes(ax[0, 0])
cp1.plot_profiles()
ax[0, 0].set_title("baseline")
ax[0, 0].set_ylim([0, 800])
plt.ylabel('Elevation (m)')
plt.xlabel(' Distance Along Profile (m)')

plt.axes(ax[0, 1])
cp2.plot_profiles()
ax[0, 1].set_title("with increase in K with slope")
ax[0, 1].set_ylim([0, 800])
plt.ylabel('Elevation (m)')
plt.xlabel(' Distance Along Profile (m)')

plt.axes(ax[0, 2])
cp3.plot_profiles()
ax[0, 2].set_title("with decrease in K with slope")
ax[0, 2].set_ylim([0, 800])
plt.ylabel('Elevation (m)')
plt.xlabel(' Distance Along Profile (m)')
plt.axes(ax[1, 0])
cp1.plot_profiles_in_map_view()
plt.scatter(mg1.x_of_node[(slp1>max_slp) & (da1>500000)], mg1.y_of_node[(slp1>max_slp) & (da1>500000)],c='red', marker='.')

plt.tight_layout()

plt.axes(ax[1, 1])
cp2.plot_profiles_in_map_view()
plt.scatter(mg1.x_of_node[(slp2>max_slp) & (da2>500000)], mg1.y_of_node[(slp2>max_slp) & (da2>500000)], c='red',marker='.')


plt.axes(ax[1, 2])
cp3.plot_profiles_in_map_view()
plt.scatter(mg1.x_of_node[(slp3>max_slp) & (da3>500000)], mg1.y_of_node[(slp3>max_slp) & (da3>500000)], c='red', marker='.')
#%% Chi Calaculations
cf1 = ChiFinder(
     mg1,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=mg1.at_node['drainage_area'].max(),
     clobber=True)

cf2 = ChiFinder(
     mg2,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=mg2.at_node['drainage_area'].max(),
     clobber=True)

cf3 = ChiFinder(
     mg3,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=mg3.at_node['drainage_area'].max(),
     clobber=True)

cf1.calculate_chi()
cf2.calculate_chi()
cf3.calculate_chi()

#%%Pulling out a profile as an array
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,4))
nowfprf={}
for i, outlet_id in enumerate(cp1.data_structure):
    nowfprf['dist'+str(i)]=np.zeros(0)
    #slowelev1=np.zeros(0)
    for j, segment_id in enumerate(cp1.data_structure[outlet_id]):
        totprof = cp1.data_structure[outlet_id][segment_id]
        totprofid=totprof["ids"]
        nowfprf['chi'+str(i)]=mg1.at_node['channel__chi_index'][totprofid]
        nowfprf['elev'+str(i)]=mg1.at_node['topographic__elevation'][totprofid]
        dist=cp1.data_structure[outlet_id][segment_id]["distances"]
        nowfprf['da'+str(i)]=mg1.at_node['drainage_area'][totprofid]
        
        nowfprf['dist'+str(i)] = np.concatenate((nowfprf['dist'+str(i)], dist))
    Nnodes=len(nowfprf['elev'+str(i)])
    nowfprf['slope'+str(i)]=np.zeros(Nnodes)
    nowfprf['slope'+str(i)][1:(Nnodes)] = (1/dx)*(nowfprf['elev'+str(i)][1:(Nnodes)] - nowfprf['elev'+str(i)][0:(Nnodes-1)]) 

for i in range(0,numchannel):
    plt.axes(ax[0,0])
    plt.plot(nowfprf['dist'+str(i)], nowfprf['elev'+str(i)])
    plt.title('No WFs')
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Elevation (m)')
    plt.ylim(0,400)
    plt.axes(ax[1,0])
    plt.plot(nowfprf['dist'+str(i)], nowfprf['slope'+str(i)])    
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Slope (m/m)')
    plt.axhline(y=0.05)

    
fastwfprf={}
for i, outlet_id in enumerate(cp2.data_structure):
    fastwfprf['dist'+str(i)]=np.zeros(0)
    #slowelev1=np.zeros(0)
    for j, segment_id in enumerate(cp2.data_structure[outlet_id]):
        totprof = cp2.data_structure[outlet_id][segment_id]
        totprofid=totprof["ids"]
        fastwfprf['chi'+str(i)]=mg2.at_node['channel__chi_index'][totprofid]
        fastwfprf['elev'+str(i)]=mg2.at_node['topographic__elevation'][totprofid]
        fastwfprf['da'+str(i)]=mg2.at_node['drainage_area'][totprofid]
        dist=cp2.data_structure[outlet_id][segment_id]["distances"]
        
        fastwfprf['dist'+str(i)] = np.concatenate((fastwfprf['dist'+str(i)], dist))
    Nnodes=len(fastwfprf['elev'+str(i)])
    fastwfprf['slope'+str(i)]=np.zeros(Nnodes)
    fastwfprf['slope'+str(i)][1:(Nnodes)] = (1/dx)*(fastwfprf['elev'+str(i)][1:(Nnodes)] - fastwfprf['elev'+str(i)][0:(Nnodes-1)]) 


for i in range(0,numchannel):
    plt.axes(ax[0,1])
    plt.plot(fastwfprf['dist'+str(i)], fastwfprf['elev'+str(i)])
    plt.title('Fast WFs')
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Elevation (m)')
    plt.ylim(0,400)
    plt.axes(ax[1,1])
    plt.plot(fastwfprf['dist'+str(i)], fastwfprf['slope'+str(i)])    
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Slope (m/m)')
    plt.axhline(y=0.05)

    
slowwfprf={}
for i, outlet_id in enumerate(cp3.data_structure):
    slowwfprf['dist'+str(i)]=np.zeros(0)
    #slowelev1=np.zeros(0)
    for j, segment_id in enumerate(cp3.data_structure[outlet_id]):
        totprof = cp3.data_structure[outlet_id][segment_id]
        totprofid=totprof["ids"]
        slowwfprf['chi'+str(i)]=mg3.at_node['channel__chi_index'][totprofid]
        slowwfprf['elev'+str(i)]=mg3.at_node['topographic__elevation'][totprofid]
        dist=cp3.data_structure[outlet_id][segment_id]["distances"]
        slowwfprf['da'+str(i)]= mg3.at_node['drainage_area'][totprofid]
        slowwfprf['dist'+str(i)] = np.concatenate((slowwfprf['dist'+str(i)], dist))
    Nnodes=len(slowwfprf['elev'+str(i)])
    slowwfprf['slope'+str(i)]=np.zeros(Nnodes)
    slowwfprf['slope'+str(i)][1:(Nnodes)] = (1/dx)*(slowwfprf['elev'+str(i)][1:(Nnodes)] - slowwfprf['elev'+str(i)][0:(Nnodes-1)]) 

for i in range(0,numchannel):
    plt.axes(ax[0,2])
    plt.plot(slowwfprf['dist'+str(i)], slowwfprf['elev'+str(i)])
    plt.title('Slow WFs')
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Elevation (m)')
    plt.ylim(0,400)
    plt.axes(ax[1,2])
    plt.plot(slowwfprf['dist'+str(i)], slowwfprf['slope'+str(i)])    
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Slope (m/m)')
    plt.axhline(y=0.05)
    
    #%% zoom in on dynamic zone
    for i in range(0,4):


        #plt.plot(fastwfprf['dist'+str(i)][min(np.where(fastwfprf['dist'+str(i)]>50)[0]):], fastwfprf['slope'+str(i)][min(np.where(fastwfprf['dist'+str(i)]>50)[0]):])    
        plt.plot(fastwfprf['dist'+str(i)], fastwfprf['slope'+str(i)])
        #min(np.where(fastwfprf['dist'+str(i)]>=1850)[0])
        plt.xlabel('Distance Upstream (m)')
        plt.ylabel('Slope (m/m)')
        plt.axhline(y=0.05)
        plt.ylim(-.1, .2)
        #plt.xlim(0,33000)
#%% run one step to mess with the dynamic zone
mg1.at_node["bedrock__elevation"][mg1.core_nodes] += U * timestep
mg1.at_node["topographic__elevation"][:] = (
    mg1.at_node["bedrock__elevation"] + mg1.at_node["soil__depth"]
)
mg2.at_node["bedrock__elevation"][mg2.core_nodes] += U * timestep
mg2.at_node["topographic__elevation"][:] = (
    mg2.at_node["bedrock__elevation"] + mg2.at_node["soil__depth"]
)
mg3.at_node["bedrock__elevation"][mg3.core_nodes] += U * timestep
mg3.at_node["topographic__elevation"][:] = (
    mg3.at_node["bedrock__elevation"] + mg3.at_node["soil__depth"]
)
fr1.run_one_step()
fr2.run_one_step()
fr3.run_one_step()

slp1 = mg1.at_node["topographic__steepest_slope"].copy()
slp2 = mg2.at_node["topographic__steepest_slope"].copy()
slp3 = mg3.at_node["topographic__steepest_slope"].copy()

mg2.at_node["bedrock_erodibility"][slp2>max_slp] = K_max
mg2.at_node["bedrock_erodibility"][slp2<=max_slp] = K_mid

mg3.at_node["bedrock_erodibility"][slp3>max_slp] = K_min
mg3.at_node["bedrock_erodibility"][slp3<=max_slp] = K_mid


# Run SPACE for one time step
sp1.run_one_step(dt=timestep)
sp2.run_one_step(dt=timestep)
sp3.run_one_step(dt=timestep)

cp1.run_one_step()
cp2.run_one_step()
cp3.run_one_step()
#%% slope area plots
plt.loglog(mg1.at_node["drainage_area"],
           mg1.at_node["topographic__steepest_slope"],
           ".",
           color='grey',
           label="baseline")
plt.loglog(mg2.at_node["drainage_area"],
           mg2.at_node["topographic__steepest_slope"],
           ".",
           color='orange',
           label="with increase in K with slope")
plt.loglog(mg3.at_node["drainage_area"],
           mg3.at_node["topographic__steepest_slope"],
           ".",
           color='blue',
           label="with decrease in K with slope")
plt.legend()
plt.ylabel("Slope")
plt.xlabel("Area")
plt.xlim([500000, 2e7])


#%% Chi plots

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
for i in range(0,numchannel):
    plt.axes(ax[0])
    plt.plot(nowfprf['chi'+str(i)], nowfprf['elev'+str(i)])
    ax[0].set_title("baseline")
    #ax[0, 0].set_ylim([0, elev_max])
    plt.ylabel('Elevation (m)')
    plt.xlabel('Chi')
    
    plt.axes(ax[1])
    plt.plot(fastwfprf['chi'+str(i)], fastwfprf['elev'+str(i)])
    ax[1].set_title("with increase in K with slope")
    #ax[0, 1].set_ylim([0, elev_max])
    plt.ylabel('Elevation (m)')
    plt.xlabel(' Chi')
    
    plt.axes(ax[2])
    plt.plot(slowwfprf['chi'+str(i)], slowwfprf['elev'+str(i)])
    ax[2].set_title("with decrease in K with slope")
    #ax[0, 2].set_ylim([0, elev_max])
    plt.ylabel('Elevation (m)')
    plt.xlabel('Chi')