""


import matplotlib.pylab as plt
import numpy as np
#import holoviews as hv
from landlab import  RasterModelGrid
from landlab.plot.graph import plot_graph
from landlab.components import LinearDiffuser, TaylorNonLinearDiffuser
from landlab.components import FlowAccumulator, FastscapeEroder, PriorityFloodFlowRouter


from landlab import RasterModelGrid, imshow_grid
from landlab.components import FlowAccumulator, LinearDiffuser, FastscapeEroder, ChannelProfiler
from landlab.components import ChiFinder

import rasterio

#%%
inimg=r'C:\Users\soph1\Downloads\KWCA_GEG_01M.tif'#r"C:\Users\soph1\Downloads\outputfile.tif"#
ds=rasterio.open(inimg)
data=ds.read()
ds.meta

#%% visualization stuff
from rasterio.plot import show
show(ds, cmap="Spectral")
#%% making ita grid
datagrid1=np.zeros([grid1.number_of_cell_rows, grid1.number_of_cell_columns])
for i in range(0, grid1.number_of_cell_rows):
    datagrid1[i,0:grid1.number_of_cell_columns]=grid1.at_node["topographic__elevation"][grid1.number_of_cell_columns*i:grid1.number_of_cell_columns*(i+1)]
    


#%% writing
#w is for write
with rasterio.open(r"C:\Users\soph1\Downloads\outputfile.tif", 'w',
                   driver=ds.driver,
                   height=grid1.number_of_cell_rows,
                   width=grid1.number_of_cell_columns,
                   count=1,
                   crs=ds.crs,
                   transform=ds.transform,
                   dtype='float64'
                   ) as dst:
    dst.write(datagrid1)
#%%
with rasterio.open(r"C:\Users\soph1\Downloads\outputmodel.tif", 'w',
                   driver=ds.driver,
                   height=ds.height,
                   width=ds.width,
                   count=ds.count,
                   crs=ds.crs,
                   transform=ds.transform,
                   dtype=data.dtype,
                   ) as dst:
    dst.write(data)
#%%
#grid = HexModelGrid((20, 10), 50)

nr = 60
nc = 50
dx = 75

np.random.seed(5000)
grid1 = RasterModelGrid((nr, nc), dx)
grid2 = RasterModelGrid((nr, nc), dx)
grid3 = RasterModelGrid((nr, nc), dx)
initial_noise = np.random.rand(grid1.core_nodes.size)

z1 = grid1.add_zeros("topographic__elevation", at="node")
z1[grid1.core_nodes] += initial_noise

z2 = grid2.add_zeros("topographic__elevation", at="node")
z2[grid2.core_nodes] += initial_noise

z3 = grid3.add_zeros("topographic__elevation", at="node")
z3[grid3.core_nodes] += initial_noise

grid1.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
grid1.set_watershed_boundary_condition_outlet_id(0, grid1.at_node['topographic__elevation'], -9999.0)


grid2.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
grid2.set_watershed_boundary_condition_outlet_id(0, grid2.at_node['topographic__elevation'], -9999.0)


grid3.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
grid3.set_watershed_boundary_condition_outlet_id(0, grid3.at_node['topographic__elevation'], -9999.0)




grid3.imshow("topographic__elevation")
#%%
# zreshape1=np.zeros((nr, nc))

# #read in the random noise into the 2D array
# for i in range(0, grid1.number_of_node_rows):
#     zreshape1[i,0:grid1.number_of_node_columns]=grid1.at_node["topographic__elevation"][grid1.number_of_node_columns*i:grid1.number_of_node_columns*(i+1)]

# #Tilt the 2D array
# for i in range(0, nr):
#     zreshape1[i,:]=zreshape1[i,:]+(((0*i)/nr))
# for j in range(0, nc):
#     zreshape1[:, j]=zreshape1[:,j]+(((0*j)/nc))

# #Now reshape back to a 1D array
# for i in range(0, nr):
#     z1[(grid1.number_of_node_columns*(i)):(grid1.number_of_node_columns*(i+1))]=zreshape1[i, :]
# grid1.imshow("topographic__elevation")
# # If that works, copy this grid to z2 and z3
# grid2.at_node["topographic__elevation"]=np.copy(z1)
# grid3.at_node["topographic__elevation"]=np.copy(z1[:])
# grid2.imshow("topographic__elevation")

#%%
# uplift, diffusivity
U = 5e-4
D = float(2e-4)*250

# range of K and max slope for slp-K linear scaling


K_min = float(5e-7)*20
K_max = float(2e-6)*20
K_mid = float(1e-6)*20
max_slp = 0.05
min_slp = 0.02

# timestep
total_time = 5e6
dt = 1000
nts = int(total_time / dt)
Sc=0.6
#%%
K1 = grid1.add_ones("water_erodibility", at="node")
K1[:] = K_mid

K2 = grid2.add_ones("water_erodibility", at="node")
K2[:] = K_mid

K3 = grid3.add_ones("water_erodibility", at="node")
K3[:] = K_mid
#%%
# save ouput

interval = 2
n_out = int(nts / interval)

z1_out = np.empty((n_out, grid1.shape[0], grid1.shape[1]))
z2_out = np.empty((n_out, grid2.shape[0], grid2.shape[1]))
z3_out = np.empty((n_out, grid3.shape[0], grid3.shape[1]))


#%%
thresh=0
fr1 = PriorityFloodFlowRouter(grid1, flow_metric="D8",  suppress_out= True)
fr2 = PriorityFloodFlowRouter(grid2, flow_metric="D8", suppress_out= True)
fr3 = PriorityFloodFlowRouter(grid3, flow_metric="D8", suppress_out= True)

sp1 = FastscapeEroder(grid1, K_sp="water_erodibility")
#nl1 = PerronNLDiffuse(grid1, 
#                      nonlinear_diffusivity=D,
#                      S_crit=33.0 * np.pi / 180.0,
#                      rock_density=2700.0,
#                      sed_density=2700.0,)
tnld1 = TaylorNonLinearDiffuser(
    grid1, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True, nterms=2
)



sp2 = FastscapeEroder(grid2, K_sp="water_erodibility")
tnld2 = TaylorNonLinearDiffuser(
    grid2, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True, nterms=2
)



sp3 = FastscapeEroder(grid3, K_sp="water_erodibility")
tnld3 = TaylorNonLinearDiffuser(
    grid3, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True, nterms=2
)

#ld3 = LinearDiffuser(grid3, linear_diffusivity=D)

#%%
dathresh=0
stp1 = grid1.add_zeros("stream_power", at="node", clobber="True")
stp2 = grid2.add_zeros("stream_power", at="node", clobber="True")
stp3 = grid3.add_zeros("stream_power", at="node", clobber="True")
#Running the model
elapsed_time=0.0
for i in range(0,nts):

    # run first half with uplift of U
   # if i > nts / 2:
    #    factor = 2
   # else:
    factor = 1

    # uplift
    z1[grid1.core_nodes] += U * dt * factor
    z2[grid2.core_nodes] += U * dt * factor
    z3[grid3.core_nodes] += U * dt * factor

    # topographic__steepest_slope is update in fa.r1s()
    fr1.run_one_step()
    fr2.run_one_step()
    fr3.run_one_step()

    # now we want to update K.
    # say that a waterfall is anything above max_slp
    # When slope is slp_min lets make K = K_min
    # When slope is max_slp or above lets make K = K_max
    # and linearly change between min and max slp
    # We could make K continue to increase, but it might get way too
    # fast... so we'll threshold it for the moment

    slp2 = grid2.at_node["topographic__steepest_slope"].copy()
    #slp2[slp2 > max_slp] = max_slp
    #slp2[slp2 < min_slp] = min_slp

   # grid2.at_node["water_erodibility"][:] = K_mid + (K_max - K_mid) * (
   #     slp2 - min_slp) / (max_slp - min_slp)
    
    #grid2.at_node["water_erodibility"][slp2>max_slp] = K_max
    #grid2.at_node["water_erodibility"][slp2<=max_slp] = K_mid
    
    # we'll also make K highest when slopes are shallowest and lowest when steepest.
    slp3 = grid3.at_node["topographic__steepest_slope"].copy()
    slp1 = grid1.at_node["topographic__steepest_slope"].copy()
    da3 = grid3.at_node["drainage_area"].copy()
    da2 = grid2.at_node["drainage_area"].copy()

    
    grid2.at_node["water_erodibility"][:] = K_mid    
    grid2.at_node["water_erodibility"][ (slp2>max_slp)] = K_max


    grid3.at_node["water_erodibility"][:] = K_mid
    grid3.at_node["water_erodibility"][ (slp3>max_slp)] = K_min
    
    #slp3[slp3 > max_slp] = max_slp
    #slp3[slp3 < min_slp] = min_slp

    #grid3.at_node["water_erodibility"][:] = K_mid + (K_min - K_mid) * (
    #    slp3 - min_slp) / (max_slp - min_slp)
    
    #grid3.at_node["water_erodibility"][slp3>max_slp] = K_min
    #grid3.at_node["water_erodibility"][slp3<=max_slp] = K_mid

    # run stream power and diffusion.
    sp1.run_one_step(dt)
    tnld1.run_one_step(dt)

    sp2.run_one_step(dt)
    tnld2.run_one_step(dt)

    sp3.run_one_step(dt)
    tnld3.run_one_step(dt)
    elapsed_time+=dt
    if elapsed_time==1e5:
         thresh=1.5e-3
         sp3 = FastscapeEroder(grid3, K_sp="water_erodibility", threshold_sp=thresh)
         sp2 = FastscapeEroder(grid2, K_sp="water_erodibility", threshold_sp=thresh)
         sp1 = FastscapeEroder(grid1, K_sp="water_erodibility", threshold_sp=thresh)

    if i % interval == 0:
        ind = int(i / interval)
        z1_out[ind, :, :] = z1.reshape(grid1.shape)
        z2_out[ind, :, :] = z2.reshape(grid2.shape)
        z3_out[ind, :, :] = z3.reshape(grid3.shape)
        
    if np.mod(elapsed_time, (total_time/10))==0:
        stp1=grid1.at_node["water_erodibility"]* (grid1.at_node['drainage_area']**.5) *grid1.at_node['topographic__steepest_slope']
        stp2=grid2.at_node["water_erodibility"]* (grid2.at_node['drainage_area']**.5) *grid2.at_node['topographic__steepest_slope']
        stp3=grid3.at_node["water_erodibility"]* (grid3.at_node['drainage_area']**.5) *grid3.at_node['topographic__steepest_slope']

        cmap='winter'
        print('%.2f of model run completed' %(elapsed_time/total_time))
        plt.figure()
        imshow_grid(grid1, 'topographic__elevation', plot_name="no waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
        plt.scatter(grid1.x_of_node[stp1>(thresh)], grid1.y_of_node[stp1>(thresh)], c='red',marker='.')
        plt.show()
        plt.figure()
        imshow_grid(grid2, 'topographic__elevation', plot_name="fast waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
        plt.scatter(grid2.x_of_node[stp2>(thresh)], grid2.y_of_node[stp2>(thresh)], c='red',marker='.')

        plt.show()
                    
        plt.figure()
        imshow_grid(grid3, 'topographic__elevation', plot_name="slow waterfalls", cmap = cmap, colorbar_label="Elevation (m)")
        plt.scatter(grid3.x_of_node[stp3>(thresh)], grid3.y_of_node[stp3>(thresh)], c='red',marker='.')

        plt.show()
#%%stream power

stp1=grid1.at_node["water_erodibility"]* (grid1.at_node['drainage_area']**.5) *grid1.at_node['topographic__steepest_slope']
stp2=grid2.at_node["water_erodibility"]* (grid2.at_node['drainage_area']**.5) *grid2.at_node['topographic__steepest_slope']
stp3=grid3.at_node["water_erodibility"]* (grid3.at_node['drainage_area']**.5) *grid3.at_node['topographic__steepest_slope']

imshow_grid(grid2,
            stp2,
            plot_name="baseline",
            vmin=1e-6,
            vmax=2e-3,
            cmap='gist_earth')   


    #%%
plt.plot(grid1.at_node["topographic__steepest_slope"][grid1.core_nodes],
         grid1.at_node["water_erodibility"][grid1.core_nodes],
         ".",
         label="baseline")
plt.plot(grid2.at_node["topographic__steepest_slope"][grid2.core_nodes],
         grid2.at_node["water_erodibility"][grid2.core_nodes],
         ".",
         label="with increase in K with slope")
plt.plot(grid3.at_node["topographic__steepest_slope"][grid2.core_nodes],
         grid3.at_node["water_erodibility"][grid2.core_nodes],
         ".",
         label="with decrease in K with slope")
plt.legend()
plt.xlabel("Slope")
plt.ylabel("K")

#%%
#Producing images of the rasters
slp1 = grid1.at_node["topographic__steepest_slope"].copy()
slp2 = grid2.at_node["topographic__steepest_slope"].copy()
slp3 = grid3.at_node["topographic__steepest_slope"].copy()

da1=grid1.at_node["drainage_area"].copy()
da2=grid2.at_node["drainage_area"].copy()
da3=grid3.at_node["drainage_area"].copy()
elev_max = max(z1.max(), z2.max(), z3.max())

plt.figure()
imshow_grid(grid1,
            z1,
            plot_name="baseline",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')          
#plt.scatter(grid1.x_of_node[stp1>thresh], grid1.y_of_node[stp1>thresh], c='red',marker='.')

plt.figure()
imshow_grid(grid2,
            z2,
            plot_name="with fast waterfall rule",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')
#plt.scatter(grid2.x_of_node[stp2>(thresh)], grid2.y_of_node[stp2>(thresh)], c='red',marker='.')


plt.figure()
imshow_grid(grid3,
            z3,
            plot_name="with slow waterfall rule",
            vmin=0,
            vmax=elev_max,
            cmap='gist_earth')
#plt.scatter(grid3.x_of_node[stp3>(thresh)], grid3.y_of_node[stp3>(thresh)], c='red',marker='.')


#%%
#Steepness maps
slp_vmax = max(grid1.at_node["topographic__steepest_slope"].max(),
               grid2.at_node["topographic__steepest_slope"].max(),
               grid3.at_node["topographic__steepest_slope"].max())

plt.figure()
imshow_grid(grid1,
            "topographic__steepest_slope",
            plot_name="baseline",
            vmin=0,
            vmax=slp_vmax)
plt.figure()
imshow_grid(grid2,
            "topographic__steepest_slope",
            plot_name="with increase in K with slope",
            vmin=0,
            vmax=slp_vmax)
plt.figure()
imshow_grid(grid3,
            "topographic__steepest_slope",
            plot_name="with decrease in K with slope",
            vmin=0,
            vmax=slp_vmax)

#%%
#Channel profiles

numchannel=1;

cp1 = ChannelProfiler(grid1, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
cp1.run_one_step()

cp2 = ChannelProfiler(grid2, number_of_watersheds=numchannel, minimum_channel_threshold=500000 ,main_channel_only=True)
cp2.run_one_step()

cp3 = ChannelProfiler(grid3, number_of_watersheds=numchannel,  minimum_channel_threshold=500000 , main_channel_only=True)
cp3.run_one_step()
slp3 = grid3.at_node["topographic__steepest_slope"].copy()
slp1 = grid1.at_node["topographic__steepest_slope"].copy()
slp2 = grid2.at_node["topographic__steepest_slope"].copy()

da3 = grid3.at_node["drainage_area"].copy()
da1 = grid1.at_node["drainage_area"].copy()
da2 = grid2.at_node["drainage_area"].copy()


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
plt.scatter(grid1.x_of_node[(slp1>max_slp) & (da1>500000)], grid1.y_of_node[(slp1>max_slp) & (da1>500000)],c='red', marker='.')

plt.tight_layout()

plt.axes(ax[1, 1])
cp2.plot_profiles_in_map_view()
plt.scatter(grid1.x_of_node[(slp2>max_slp) & (da2>500000)], grid1.y_of_node[(slp2>max_slp) & (da2>500000)], c='red',marker='.')


plt.axes(ax[1, 2])
cp3.plot_profiles_in_map_view()
plt.scatter(grid1.x_of_node[(slp3>max_slp) & (da3>500000)], grid1.y_of_node[(slp3>max_slp) & (da3>500000)], c='red', marker='.')


#%%
plt.loglog(grid3.at_node["drainage_area"],
          grid3.at_node["topographic__steepest_slope"],
          ".",
          color='blue',
          label="with decrease in K with slope")
plt.loglog(grid1.at_node["drainage_area"],
           grid1.at_node["topographic__steepest_slope"],
           ".",
           color="grey",
           label="baseline")
plt.loglog(grid2.at_node["drainage_area"],
           grid2.at_node["topographic__steepest_slope"],
           ".",
           color='orange',
           label="with increase in K with slope")

#plt.xlim([500000, 2e7])
plt.legend()
plt.ylabel("Slope")
plt.xlabel("Area")
#%% Drainage density
from landlab.components import DrainageDensity
channels = np.array(grid1.at_node['drainage_area'] > 500000, dtype=np.uint8)

dd = DrainageDensity(grid1,
                      area_coefficient=1.0,
                      slope_coefficient=1.0,
                      area_exponent=1.0,
                      slope_exponent=0.0,
                      channelization_threshold=5)
dd = DrainageDensity(grid1, channel__mask=channels)
mean_drainage_density = dd.calculate_drainage_density()
   # >>> np.isclose(mean_drainage_density, 0.3831100571)
   # True



#%%
#Trying to make slope plots
# prf1 = ChannelProfiler(
#     grid1,
#     number_of_watersheds=4,
#     main_channel_only=True,
#     #minimum_channel_threshold=dx**2,
# )
# prf2 = ChannelProfiler(
#     grid2,
#     number_of_watersheds=4,
#     main_channel_only=True,
#     #minimum_channel_threshold=dx**2,
# )

# prf3 = ChannelProfiler(
#     grid3,
#     number_of_watersheds=4,
#     main_channel_only=True,
#     #minimum_channel_threshold=dx**2,
# )
# prf1.run_one_step()
# prf2.run_one_step()
# prf3.run_one_step()
# #%%
# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,4))
# for i, outlet_id in enumerate(cp1.data_structure):
#     for j, segment_id in enumerate(cp1.data_structure[outlet_id]):
#         if j == 0:
#             label = f"channel {i + 1}"
#         else:
#             label = "_nolegend_"
#         segment = cp1.data_structure[outlet_id][segment_id]
#         profile_ids = segment["ids"]
#         color = segment["color"]
#         plt.axes(ax[1, 0])
#         plt.plot(
#             cp1.data_structure[outlet_id][segment_id]['distances'],

#             grid1.at_node["topographic__steepest_slope"][profile_ids],
            
#             "-",
#             color=color,
#             label=label,
            
#         )
#         plt.axhline(y=0.05)
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('slope')
        
#         plt.axes(ax[0, 0])
#         plt.plot(
#             #grid1.at_node['distances'][profile_ids],
#             cp1.data_structure[outlet_id][segment_id]['distances'],
#             grid1.at_node["topographic__elevation"][profile_ids],
#             "-",
#             color=color,
#             label=label,
#         )
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('elevation (m)')

#         plt.title('no waterfalls')

# for i, outlet_id in enumerate(cp2.data_structure):
#     for j, segment_id in enumerate(cp2.data_structure[outlet_id]):
#         if j == 0:
#             label = f"channel {i + 1}"
#         else:
#             label = "_nolegend_"
#         segment = cp2.data_structure[outlet_id][segment_id]
#         profile_ids = segment["ids"]
#         color = segment["color"]
#         plt.axes(ax[1, 1])
#         plt.plot(
#             cp2.data_structure[outlet_id][segment_id]['distances'],

#             grid2.at_node["topographic__steepest_slope"][profile_ids],
#             "-",
#             color=color,
#             label=label,
            
#         )
#         plt.axhline(y=0.05)
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('slope')
#         plt.axes(ax[0, 1])
#         plt.plot(
#             cp2.data_structure[outlet_id][segment_id]['distances'],
#             grid2.at_node["topographic__elevation"][profile_ids],

#             "-",
#             color=color,
#             label=label,
#         )
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('elevation (m)')

#         plt.title('fast waterfalls')

        
        
# for i, outlet_id in enumerate(cp3.data_structure):
#     for j, segment_id in enumerate(cp3.data_structure[outlet_id]):
#         if j == 0:
#             label = f"channel {i + 1}"
#         else:
#             label = "_nolegend_"
#         segment = cp3.data_structure[outlet_id][segment_id]
#         profile_ids = segment["ids"]
#         color = segment["color"]
#         plt.axes(ax[1,2])
#         plt.plot(
#             segment['distances'],

#             grid3.at_node["topographic__steepest_slope"][profile_ids],
#             "-",
#             color=color,
#             label=label,
            
#         )
#         plt.axhline(y=0.05)
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('slope')


#         plt.axes(ax[0, 2])
#         plt.plot(
#             cp3.data_structure[outlet_id][segment_id]['distances'],

#             grid3.at_node["topographic__elevation"][profile_ids],

#             "-",
#             color=color,
#             label=label,
#         )
#         plt.xlabel('distance from outlet (m)')
#         plt.ylabel('elevation (m)')

#         plt.title('slow waterfalls')


#%% Chi Calaculations
cf1 = ChiFinder(
     grid1,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=grid1.at_node['drainage_area'].max(),
     clobber=True)

cf2 = ChiFinder(
     grid2,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=grid2.at_node['drainage_area'].max(),
     clobber=True)

cf3 = ChiFinder(
     grid3,
     min_drainage_area=500000.,
     use_true_dx=True,
     reference_concavity=0.5,
     reference_area=grid3.at_node['drainage_area'].max(),
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
        nowfprf['chi'+str(i)]=grid1.at_node['channel__chi_index'][totprofid]

        nowfprf['elev'+str(i)]=grid1.at_node['topographic__elevation'][totprofid]
        dist=cp1.data_structure[outlet_id][segment_id]["distances"]
        nowfprf['da'+str(i)]=grid1.at_node['drainage_area'][totprofid]
        
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
        fastwfprf['chi'+str(i)]=grid2.at_node['channel__chi_index'][totprofid]

        fastwfprf['elev'+str(i)]=grid2.at_node['topographic__elevation'][totprofid]
        fastwfprf['da'+str(i)]=grid2.at_node['drainage_area'][totprofid]
        dist=cp2.data_structure[outlet_id][segment_id]["distances"]
        fastwfprf['chi'+str(i)]=grid2.at_node['channel__chi_index'][totprofid]
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
        slowwfprf['chi'+str(i)]=grid3.at_node['channel__chi_index'][totprofid]

        slowwfprf['elev'+str(i)]=grid3.at_node['topographic__elevation'][totprofid]
        dist=cp3.data_structure[outlet_id][segment_id]["distances"]
        slowwfprf['da'+str(i)]=grid3.at_node['drainage_area'][totprofid]
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


    plt.plot(fastwfprf['dist'+str(i)][min(np.where(fastwfprf['dist'+str(i)]>1250)[0]):], fastwfprf['slope'+str(i)][min(np.where(fastwfprf['dist'+str(i)]>1250)[0]):])    
    #min(np.where(fastwfprf['dist'+str(i)]>=1850)[0])
    plt.xlabel('Distance Upstream (m)')
    plt.ylabel('Slope (m/m)')
    plt.axhline(y=0.05)

#%%
#plt.loglog(grid1.at_node["drainage_area"],
#           grid1.at_node["topographic__steepest_slope"],
#           ".",
#           label="baseline")
for i in range(0,numchannel):
    plt.loglog(fastwfprf["da"+str(i)],
               fastwfprf["slope"+str(i)],
               ".",
               color='orange',
               label="with increase in K with slope"
               )
    plt.loglog(slowwfprf["da"+str(i)],
               slowwfprf["slope"+str(i)],
               ".",
               color='blue',
               label="with decrease in K with slope"
               )
    plt.loglog(nowfprf["da"+str(i)],
               nowfprf["slope"+str(i)],
               ".",
               color='grey',
               label="with decrease in K with slope"
               )
    plt.legend()
    plt.xlim([500000, 2e8])




plt.legend()
plt.ylabel("Slope")
plt.xlabel("Area")
#%% Chi plots

cf1.calculate_chi()
cf2.calculate_chi()
cf3.calculate_chi()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
for i in range(0,numchannel):
    plt.axes(ax[0])
    plt.plot(nowfprf['chi'+str(i)], nowfprf['elev'+str(i)])
    ax[0].set_title("baseline")
    ax[ 0].set_ylim([0, max(grid1.at_node['topographic__elevation'])])
    plt.ylabel('Elevation (m)')
    plt.xlabel('Chi')
    
    plt.axes(ax[1])
    plt.plot(fastwfprf['chi'+str(i)], fastwfprf['elev'+str(i)])
    ax[1].set_title("with increase in K with slope")
    ax[1].set_ylim([0, max(grid1.at_node['topographic__elevation'])])
    plt.ylabel('Elevation (m)')
    plt.xlabel(' Chi')
    
    plt.axes(ax[2])
    plt.plot(slowwfprf['chi'+str(i)], slowwfprf['elev'+str(i)])
    ax[2].set_title("with decrease in K with slope")
    ax[2].set_ylim([0, max(grid1.at_node['topographic__elevation'])])
    plt.ylabel('Elevation (m)')
    plt.xlabel('Chi')