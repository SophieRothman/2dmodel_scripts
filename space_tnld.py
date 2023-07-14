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
from landlab.components import BedrockLandslider  # BedrockLandslider model

## Import Landlab utilities
from landlab import RasterModelGrid  # Grid utility
from landlab import imshowhs_grid, imshow_grid  # For plotting results; optional
#%%
# Set grid parameters
num_rows = 50
num_columns = 50
node_spacing = 25.0

# track sediment flux at the node adjacent to the outlet at lower-left
node_next_to_outlet = num_columns + 1

# Instantiate model grid
mg1 = RasterModelGrid((num_rows, num_columns), node_spacing)
mg2 = RasterModelGrid((num_rows, num_columns), node_spacing)
mg3 = RasterModelGrid((num_rows, num_columns), node_spacing)
# add field ’topographic elevation’ to the grid
mg1.add_zeros("node", "topographic__elevation")
mg2.add_zeros("node", "topographic__elevation")
mg3.add_zeros("node", "topographic__elevation")
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
    bottom_is_closed=False,
    left_is_closed=False,
    right_is_closed=False,
    top_is_closed=False,
)
mg3.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=False,
    left_is_closed=False,
    right_is_closed=False,
    top_is_closed=False,
)
mg2.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=False,
    left_is_closed=False,
    right_is_closed=False,
    top_is_closed=False,
)


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
K_min = float(0.83e-5)
K_max = float(7.5e-5)
K_mid = float(2.5e-5)
max_slp = 0.05
min_slp = 0.02
# Instantiate SPACE model with chosen parameters
K1 = mg1.add_ones("bedrock_erodibility", at="node", clobber=True)
K2 = mg2.add_ones("bedrock_erodibility", at="node", clobber=True)
K3 = mg3.add_ones("bedrock_erodibility", at="node", clobber=True)
K1[:] = K_mid
K2[:] = K_mid
K3[:] = K_mid

sp1 = SpaceLargeScaleEroder(
    mg1,
    K_sed=5e-5,
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
    K_sed=5e-5,
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
    K_sed=5e-5,
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
#%%
# Set model timestep
timestep = 1e3  # years

# Set elapsed time to zero
elapsed_time = 0.0  # years

# Set timestep count to zero
count = 0

# Set model run time
run_time = 5e5 # years

# Array to save sediment flux values
sed_flux = np.zeros(int(run_time // timestep))

# Uplift rate in m/yr
U = 1e-3

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
    
    mg2.at_node["bedrock_erodibility"][slp2>max_slp] = K_max
    mg2.at_node["bedrock_erodibility"][slp2<=max_slp] = K_mid

    mg3.at_node["bedrock_erodibility"][slp3>max_slp] = K_min
    mg3.at_node["bedrock_erodibility"][slp3<=max_slp] = K_mid
    

    # Run SPACE for one time step
    sp1.run_one_step(dt=timestep)
    sp2.run_one_step(dt=timestep)
    sp3.run_one_step(dt=timestep)

    # Add to value of elapsed time
    elapsed_time += timestep

    #mg.at_node["water_erodibility"][slp2>max_slp] = K_max
    #mg.at_node["water_erodibility"][slp2<=max_slp] = K_mid


    if np.mod(elapsed_time, 1e5)==0:
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
        
