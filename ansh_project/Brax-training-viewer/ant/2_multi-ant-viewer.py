import mujoco
import mujoco.viewer
import numpy as np
import time

# Ask user for inputs
xml_path = input("Enter the XML file name (e.g., ant.xml): ")
num_envs = int(input("Enter the number of ants to generate: "))

env_separation = 3  
envs_per_row = int(np.ceil(np.sqrt(num_envs)))  

def generate_unique_colors(num_colors):
    """Generate distinct colors for each agent."""
    return np.random.rand(num_colors, 3).tolist()


def replicate(num_envs, env_separation, envs_per_row, xml_string):
    spec = mujoco.MjSpec.from_string(xml_string)
    spec.copy_during_attach = True 
    
    new_spec = mujoco.MjSpec()
    new_spec.copy_during_attach = True
    
    colors = generate_unique_colors(num_envs)
    
    
    new_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .1], rgba=[0.5, 0.5, 0.5, 1]
    )
    
    
    agent_bodies = []
    for i in range(num_envs):
        render_spec = mujoco.MjSpec.from_string(xml_string)
        render_spec.copy_during_attach = True 
        
        row, col = divmod(i, envs_per_row)
        x_pos, y_pos = col * env_separation, row * env_separation
        frame = new_spec.worldbody.add_frame(pos=[x_pos, y_pos, 0])  
        agent = frame.attach_body(render_spec.body('torso'), str(i), '')  
        
        
        for geom in agent.geoms:
            geom.rgba = colors[i] + [1]
        
        agent_bodies.append(agent)
    
    model = new_spec.compile()
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = num_envs * env_separation * 0.75 
        viewer.cam.azimuth = 180  
        viewer.cam.elevation = -30 
        
        while viewer.is_running():
            avg_position = np.mean(data.qpos.reshape(num_envs, -1), axis=0)
            print(f"Avg Position: [x:{avg_position[0]:.2f}, y:{avg_position[1]:.2f}, z:{avg_position[2]:.2f}]")
            
            data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)  
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

# Load XML file
with open(xml_path, 'r') as f:
    xml_string = f.read()

replicate(num_envs, env_separation, envs_per_row, xml_string)
