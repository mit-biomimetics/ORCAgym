This is the readme file for the available URDFs of the Minicheetah. 

## Options to view the URDFs

### Option 1: Matlab
- Run the matlab script in `````$GPU_GYM_PATH/resources/robots/mini_cheetah/urdf/evaluateURDFMiniCheetah.m`````
- With this code, one can visualize the colision and/or the visual meshes. So you can easily edit the URDF and visualize it with which ever configuration you want. 

### Option 2: Online
- View the URDF online via the following link https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/index.html
- With this link, you can visualize the visual only, but you can play with the joint configurations. 

### Option 3: Isaac Gym
- As a last step, one can run `````$GPU_GYM_PATH/gpugym/tests/test_env.py````` (or `````$GPU_GYM_PATH/gpugym/scripts/play.py````` or `````$GPU_GYM_PATH/gpugym/scripts/train.py`````)
- With this option, you need to make sure you check the box "Render collision meshes" in the "Viewer" tab in the main menue of Isaac Gym.
- I usually use this option in the end, to make sure that Isaac Gym is parsing the URDF correctly.  

--- 

## The URDFs

### The Simple URDF ```mini_cheetah_simple.urdf```
- This is a work in progress, but the idea is similar to the Humanoid, we should have a URDF version of the MiniCheetah with a simple collision mesh.
