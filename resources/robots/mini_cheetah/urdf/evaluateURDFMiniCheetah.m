clear all; close all; clc
% WARNING : The collision bodies between Matlab and and Isaac gym are
% different, and the URDF should be validated in Isaac Gym.

% Define the path of the URDF, the number of DoFs, and the default config
simple_urdf_name = "./mini_cheetah_simple.urdf"; % URDF path
ndof=12; % number of DoFs
config = [0.0,-0.785398,1.596976,0.0,-0.785398,1.596976,0.0,-0.785398,1.596976,0.0,-0.785398,1.596976]'; % Default configuration

% Import the URDF
mini_cheetah_simple = importrobot(simple_urdf_name);
mini_cheetah_simple.DataFormat = 'column';

% Show the URDF, you have the option to show the colision meshes and/or the visualization mesh
figure(1)
show(mini_cheetah_simple, config, 'Collisions','on', 'Visuals','off'); title("Simple URDF")
