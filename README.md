# ViZDoom_EA
ViZDoom controller experiments using Evolutionary algorithms and Deep learning

To use, make sure you have all ViZDoom and Keras dependencies installed. You will also need a link to the vizdoom.so library situated at ViZDoom/bin/python folder (place it on the root folder of this repository)

Requirements:
- Ubuntu 14.04
- ViZDoom installed and configured https://github.com/Marqt/ViZDoom
- Keras installed and configured https://keras.io/
- MultiNEAT python installation https://github.com/peter-ch/MultiNEAT/tree/master/src
- OpenCV 3.0
- Python concurrent.futures library
- Theano (configure for GPU and cuDNN)

Once ViZDoom is installed, remember to copy the vizdoom.so link library to the root folder of ViZDoom_EA before executing the code.

The code expects ViZDoom to be in a sibling folder to ViZDoom_EA (so both folders should be contained in the same folder).

To run a demo of one of the CNN evolved solutions on pursuit and gather scenario, simply run the python script evolve_controller.py (it is already configured). To run the solution on the health gathering scenario, edit the evolve_controller script and substitute "pursuit_and_gather" for "health_gathering_supreme" throughout lines 28 to 32.

If you have any questions or want to see more, please contact me at carlos.fernandez.musoles@gmail.com 
