# cs4701-project

## Main Idea of Our Project:

This project utilizes Reinforcement Learning and other machine learning techniques to teach an agent to correct the orbit trajectory within a particle accelerator.

We recommend you run the Jupyter Notebooks on Colab, especially if you have access to Colab's GPU/TPU resources. If you are running locally, ignore the colab install block 
(you might also have to run pip install locally instead of conda install for some of the packages in the Jupyter Notebook, so ignore those blocks as well).
 
## File Structure Organization:

* Lattice Files: Contains the various scripts that initialize the accelerator configuration, run the Bmad simulation with action values as input, and report the orbit values
* pytao_test.ipynb: A test notebook for the Python Package of Bmad (PyTao) that tests its various simulation capabilities and data retrieval api
* stable_baselines_test.ipynb: A test notebook for our RL algorithm package that performs a very simply model training on a toy problem
* environ.py: The script for our accelerator environment. Uses PyTao to generate the state and perform actions, and calculates rewards with different possible functions
* orbit_correctionsv1.ipynb: The main notebook for model training
* hyperparameter_tuning.ipynb: The main notebook for hyperparameter tuning via Bayesian Optimization
