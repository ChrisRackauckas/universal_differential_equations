# Model-based learning of the Fisher-KPP equation 

The following scripts generate data by solving the Fisher-KPP equation and fitting a universal partial differential equation (UPDE) as follows:

1. Fisher-KPP-CNN.jl (used for this paper): fits the reaction term with a feed-forward neural network and the diffusion term with a CNN 

2. Fisher-KPP-NN.jl: fits the reaction term with a feed-forward neural network while assuming the diffusion term 

3. Fisher-KPP-NN-D.jl: fits the reaction term with a feed-forward neural network and optimizes the diffusion coefficient simultaneously 


