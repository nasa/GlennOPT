Multi-objective Optimization
==============================
Optimization strategies that are multi-objective can be grouped into gradient based and genetic based strategies. This program is geared towards the genetic type. However, we would like to add a gradient based strategy in the future. 

The strategy that we have implemented is Differential evolution and NSGA3 which pairs differental evolution strategies with sorting based on reference point and non-dominated sorting. 

Using NSGA3 - Multi-objective 
-----------------------------------------
NSGA3 is a differential evolution algorithm for optimization of a function using multi-objective. To achieve actual multi-objectivity, the algorithm uses a sorting strategy with reference points to rank the individual. Non-dimensional sorting is used to discard individuals that are poor performing. THis is the key fundamental of how NSGA3 works. 

The other key fundamentals that you shoud know is that the optimizer keeps track of ``2*population size`` in the restart file. 



Mutation strategies
-----------------------------------------
