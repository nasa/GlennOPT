# GlennOPT
## Objective
The objective of this project is to develop a standalone optimization tool that can be easily integrated into openmdao at a later stage. 
This tool will be entirely written in python making it compatible with windows and linux. 

## Why this tool?
This tool overcomes some of the limitations of gradient based optimization where F(x) = y. This isn't always true for all simulations. If you need gradient optimization then go with OpenMDAO.

## Summary
Many optimization packages seem like a compile of past tools written in other langauges, they lack unverisal features described above that can make big data really happen at Glenn

## Project Goals and Tasks
Key Features
*  Keeps track of each evaluation
*  Restarts from an Population folder
*  Execution timeouts (If CFD is stuck, kill it, but keep eval folder)
*  Exports all results to tecplot, csv 
*  Track performance parameters
*  Apply constraints to performance parameters 
Future Features
*  Addition of other optimization algorithms
*  Incorporate metamodels (machine learning, kriging) directly in the optimization


# Tutorials
[Multi-Objective Kursawe Function](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/KUR/multi_objective_example.ipynb)

[Single Objective Rosenbrock](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/Rosenbrock/RosenbrockExample.ipynb)

[Multi and Single Objective Probe Placement](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/ProbePlacement_multi/ProbePlacementExample.ipynb)

# Contributors 
|                   | Position      | Dates            | Responsibility                      |   |
|-------------------|---------------|------------------|-------------------------------------|---|
| Justin Rush       | LERCIP Intern | Fall 2020        | CCD DOE                             |   |
| Nadeem Kever      | LERCIP Intern | Summer-Fall 2020 | Single Objective DE                 |   |
| Trey Harrison     | Researcher    | Fall 2020-       | Probe Optimization test case        |   |
| Paht Juangphanich | Researcher    | Spring 2020-     | Architect                           |   |


