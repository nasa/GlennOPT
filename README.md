# GlennOPT

## Installation Instructions
> `pip install glennopt`

## Objective
The objective of this project is to develop a standalone optimization tool that can be easily integrated into openmdao at a later stage. 
This tool will be entirely written in python making it compatible with windows and linux. 

## Why this tool?
This tool overcomes some of the limitations of gradient based optimization where F(x) = y which isn't always true for all simulations. GlennOPT uses a variation of Genetic Optimizers called Differential Evolution (DE). DE is capable of handling failed simulations. In the event of a failure the objective is set to a high value making the individual" undesireable for mutation and crossover. The other reason why someone would use this tool is if they had other parameters besides the objective that they wanted to keep track of or constrain. A good example of this is optimizing turbomachinery. There are many other parameters other than efficiency and power that matter. You might want to keep track of the mach number entering and exiting the geometry, flow angles, just to name a few. If you need gradient optimization, I recommend OpenMDAO or scipy minimize. 

## Summary
Many optimization packages seem like a compile of past tools written in other languages, they lack universal features described above that can make big data really happen at Glenn

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
[Multi-Objective Kursawe Function](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/kur/multi_objective_example.ipynb)

[Single Objective Rosenbrock](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/Rosenbrock/RosenbrockExample.ipynb)

[Multi and Single Objective Probe Placement](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/ProbePlacement_multi/ProbePlacementExample.ipynb)

# Contributors 
|                   | Position      | Dates            | Responsibility                      |   |
|-------------------|---------------|------------------|-------------------------------------|---|
| Justin Rush       | LERCIP Intern | Fall 2020        | CCD DOE                             |   |
| Nadeem Kever      | LERCIP Intern | Summer-Fall 2020 | Single Objective DE                 |   |
| Trey Harrison     | Researcher    | Fall 2020-       | Probe Optimization test case        |   |
| Paht Juangphanich | Researcher    | Spring 2020-     | Architect                           |   |

# Funding
The development of this library was supported by NASA AATT (Advance Air Transport Technology). The development is supported by NASA Glenn LTE Branch however anyone is welcome to contribute by submitting a pull request. 

# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)
