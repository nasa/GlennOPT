# GlennOPT

## Installation Instructions
> `pip install glennopt`

## Objective
The objective of this project is to develop a standalone multi-objective optimization tool that handles failed simulations and can be used to spam the cluster. 
This tool will be entirely written in python making it compatible with windows and linux. 

## Why this tool?
This tool overcomes some of the limitations of gradient based optimization where F(x) = y which isn't always true for all simulations. GlennOPT uses a variation of Genetic Optimizers called Differential Evolution (DE). DE is capable of handling failed simulations. In the event of a failure the objective is set to a high value making the individual" undesireable for mutation and crossover. The other reason why someone would use this tool is if they had other parameters besides the objective that they wanted to keep track of or constrain. A good example of this is optimizing turbomachinery. There are many other parameters other than efficiency and power that matter. You might want to keep track of the mach number entering and exiting the geometry, flow angles, just to name a few. I recommend scipy minimize if you need something Gradient based.

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

[Dataframe to Tecplot](https://colab.research.google.com/github/nasa/GlennOPT/blob/main/test/csv_to_tecplot/csv_to_tecplot.ipynb)

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


# Complaints about NASA IT
If GitHub Pages doesn’t deploy properly, the issue is likely related to NASA IT support. I’ve repeatedly filed internal tickets requesting help in resolving this problem, but they are often marked as resolved without any communication or follow-up. When a ticket is filed, an email is sent containing only a long ticket number, with no description of the issue. Later, IT may contact me referencing just that number (e.g., “11192345”), and I’m expected to recall what the issue was — which is not practical (Issue #1).

Another challenge is that there’s no accessible history of submitted tickets — unlike, for example, Amazon, where you can easily view your past orders. This lack of transparency makes it difficult to track progress or follow up effectively (Issue #2).

Unfortunately, NASA IT is currently dysfunctional. There’s no unified knowledge base, and many systems seem to be developed by different external vendors, with little to no integration or coordination. It doesn’t appear that any testing is done to ensure these systems communicate with one another. As a result, the burden of identifying and troubleshooting systemic issues often falls on individual researchers. Despite more than a year of digital transformation meetings, there still seems to be no coherent vision for how systems should interoperate or how to empower researchers to work more efficiently — either with each other or with the public.

I sincerely apologize to users of this tool and any NASA software I support. I truly want to provide a better experience. But please understand — I don’t have a team. It’s just me, Paht, maintaining and developing the code, and fixing the bugs.
