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
If GitHub Pages doesn’t deploy properly, the issue is likely related to NASA IT support. I’ve repeatedly filed internal tickets to resolve this problem, but they are often marked as resolved without any communication or follow-up. When a ticket is filed, an email is sent with a long ticket number, but no description of the issue. Later, IT may contact me referencing just the ticket number (e.g., “11192345”), and I’m expected to remember what the issue was — which is not practical (Issue #1).

Another challenge is that there is no accessible history of tickets I’ve submitted — unlike, for example, how Amazon shows your past orders. This lack of transparency makes it difficult to track progress or follow up (Issue #2).

Unfortunately, NASA IT is currently dysfunctional. There’s no unified knowledge base, and many systems seem to be developed by different external vendors with little to no integration or coordination. There appears to be minimal testing to ensure that these systems work together. As a result, the burden often falls on individual researchers to identify and troubleshoot systemic issues. Despite over a year of digital transformation meetings, there is still no clear vision for how tools should interconnect or how to better support researchers in working efficiently — both internally and with the public.

I sincerely apologize to the users of this tool and any NASA software I support. I want to provide a better experience. But please understand, I don’t have a team — it’s just me, Paht, maintaining the code and fixing the bugs.
