# GlennOPT
## Objective
The objective of this project is to develop a standalone optimization tool that can be easily integrated into openmdao at a later stage. 
This tool will be entirely written in python making it compatible with windows and linux. 

## Why this tool?
Currently OpenMDAO uses pyoptsparse for it's gradient free optimization. 
Pyoptsparse lacks in several areas
1.  The ability to read an optimization input file (this can simplifiy the setup, also provide some sort of documentation and easy changes)
2.  pyOptSparse is written in multiple languages and requires swig.exe and fortran to compile. Users of windows may not have fortran libraries install. 
3.  The optimizer needs to not only track the objectives and constraints, but there needs to be a way to track the parameters of each individual. 
    Parameter tracking example: You optimize a compressor or turbine, you want to track the massflow weighted inter-stage total and static quantities as well as the massflow
                                This is essential in constructing a database as well as debugging if there is something wrong in the simulation. 
                                Parameter tracking can also be exported into tecplot where beautiful contour plots can bew generated (also really helps in debugging and publicatons)
4. Optimization can either be performed in memory or in separate folders for example IND001 (Individual 1) and this contains the geometry, mesh, solution, post-processing
    We need this in case there are any additional analysis that need to be done, you can batch this and grab the results
5. There is a restart feature but not sure how to use it, not properly documented
6. Metamodels and Machine Learning: This tool doesn't have ability to use metamodels or machine learning. Since we are using python, we can easily integrate machine learning algorithms to speed up the optimization.


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
End Goal
Integrate into OpenMDAO for a complete optimization package (needs to be debated)

# Tutorials
[Simple Tutorial](/%2E%2E/wikis/Tutorial:-Prerequisites-and-simplified-example)

[Advance Reference](/%2E%2E/wikis/Tutorial:-Advance)

# Contributors 
Person | Position | Dates | Responsibilities 
Nadeem Kever | NASA LERCIP Intern | Summer-Fall 2020 | Single Objective DE, Gradient Multi-Gradient 
Paht Juangphanich | NASA Researcher | Creator | NSGA-III

