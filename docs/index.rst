.. GlennOPT documentation master file, created by
   sphinx-quickstart on Mon Apr  5 15:19:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



GlennOPT Documentation
==================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/plot3d
   notes/connectivity

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/base
   modules/doe
   modules/face
   modules/read
   modules/wr
   modules/read
   modules/write
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Background 
=================

GlennOPT is a library used for single and multi-objective optimization for Computational Applications. There are many libraries out there that can do optimization but many of these are geared towards solving a particular function, calling it and retaining the `objectives` in memory while the optimizer goes about minimizing the objectives. This is great for a modeling where you always have a solution and maybe that's all you need. 

Where GlennOPT really differs from other tools is it's ability to save and retain all the performance parameters, parameters, and the objectives the optimization along with the solutions. While many of these evaluations may not yield an optimized result, some may even fail. These extra data is useful for debugging/understanding the trends of your design space.

The Individual
-----------------

In this documentation and the code you will hear a lot about `individual`. This represents your design parameters for a single evaluation. So lets take a simple function :math:`f(x) = x_1 + x_2 + (x_3)^2` the :math:`x` in :math:`f(x)` is a vector containing :math:`x_1`, :math:`x_2`, :math:`x_3`. This vector of :math:`x` is the evaluation parameters of an individual. An individual in GlennOPT is a class containing the evaluation parameters, performance parameters, objective values, and constraints.

Say your objectives include these two functions :math:`f_1(x) and :math:f_2(x)` where x is a vector shared between these functions. The individual will contain f_1 and f_2. If you have other performance parameter such as Pressure(x) and Speed(x) these can also be tracked within an individual. 

When evaluating using GlennOPT, you specify how many individuals per population and how many populations to run for. Think of it as keeping track of people - you have a population of the group and the number of generations which you are recording data for. The individual represents a single person and all the properties and objectives. 

Now that you have a rough background, please check out the tutorials and documentation on single and multi-objective.

