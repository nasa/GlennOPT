#/bin/env python
"""
Optimizer - A base abstract class where all optimizers will inherit from. This class is never to be instantiated directly 
"""
"""
    Python Modules
"""
import sys
import os, glob, copy, signal, platform, ctypes
from typing import TypeVar,List
import subprocess
import time
import math
import pprint

"""
    External Modules
"""
import numpy as np
from glennopt.base_classes import Individual
from glennopt.helpers import Parameter, copy_helper, parallel_settings

class Optimizer: 
    """
        Base class for starting an optimization
    """
    def __init__(self,name:str,eval_script:str, eval_folder:str = None,opt_folder:str=None, eval_parameters:List[Parameter]=[], objectives:List[Parameter] = [], performance_parameters:List[Parameter] = [], single_folder_eval=False):
        """
            initializes the optimizer base class
            Inputs:
                name - name of the optimizer
                eval_folder - If specified, copy the evaluation folder for each execution
                eval_script - This is the location of the evaluation script
                opt_folder  - Working directory where you want to start the optimization (calculatio folder will be created here)
                eval_parameters - This is the list of x1-xn that feeds into your execution code
                objectives - how many objectives do you want to keep track of and the value for when they fail
                performance_parameters - number of parameters 
                single_folder_eval - saves space by deleting the population folder when completed. by default this is false
        """
        self.name = name        

        if (not os.path.isabs(eval_script)):
            eval_script = os.path.join(os.getcwd(),eval_script)
            if (not os.path.isfile(eval_script)):
                raise Exception('Invalid relative path could not be found for eval script {0}'.format(eval_script))

        if (not os.path.isabs(eval_folder)):
            eval_folder = os.path.join(os.getcwd(),eval_folder)
            if (not os.path.isdir(eval_folder)):
                raise Exception('Invalid relative path could not be found for eval folder {0}'.format(eval_folder))

        if (not os.path.isfile(eval_script)):
            raise Exception('Invalid absolute path could not be found for eval script {0}'.format(eval_script))

        if (not os.path.isdir(eval_folder)):
            raise Exception('Invalid absolute path could not be found for eval folder {0}'.format(eval_folder))


        self.evaluation_folder = eval_folder
        self.evaluation_script = eval_script

        if (opt_folder):
            self.optimization_folder = opt_folder
        else:
            self.optimization_folder = os.getcwd()        
        
        if (not os.path.isabs(self.optimization_folder)):
            self.optimization_folder = os.path.join(os.getcwd(),eval_script)


        self.population_track = []
        self.input_file = 'input.dat'
        self.output_file = 'output.txt'
        self.__use_calculation_folders = True

        # Optimization variables to track
        self.eval_parameters = eval_parameters
        self.objectives = objectives
        self.performance_parameters = performance_parameters
        self.__parallel_settings = parallel_settings()
        self.cores_per_evalulation = []
        # Cache storage
        self.pandas_cache = {} # Appends individuals to pandas dataframe each dictionar contains the population number
        self.single_folder_eval = single_folder_eval
    @property
    def use_calculation_folder(self):
        """
            Allows the optimizer to define calculation folders for each call.
        """
        return self.__use_calculation_folder

    @use_calculation_folder.setter
    def use_calculation_folder(self,b):
        """
            Allows the optimizer to define calculation folders for each call.
        """
        self.__use_calculation_folder=b
       
    def change_working_dir(self,new_dir:str):
        """
            Changes the current working directory            
        """
        self.optimization_folder = new_dir
    
    def get_current_directory(self):
        """
            Returns the current directory
        """
        return self.optimization_folder
    
    @property
    def parallel_settings(self):
        return self.__parallel_settings

    @parallel_settings.setter
    def parallel_settings(self,settings:parallel_settings):
        self.__parallel_settings = settings
        # Check for machine file
        machinefile = os.path.join(self.optimization_folder,settings.machine_filename)
        if os.path.exists(machinefile):
            with open(machinefile, 'r') as f:
                x = f.read().splitlines()
                temp = []
                for i in range(len(x)):
                    temp.append(x[i])
                    if (i+1) % self.__parallel_settings.cores_per_execution == 0:
                        self.cores_per_evalulation.append(copy.deepcopy(temp))   # Splits up the machine file
                        temp.clear()
                    

    def __create_input_file__(self,individual:Individual):
        """
            Creates an input file 'inputs.txt' which the evaluation script reads
            Individual's evaluation parameters are read x[1-N] are printed to an evaluation script
        """        
        with open(self.input_file,'w') as f:
            eval_params = individual.get_eval_parameter_list()
            for p in range(len(eval_params)):
                f.write('{0} = {1}\n'.format(eval_params[p].name,eval_params[p].value))
    

    def __read_output_file__(self, individual:Individual):
        """
            Reads the output file i.e. output.txt line by line and parses it for objectives and parameters
        """        
        # Read the output file
        if (os.path.exists(self.output_file)):
            with open(self.output_file,'r') as f:
                for line in f:
                    split_val = line.split('=')
                    key = split_val[0].strip()
                    val = float(split_val[1].strip())
                    individual.set_objective(name=key,val=val)
                    individual.set_performance_parameter(name=key,val=val)
        return individual
    
    def __read_input_file__(self, individual:Individual):
        """
            Reads the input file i.e. input.dat line by line and parses it for evaluation parameters
        """        
        # Read the input file
        with open(self.input_file,'r') as f:
            for line in f:
                split_val = line.split('=')
                key = split_val[0].strip()
                val = float(split_val[1].strip())
                individual.set_eval_parameter(name=key,val=val)
        return individual
        
    def __check_population_folder__(self,population_number:int=0):
        """
            Formats the population folder
        """        
        if (population_number<0): #  Check current directory for calculation folder
            population_folder = "Calculation/DOE"
        else:
            population_folder = "Calculation/POP{:03d}".format(population_number)
        return population_folder

    
    def read_population(self,population_number:int=0):
        """
            Reads the output file in the population/DOE folder
            inputs:
                population number - which population we are evaluating

            returns:
                individuals - list of individuals within the population
        """  
        population_folder = self.__check_population_folder__(population_number)
        pop_path = os.path.join(self.optimization_folder,population_folder)
        if (not os.path.exists(pop_path)):
            raise Exception("Restart population was not found")
        else:
            ind_directories = [dI for dI in os.listdir(pop_path) if os.path.isdir(os.path.join(pop_path,dI)) and ('IND' in dI)]
        
        ind_directories = sorted(ind_directories)
        
        individuals = []
        for ind_dir in ind_directories:
            ind = Individual(objectives=self.objectives,eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters)
            ind.name = ind_dir
            ind.population = population_number
            current_directory = os.getcwd()
            ind_dir = os.path.join(pop_path,ind_dir)
            os.chdir(ind_dir)
            ind = self.__read_output_file__(ind)
            ind = self.__read_input_file__(ind)
            individuals.append(copy.deepcopy(ind))
            os.chdir(current_directory)

        return individuals

    def __select_cores_per_execution__(self,pid_list:list):
        '''
            returns the index of cores_per_execution that is not being used 
        '''
        if len(pid_list)==0:
            return 0
        c_range = range(len(self.cores_per_evalulation))
        c_range_used = [p['cores_per_execution_indx'] for p in pid_list]
        available_indicies = [x for x in c_range if x not in c_range_used]
        if (len(available_indicies)>0):
            return available_indicies[0]             

    def evaluate_population(self,individuals:List[Individual],population_number:int=0):
        """
            Evaluates a population of individuals, checks to see if the population exists in the directory. If it already exists, do nothing, read the results, evaluate if there is no output file
        """
        population_folder = self.__check_population_folder__(population_number)

        pop_path = os.path.join(self.optimization_folder,population_folder)
        if (not os.path.exists(pop_path)):
            os.makedirs(pop_path)
        
        if len(self.cores_per_evalulation)>0:     # Determine whether to use machine file or not
            n_evalulations = len(self.cores_per_evalulation)
            use_cores = True
        else:
            n_evalulations = self.parallel_settings.concurrent_executions
            use_cores = False

        pid_list = [] # keep track of the process id and start time
        eval_count = 0;# c is the counter for cores per interation [['paht_cpu1','paht_cpu1','paht_cpu1'],['paht_cpu2','paht_cpu2','paht_cpu2']]
        # Evaluate each individual
        for ind_indx in range(len(individuals)):            
            individuals[ind_indx].name = 'IND{:03d}'.format(ind_indx)
            individuals[ind_indx].population = population_number
            ind_dir = os.path.join(pop_path, individuals[ind_indx].name)
            if (not os.path.exists(ind_dir)):
                os.makedirs(ind_dir)
            # Check if eval folder exists
            if (self.evaluation_folder):
                # Evaluate Individual (calls method in base class)
                if (use_cores):
                    c = self.__select_cores_per_execution__(pid_list)
                    pid = self.__evaluate_individual__(individual=individuals[ind_indx],individual_directory=ind_dir,cores_per_execution=self.cores_per_evalulation[c])
                    pid_list.append({'pid':pid,'start_time':time.perf_counter(),'cores_per_execution_indx':c})                    
                else:                    
                    pid = self.__evaluate_individual__(individual=individuals[ind_indx],individual_directory=ind_dir)
                    pid_list.append({'pid':pid,'start_time':time.perf_counter()})

                while (len(pid_list)==n_evalulations):
                    # Loop to check if PID is still active and execution time is less than timeout
                    inActive_pids = []
                    for i in range(len(pid_list)):
                        pid = pid_list[i]['pid']
                        if self.__check_PID_running__(pid): # If PID is running check if it exceeded the execution time
                            start_time = pid_list[i]['start_time']
                            time.sleep(0.1) # Check every 0.01 seconds
                            if (time.perf_counter() - start_time)/60 > self.parallel_settings.execution_timeout:
                                os.kill(pid, signal.SIGTERM)
                                inActive_pids.append(i)
                        else: # PID isn't running, remove it 
                            inActive_pids.append(i)
                    
                    for index in sorted(inActive_pids, reverse=True): # removing the inactive PIDs from the list 
                        del pid_list[index]                    
                    
            else: # TODO execute individual without creating a bunch of directories. Maybe create execution directories and delete them
                output = subprocess.check_output(['python', self.evaluation_script])
                # Append output to results, need to check first how the output is structured
    
    def __check_PID_running__(self,pid):
        """
            Checks if a pid is still running (UNIX works, windows we'll see)
            Inputs:
                pid - process id
            returns:
                True if running, False if not
        """
        if (platform.system() == 'Linux'):
            try:
                os.kill(pid, 0)
                if pid<0:               # In case code terminates
                    return False
            except OSError:
                return False 
            else:
                return True
        elif (platform.system() == 'Windows'):
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x100000

            process = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
            if process != 0:
                kernel32.CloseHandle(process)
                return True
            else:
                return False
    
    
    def __evaluate_individual__(self,individual:Individual,individual_directory:str,cores_per_execution:list=[]):
        """
            Evaluates the individual by copying the evaluation folder into the individual's directory
            Returns the process id of the execution      
        """
        # Check for output.txt
        if (not os.path.exists(os.path.join(individual_directory,self.output_file))): # This prevents an evaluation if there is already an output
            copy_helper.copy(self.evaluation_folder,individual_directory)
            cur_dir = os.getcwd()
            os.chdir(individual_directory)            
            # Create the input file
            self.__create_input_file__(individual)
            # Create local machine file            
            if (len(cores_per_execution)>0):                
                with open(self.parallel_settings.machine_filename,'w') as f:
                    f.write('\n'.join(cores_per_execution))
                    
            # Evaluate
            p = subprocess.Popen(['python', self.evaluation_script]) # Note: evaluation_script has to read the local machine file. This is not my problem
            os.chdir(cur_dir)
            return p.pid
        return -1
    
    
    def append_restart_file(self, individuals:List[Individual]):
        """
            Appends self.population_track to a restart file, these are the individuals matter and can be restarted from. Instead of restarting from a population, lets restart from the best individuals 

            Inputs:
                individuals - list of individuals, this can be any size. these individuals will be added to the restart file
        """        
        df = self.to_pandas(individuals=individuals,bReturnPandas=True)
        df.to_csv(os.path.join(self.optimization_folder,'restart_file.csv'))


    def read_restart_file(self):
        """
            Appends self.population_track to a restart file, these are the individuals matter and can be restarted from. Instead of restarting from a population, lets restart from the best individuals 

            Inputs:
                None
            Returns:
                individual list or empty list
        """

        import pandas as pd

        restart_file = os.path.join(self.optimization_folder,'restart_file.csv')
        if os.path.exists(restart_file):
            df = pd.read_csv(restart_file)
            df_columns = df.columns
            n_eval = len(self.eval_parameters)
            n_obj = len(self.objectives)
            n_perf = len(self.performance_parameters)
            if ((len(df_columns)-3) != (n_eval+n_obj+n_perf)):
                raise Exception("this is not the right restart file. number of columns does not match the sum of evaluation, objective, and performance parameters")
            else:
                individual_list = []
                offset = 3 # this is the column where data starts
                for index,row in df.iterrows():
                    ind = Individual(eval_parameters=self.eval_parameters,objectives=self.objectives,performance_parameters=self.performance_parameters)
                    [ind.set_eval_parameter(p.name,row[p.name]) for p in self.eval_parameters]
                    [ind.set_objective(p.name,row[p.name]) for p in self.objectives]
                    [ind.set_performance_parameter(p.name,row[p.name]) for p in self.performance_parameters]                    
                    individual_list.append(copy.deepcopy(ind))
                return individual_list
        
        return []

    def to_pandas(self,individuals:List[Parameter],pop_number:int=0,bReturnPandas:bool = False):
        """
            Creates a new dataframe with the results of a population. Call this in a loop for each population
        """
        import pandas as pd

        def add_value_to_dict(dict_obj,key,val):
            indx = 1
            while (key in dict_obj):
                key = key + "_" + str(indx)
                indx+=1
            dict_obj[key] = val
            return dict_obj
        
        # Create data as a dict of names, the only thing that will break this is if the dictionary has repeated keys
        data = []
        for individual in individuals:
            if (individual.population<0):
                Population_Name = 'DOE'
            else:
                Population_Name = 'POP{0:03d}'.format(individual.population)
            
            temp_dict = {}
            add_value_to_dict(temp_dict,'population',Population_Name) # name of the population IND000 for example
            add_value_to_dict(temp_dict,'individual',individual.name) # name of the individual IND000 for example
            
            for eval_param in individual.get_eval_parameter_list():            
                add_value_to_dict(temp_dict,eval_param.name,eval_param.value)

            for objective in individual.get_objectives_list():
                add_value_to_dict(temp_dict,objective.name,objective.value)

            for perf_param in individual.get_performance_parameters_list():
                add_value_to_dict(temp_dict,perf_param.name,perf_param.value)
                
            data.append(temp_dict) # TODO Check if this is duplicated, if so then probably needs to be copied
        if (bReturnPandas):
            return pd.DataFrame(data)

        if pop_number<0:
            self.pandas_cache['DOE'] = pd.DataFrame(data)
        else:
            self.pandas_cache['POP{0:03d}'.format(pop_number)] = pd.DataFrame(data)

    def to_tecplot(self):
        """
            Converts the dataframe to a tecplot file (.tec)
        """
        if len(self.pandas_cache)==0:
            return
        firstKey = next(iter(self.pandas_cache))
        firstPandas = self.pandas_cache[firstKey]

        # header
        variables = []
        for col in firstPandas.columns:
            variables.append('\"{0}\"'.format(col))
        head = 'VARIABLES = ' + ','.join(variables) + '\n'

        # zones
        zones = []
        for key, df in self.pandas_cache.items():
            zone_str = 'ZONE T = \"{0}\"\n'.format(key+"_"+df.iloc[0]['name'])
            for index, row in self.df.iterrows():
                data = []
                for col in self.df.columns:
                    if (col != 'name'):
                        data.append(row[col])
                data_str = ' '.join(data) + '\n'

            zones.append(zone_str)
            zones.append(data_str)
        
        # write to .tec file
        if (not os.path.exists(os.path.join(self.optimization_folder,'Database'))):
            os.makedirs(os.path.join(self.optimization_folder,'Database'))

        db_filename = os.path.join(self.optimization_folder,'Database','database.tec')
        with open(db_filename,'w') as f:
            # write headers
            f.write(head)
            # write zones
            for zone in zones:
                f.write(zone)
    
    def create_restart(self):
        ''' 
            Create a restart file containing all individuals of all populations
        '''
        individuals = self.read_calculation_folder()
        self.append_restart_file(individuals)

    def plot_2D(self,obj1_name:str,obj2_name:str):
        """
            Creates a 2D plot scatter plot of all the individuals for the two objectives specified

        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig,ax = plt.subplots()

        colors = cm.rainbow(np.linspace(0, 1, len(self.pandas_cache.keys())))        
        indx = 0
        legend_labels = []
        # Scan the pandas file, grab objectives for each population
        for key, df in self.pandas_cache.items():
            obj1_data = []
            obj2_data = []
            c=colors[indx]
            for index, row in df.iterrows():
                obj1_data.append(row[obj1_name])
                obj2_data.append(row[obj2_name])
            # Plot the gathered data
            ax.scatter(obj1_data, obj2_data, color=c, s=5,alpha=0.5)
            legend_labels.append(key)
            indx+=1

        ax.set_xlabel(obj1_name)
        ax.set_ylabel(obj2_name)
        ax.legend(legend_labels)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
    
    def get_best(self):
        '''
            Get the best individuals from each population

            Returns:
                best_individuals - this is an array of individuals that are best at each objective
                    [ 
                        POP001: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                        POP002: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                        POP003: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                    ]
                comp_individuals - this is an array of individuals that is the best compromise between all the objectives
        '''
        # Read calculation folder
        individuals = self.read_calculation_folder()            # Reads as an array of arrays 
        
        # (Compromise Target) At the minimum index of each objective what are the values of the other objectives
        nobjectives = len(self.objectives)        
        min_objective_values = np.ndarray( (nobjectives,),dtype=float)
        
        for indx,pop_individuals in enumerate(individuals):
            for ind in pop_individuals:
                objectives_temp = ind.objectives
                for o in range(nobjectives):
                    if indx==0:
                        min_objective_values[o] = objectives_temp[o]
                    else:
                        if (objectives_temp[o] < min_objective_values[o]):
                            min_objective_values[o] = objectives_temp[o]
        #
        import operator # for sorting list of classes         

        best_individuals = dict()
        dist = list(); comp_individual = list()
        dist_temp = list(); comp_individual_temp = list()
        prev_pop = individuals[0][0].population

        for pop_individuals in individuals:
            pop = pop_individuals[0].population
            for ind in pop_individuals:
                if pop not in best_individuals.keys():        # Prepopulate
                    best_individuals[pop] = list()
                    for o in range(nobjectives):
                        best_individuals[pop].append(ind) 
                else:                                       # Compare   
                    for o in range(nobjectives): # Checks for the best objective
                        current_best = best_individuals[pop][o].objectives[o]
                        if ind.objectives[o]<current_best:
                            best_individuals[pop][o] = ind 
            
                # Checks for best compromise that has the smallest distance to minimum of all objective values 
                
                current_objectives = ind.objectives
                d = 0 
                for o in range(len(objectives_temp)):
                    current_best = best_individuals[pop][o].objectives[o]
                    d = ((current_objectives[o] - current_objectives[o])/min_objective_values[o])**2
                d = math.sqrt(d)
                dist_temp.append(d)
                comp_individual_temp.append(ind)
                
            # Lets find the miniumum distance and best compromise for a given population
            min_indx = dist_temp.index(min(dist_temp))
            dist.append(dist_temp[min_indx])
            comp_individual.append(comp_individual_temp[min_indx])
            dist_temp.clear()
            comp_individual_temp.clear()
            prev_pop = pop
        return best_individuals, comp_individual
        
    def plot_best_objective(self,objective_index):
        """
            Creates a plot of the population vs the objective value
            INPUTS:
                objective_name - objective to compare 
        """
        best_individuals, comp_individual = self.get_best()
        
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        indx = 0
        legend_labels = []
        
        objective_data = list()
        best_indivduals = list()
        for _,ind in best_individuals.items():
            objective_data.append(ind[objective_index].objectives[objective_index])
            best_indivduals.append(ind[objective_index].name)
        
        fig,ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, len(self.pandas_cache.keys())))
        ax.scatter(best_individuals.keys(), objective_data,color='blue',s=5)
        ax.set_yscale('log')
        ax.set_xticks(list(best_individuals.keys()))
        ax.set_xlabel('Population')
        ax.set_ylabel('Objective Value')
        ax.legend(legend_labels)
        ax.set_title('Objective Index: ' + str(objective_index))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

        print('Best Individuals for Objective Index ' + str(objective_index))

    # def plot_best_compromise(self)

    def read_calculation_folder(self):
        """
            Reads the entire calculation folder to a dataframe and returns all the individuals as an array
            this can be useful for restarting a population

            returns:
                individual_list - list of all individuals for all populations (could be used as a restart)
        """
        pop_path = os.path.join(self.optimization_folder,'Calculation')
        try:
            list_subfolders = [int(f.name.replace('POP','').replace('DOE','-1')) for f in os.scandir(pop_path) if f.is_dir()]
        except:
            raise Exception("Invalid directory found. Make sure directories are either DOE or POP001")
        
        individual_list = []
        self.pandas_cache = {}
        for pop_indx in list_subfolders:
            individuals = self.read_population(pop_indx)            
            self.to_pandas(copy.deepcopy(individuals),pop_indx)
            individual_list.append(copy.deepcopy(individuals))
        return individual_list
