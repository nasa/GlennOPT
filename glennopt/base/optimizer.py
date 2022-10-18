#/bin/env python
"""
Optimizer - A base abstract class where all optimizers will inherit from. This class is never to be instantiated directly 
"""
"""
    Python Modules
"""
from os import name
import sys
import os, glob, copy, signal, platform, ctypes
from typing import TypeVar,List, Dict, Tuple
import subprocess
import time
import math
import pprint
import logging
from warnings import catch_warnings
import pandas as pd
"""
    External Modules
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from .individual import Individual
from .parameter import Parameter
from ..helpers import copy_helper, parallel_settings, convert_to_ndarray, check_if_duplicates
import psutil

class Optimizer: 
    """
        Base class for starting an optimization
    """
    def __init__(self,name:str,eval_command:str, eval_folder:str = None,opt_folder:str=None, eval_parameters:List[Parameter]=[], objectives:List[Parameter] = [], performance_parameters:List[Parameter] = [], single_folder_eval=False, overwrite_input_file=False):
        """Initializes the optimizer. This is typically inherited by a particular type of optimizer like NSGA (multi-objective differential evolution) or SODE (single objective differential evolution). Any new optimization strategy should inherit from this class.
        This class handles most of the heavy work of evaluation and reading the results.

        Args:
            name (str): name of the optimizer
            eval_command (str): This is the location of the evaluation script. Optimizer will call this script and read output.txt containing the values of the objectives and any performance parameters, see examples. 
            eval_folder (str, optional): If specified, the evaluation folder will be copied for each execution. Defaults to None.
            opt_folder (str, optional): Working directory where you want to start the optimization (calculatio folder will be created here). Defaults to None.
            eval_parameters (List[Parameter], optional): This is the list of x1-xn that feeds into your execution code. Defaults to [].
            objectives (List[Parameter], optional): The objectives you want to keep track of and the value for when they fail. Defaults to [].
            performance_parameters (List[Parameter], optional): Number of parameters. Defaults to [].
            single_folder_eval (bool, optional): Saves space by deleting the population folder when completed. by default this is false. Defaults to False.
            overwrite_input_file(bool, optional): Specifies whether or not to overwrite the input file when restarting a simulation. Defaults to False.

        Raises:
            Exception: Invalid relative path could not be found for eval script
            Exception: Invalid relative path could not be found for eval folder
            Exception: Invalid absolute path could not be found for eval script
            Exception: Invalid absolute path could not be found for eval folder
        """
        self.name = name        
        assert opt_folder is not None

        try:
            logging.basicConfig(filename=os.path.join(opt_folder,'log.dat'),  level=logging.DEBUG)
            self.logger = logging.getLogger()
        except:
            self.logger = None
        

        if (not os.path.isabs(eval_folder)):
            eval_folder = os.path.join(os.getcwd(),eval_folder)
            if (not os.path.isdir(eval_folder)):
                raise Exception('Invalid relative path could not be found for eval folder {0}'.format(eval_folder))

        if (not os.path.isdir(eval_folder)):
            raise Exception('Invalid absolute path could not be found for eval folder {0}'.format(eval_folder))


        self.evaluation_folder = eval_folder
        self.evaluation_command = eval_command

        if (opt_folder):
            self.optimization_folder = opt_folder
        else:
            self.optimization_folder = os.getcwd()        
        
        if (not os.path.isabs(self.optimization_folder)):
            self.optimization_folder = os.getcwd()


        self.population_track = []
        self.overwrite_input_file = overwrite_input_file
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

        
        objective_names = [o.name for o in self.objectives]
        eval_parameter_names = [o.name for o in self.eval_parameters]
        performance_parameter_names = [o.name for o in self.performance_parameters]
        assert check_if_duplicates(objective_names) == False, "Objective names have to be unique"
        assert check_if_duplicates(eval_parameter_names) == False, "Evaluation Parameter names have to be unique"
        assert check_if_duplicates(performance_parameter_names) == False, "Performance Parameter names have to be unique"

        
    @property
    def use_calculation_folder(self) -> bool:
        """Allows the optimizer to define calculation folders for each call.

        Returns:
            bool: True = use calcuation folder  
        """
        return self.__use_calculation_folder

    @use_calculation_folder.setter
    def use_calculation_folder(self,b:bool=True):
        """ Allows the optimizer to define calculation folders for each call.

        Args:
            b (bool): True = use calcuation folder. Defaults to True
        """
        self.__use_calculation_folder=b

    def change_working_dir(self,new_dir:str):
        """Changes the current working directory

        Args:
            new_dir (str): path to new directory 
        """
        self.optimization_folder = new_dir
    
    def get_current_directory(self) -> str:
        """returns the current working directory

        Returns:
            str: Path to current directory
        """
        return self.optimization_folder
    
    @property
    def parallel_settings(self):
        """Returns the parallel settings dataclass 

        Returns:
            parallel_settings: parallel settings class object
        """
        return self.__parallel_settings

    @parallel_settings.setter
    def parallel_settings(self,settings:parallel_settings):
        """Sets the parallel settings
            This also reads the master machine file if exists and splits up the cores per execution.
            Machine file is basically the computer name followed by an index. See an example below:
                paht-ryzen0
                paht-ryzen1
                paht-ryzen2
                paht-ryzen3
                steve-intel0
                steve-intel1
                steve-intel2
                steve-intel3
                
        Args:
            settings (parallel_settings): dataclass representing the parallel settings to use
        """
        self.__parallel_settings = settings
        # Check for machine file
        machinefile = os.path.join(self.optimization_folder,settings.machine_filename)        
        if os.path.exists(machinefile):
            with open(machinefile, 'r') as f:
                x = f.read().splitlines()
                temp = []
                for i in range(len(x)):
                    temp.append(x[i])
                    if self.__parallel_settings.cores_per_execution > 0:
                        if (i+1) % self.__parallel_settings.cores_per_execution == 0:
                            self.cores_per_evalulation.append(copy.deepcopy(temp))   # Splits up the machine file
                            temp.clear()
                    

    def __create_input_file__(self,individual:Individual):
        """Creates an input file 'inputs.txt' which the evaluation script reads. Will only  create if inputs.txt does not already exist

        Args:
            individual (Individual): Individual's evaluation parameters are read x[1-N] are printed to an evaluation script
        """
        def write_input():
            with open(self.input_file,'w') as f:
                eval_params = individual.get_eval_parameter_list()
                for p in range(len(eval_params)):
                    f.write('{0} = {1}\n'.format(eval_params[p].name,eval_params[p].value))

        if os.path.exists(self.input_file) and self.overwrite_input_file:
            write_input()
        else:       # input file does not exists
            write_input()
       
    def __read_output_file__(self, individual:Individual) -> Individual:
        """Reads the output file i.e. output.txt line by line and parses it for objectives and parameters

        Args:
            individual (Individual): Individual object that is read and parameters are set 

        Returns:
            Individual: [description]
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
        else:   # no output.txt found, something bad happened
            for objective in individual.get_objectives_list():                
                individual.set_objective(name=objective.name,val=objective.value_if_failed)
            for param in individual.get_performance_parameters_list():
                individual.set_performance_parameter(name=param.name,val=param.value_if_failed)
        return individual
    
    def __read_input_file__(self, individual:Individual) -> Individual:
        """Reads the input file i.e. input.dat line by line and parses it for evaluation parameters

        Args:
            individual (Individual): Individual object that will be set

        Returns:
            Individual: returns an individual with updated values 
        """
        # Read the input file
        with open(self.input_file,'r') as f:
            for line in f:
                split_val = line.split('=')
                key = split_val[0].strip()
                val = float(split_val[1].strip())
                individual.set_eval_parameter(name=key,val=val)
        return individual
    
    def __check_population_folder__(self,population_number:int=0) -> str:
        """Formats the population folder

        Args:
            population_number (int, optional): Population Number for example 20. Defaults to 0.

        Returns:
            str: relative path to the population folder 
        """

        if (population_number<0): #  Check current directory for calculation folder
            population_folder = "Calculation/DOE"
        else:
            population_folder = "Calculation/POP{:03d}".format(population_number)
        return population_folder


    def read_population(self,population_number:int=0) -> List[Individual]:
        """Reads the output file in the population/DOE folder

        Args:
            population_number (int, optional): Which population we are evaluating. Defaults to 0.

        Raises:
            Exception: Restart population was not found

        Returns:
            List[Individual]: list of individuals within the population
        """
        population_folder = self.__check_population_folder__(population_number)
        pop_path = os.path.join(self.optimization_folder,population_folder)
        if (not os.path.exists(pop_path)):
            raise Exception("Restart population was not found")
        else:
            ind_directories = [dI for dI in os.listdir(pop_path) if os.path.isdir(os.path.join(pop_path,dI)) and ('IND' in dI)]
        
        ind_directories = sorted(ind_directories)
        
        individuals = list()
        for ind_dir in ind_directories:
            ind = Individual(objectives=copy.deepcopy(self.objectives),eval_parameters=copy.deepcopy(self.eval_parameters),performance_parameters=copy.deepcopy(self.performance_parameters))
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
        """Returns the index of cores_per_execution that is not being used 
            Used to find out which process id has been freed 

        Args:
            pid_list (list): list of process ids from pOpen

        Returns:
            int: index of process id that is no longer being used. 
        """

        if len(pid_list)==0:
            return 0
        c_range = range(len(self.cores_per_evalulation))
        c_range_used = [p['cores_per_execution_indx'] for p in pid_list]
        available_indicies = [x for x in c_range if x not in c_range_used]
        if (len(available_indicies)>0):
            return available_indicies[0]             

    def evaluate_population(self,individuals:List[Individual],population_number:int=0):
        """Evaluates a population of individuals, checks to see if the population exists in the directory. If it already exists, do nothing, read the results, evaluate if there is no output file

        Args:
            individuals (List[Individual]): List of individuals to evaluate 
            population_number (int, optional): population corresponding to the set of individuals. Defaults to 0.
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

        pid_list = list()   # keep track of the process id and start time        
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
                    pid,p = self.__evaluate_individual__(individual=individuals[ind_indx],individual_directory=ind_dir,cores_per_execution=self.cores_per_evalulation[c])
                    pid_list.append({'pid':pid,'start_time':time.perf_counter(),'cores_per_execution_indx':c,'proc':p,'pop':str(individuals[ind_indx].population),'ind':individuals[ind_indx].name})
                else:                    
                    pid,p = self.__evaluate_individual__(individual=individuals[ind_indx],individual_directory=ind_dir)
                    pid_list.append({'pid':pid,'start_time':time.perf_counter(),'proc':p,'pop':str(individuals[ind_indx].population),'ind':individuals[ind_indx].name})
                
                while (len(pid_list)==n_evalulations): # Pause any new executions once we reach maximum number of evaluations 
                    inActive_pids = list()
                    # Loop to check if PID is still active and execution time is less than timeout                    
                    for i in range(len(pid_list)):
                        pid = pid_list[i]['pid']; p = pid_list[i]['proc']; pop = pid_list[i]['pop']; ind_name = pid_list[i]['ind']
                        if self.__check_process_running__(p): # If PID is running check if it exceeded the execution time
                            start_time = pid_list[i]['start_time']
                            time.sleep(0.1) # Check every "x" seconds
                            if (time.perf_counter() - start_time)/60 > self.parallel_settings.execution_timeout:
                                try:
                                    self.__write_proc_log__(p,pop,ind_name)
                                    os.kill(pid, signal.SIGTERM)
                                except:
                                    pass
                                inActive_pids.append(i)
                        else: # PID isn't running, remove it
                            self.__write_proc_log__(p,pop,ind_name)
                            inActive_pids.append(i)
                    
                    for index in sorted(inActive_pids, reverse=True): # removing the inactive PIDs from the list 
                        del pid_list[index]  
                    
            else: # TODO execute individual without creating a bunch of directories. Maybe create execution directories and delete them
                output = subprocess.check_output([self.evaluation_command])
                # Append output to results, need to check first how the output is structured
        time.sleep(0.1)
    
    def __check_process_running__(self,p:subprocess.Popen) -> bool:
        """Checks if a process id is running. This checks by using the poll method.

        Args:
            p (subprocess.Popen): Popen class 

        Returns:
            bool: true if process id is still running 
        """
        if p is not None:
            poll = p.poll()
            if poll == None:
                return True
        return False

    def __check_PID_running__(self,pid:int) -> bool:
        """Checks if process id is running. this checks by trying to kill the process. if there's an error then it's running

        Args:
            pid (int): integer corresponding to the process id 

        Returns:
            bool: true if process is still running
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
            return pid in (p.pid for p in psutil.process_iter())
            
    
    
    def __evaluate_individual__(self,individual:Individual,individual_directory:str,cores_per_execution:list=[]) -> Tuple[int,subprocess.Popen]:
        """Evaluates the individual by copying the evaluation folder into the individual's directory

        Args:
            individual (Individual): [description]
            individual_directory (str): [description]
            cores_per_execution (list, optional): [description]. Defaults to [].

        Returns:
            (tuple): tuple containing:

                **pid** (int): process id
                **process** (subprocess.Popen): logging message string 
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
            p = subprocess.Popen(self.evaluation_command.split(' '),stdout=subprocess.PIPE, stderr=subprocess.STDOUT) # Note: evaluation_command has to read the local machine file. This is not my problem            
            os.chdir(cur_dir)
            return p.pid, p
        return -1,None

    def __write_proc_log__(self,p:subprocess.Popen,pop:int,ind_name:str):
        """Write the log for each process 

        Args:
            p (subprocess.Popen): the Popen object 
            pop (int): population as a ninteger 
            ind_name (str): name of the individual 
        """
        if p is not None:     
            for line in p.stdout:
                if (self.logger):
                    self.logger.debug('POP {0} Indivudual: {1} Message: {2}'.format(pop,ind_name,line.decode("utf-8").replace('\n', ' ').replace('\r', '')).strip())
        
    def load_history_file(self):
        """Reads the history file if exists 
        """
        self.__history_filename = os.path.join(self.optimization_folder,"history.csv")
        if (os.path.exists(self.__history_filename)):
            df_temp = pd.read_csv(self.__history_filename)
            # Check for matching columns
            headers = ['Unnamed: 0','Population','Best Individual']
            eval_param_names =  [p.name for p in self.eval_parameters]
            objective_names = [o.name for o in self.objectives]
            perf_param_names = [o.name for o in self.performance_parameters]
            
            headers.extend(eval_param_names)
            headers.extend(objective_names)
            headers.extend(perf_param_names)
            headers.extend(['pop_diversity','pop_avg_distance'])
            

            if len(headers) == len(list(df_temp.columns)) and headers == list(df_temp.columns):
                self.history = df_temp
            else:   # if the number of parameters change then start from a new file 
                self.history = None
                os.remove(self.__history_filename)

    def append_history_file(self, pop:int, best_ind:Individual,diversity:float,distance:float, train_loss:float=0, test_loss:float=0,mse:float=0):
        """Writes a history.csv file containing the best design(s) this function is called by the inheriting class

        Args:
            pop (int): Population index
            best_ind (Individual): best performing individual 
            diversity (float): diversity of the population 
            distance (float): average distance between individuals of the population
        """
        eval_params = best_ind.eval_parameters
        eval_param_names =  [p.name for p in best_ind.get_eval_parameter_list()]
        
        objectives = best_ind.objectives
        objective_names = [o.name for o in best_ind.get_objectives_list()]
        
        perf_param = best_ind.performance_parameters
        perf_param_names = [o.name for o in best_ind.get_performance_parameters_list()]
        
        header = ['Population', 'Best Individual']
        pop_dir = self.__check_population_folder__(best_ind.population).replace('Calculation/','')
        data = [pop, '{:s}_{:s}'.format(pop_dir,best_ind.name)]
        
        def write_arrays(names,vals):
            for name,val in zip(names,vals):
                header.append(name)
                data.append(val)

        write_arrays(eval_param_names,eval_params)
        write_arrays(objective_names,objectives)
        write_arrays(perf_param_names,perf_param)
        
        header.extend(['pop_diversity','pop_avg_distance','train_loss','test_loss','actual_mse_loss'])
        data.extend([diversity,distance,train_loss,test_loss,mse])
        if (not os.path.exists(self.__history_filename)):                        
            self.history = pd.DataFrame(dict(zip(header, data)),index=[0])
            self.history.to_csv(self.__history_filename)
        else:            
            self.history = self.history.append(dict(zip(header, data)),ignore_index=True)
            self.history.to_csv(self.__history_filename)

    def append_restart_file(self, individuals:List[Individual]):
        """Appends self.population_track to a restart file, these are the individuals matter and can be restarted from. Instead of restarting from a population, lets restart from the best individuals 

        Args:
            individuals (List[Individual]): list of individuals, this can be any size. these individuals will be added to the restart file
        """
        df = self.to_pandas(individuals=individuals,bReturnPandas=True)
        df.to_csv(os.path.join(self.optimization_folder,'restart_file.csv'))

   
    def read_restart_file(self) -> List[Individual]:
        """Appends self.population_track to a restart file, these are the individuals matter and can be restarted from. Instead of restarting from a population, lets restart from the best individuals 

        Raises:
            Exception: Error about restart file not being the same. this can happen if you add evaluation parameters or performance parameters. Make sure number of columns match the number of parameters. You can add null or dummy values. 

        Returns:
            List[Individual]: individual list or empty list
        """

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
                    ind.name = row['individual']
                    ind.population = int(row['population'].replace('DOE','-1').replace('POP',''))
                    [ind.set_eval_parameter(p.name,row[p.name]) for p in self.eval_parameters]
                    [ind.set_objective(p.name,row[p.name]) for p in self.objectives]
                    [ind.set_performance_parameter(p.name,row[p.name]) for p in self.performance_parameters]                    
                    individual_list.append(copy.deepcopy(ind))
                return individual_list
        
        return []

    def to_pandas(self,individuals:List[Parameter],pop_number:int=0,bReturnPandas:bool = False):
        """Exports the results to a pandas file 

        Args:
            individuals (List[Parameter]): [description]
            pop_number (int, optional): [description]. Defaults to 0.
            bReturnPandas (bool, optional): [description]. Defaults to False.

        Returns:
            pd.DataFrame: dataframe object with all the optimization results 
        """

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

    @staticmethod
    def df_to_tecplot(df_dict:Dict[str,pd.DataFrame],filename:str):
        """Staticmethod to convert a normal pandas dataframe to tecplot file

        Args:
            df (Dict[str,pd.DataFrame]): For example {'DOE': pd.DataFrame}
        """
        firstKey = next(iter(df_dict))
        firstPandas = df_dict[firstKey]

        # header
        variables = []
        for col in firstPandas.columns:
            variables.append('\"{0}\"'.format(col))
        head = 'VARIABLES = ' + ','.join(variables) + '\n'

        # zones
        zones = []
        for key, df in df_dict.items():
            i = 0
            for index, row in df_dict[key].iterrows():
                zone_str = 'ZONE T = \"{0}\"\n'.format(key+"_"+df.iloc[i]['individual'])
                data_str = ' '
                data = []
                i+=1
                for col in df_dict[key].columns:
                    if col=='population':
                        if row[col] == "DOE":
                            data.append(str(-1))
                        else:
                            data.append(str(row[col].replace('POP','')))
                    elif col=="individual":
                        data.append(str(row[col].replace('IND','')))
                    else:
                        if type(row[col])== float:
                            data.append("{:.6E}".format(row[col]))
                        elif type(row[col])== int:
                            data.append("{0}".format(row[col]))
                        else:
                            data.append(row[col])
                zones.append(zone_str)
                zones.append(' '.join(data) + '\n')
        
        # write to .tec file
        with open(filename,'w') as f:
            # write headers
            f.write(head)
            # write zones
            for zone in zones:
                f.write(zone)

    def to_tecplot(self):
        """Converts the dataframe to a tecplot file (.tec)
        """
        if len(self.pandas_cache)==0:
            return
        if (not os.path.exists(os.path.join(self.optimization_folder,''))):
            os.makedirs(os.path.join(self.optimization_folder,''))

        filename = os.path.join(self.optimization_folder,'','database.tec')
        self.df_to_tecplot(self.pandas_cache,filename)

    
    def create_restart(self):
        """Create a restart file containing all individuals of all populations
        """
    
        pop_individuals = self.read_calculation_folder()
        for individuals in pop_individuals:
            self.append_restart_file(individuals)

    def plot_2D(self,obj1_name:str,obj2_name:str,xlim:list=None,ylim:list=None):
        """Creates a 2D plot scatter plot of all the individuals for the two objectives specified

        Args:
            obj1_name (str): name of the x-axis
            obj2_name (str): name of the y-axis
            xlim (list, optional): xbounds example [-1,2]. Defaults to None.
            ylim (list, optional): ybounds example [-5,5]. Defaults to None.
        """
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
        if xlim is not None:
            ax.set_xlim(xlim[0],xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])
        ax.legend(legend_labels)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
 
    def read_calculation_folder(self) -> List[Individual]:
        """Reads the entire calculation folder to a dataframe and returns all the individuals as an array
            this can be useful for restarting a population

        Raises:
            Exception: [description]

        Returns:
            List[Individual]: list of all individuals for all populations (could be used as a restart)
        """
        
        pop_path = os.path.join(self.optimization_folder,'Calculation')
        try:
            list_subfolders = [int(f.name.replace('POP','').replace('DOE','-1')) for f in os.scandir(pop_path) if f.is_dir()]
        except:
            raise Exception("Invalid directory found. Make sure directories are either DOE or POP001")
        
        individual_list = []
        self.pandas_cache = dict()
        for pop_indx in list_subfolders:
            individuals = self.read_population(pop_indx)            
            self.to_pandas(copy.deepcopy(individuals),pop_indx)
            individual_list.append(copy.deepcopy(individuals))
        return individual_list


    def to_dict(self):
        """Export the settings used to create the optimizer to dict. Also exports the optimization results if performed

        Returns:
            dict: list of all the settings, could be used for creating an optimization object
        """
        settings = dict()
        settings['eval_folder'] = self.evaluation_folder
        settings['opt_folder'] = self.optimization_folder
        settings['single_folder_eval'] = self.single_folder_eval
        settings['eval_parameters'] = [p.to_dict() for p in self.eval_parameters]
        settings['objectives'] = [o.to_dict() for o in self.objectives]
        settings['perf_parameters'] = [p.to_dict() for p in self.performance_parameters]

        self.read_calculation_folder()
        results = dict()
        for k in self.pandas_cache.keys():
            results[k] = self.pandas_cache[k].to_dict()
        
        settings['results'] = results
        return settings
    
    def from_dict(self,settings:dict):
        """Reads the dictionary file and creates the base object

        Args:
            settings (dict): dictionary created by to_dict()
        """
        self.evaluation_folder = settings['eval_folder']
        self.optimization_folder = settings['opt_folder']
        self.single_folder_eval = settings['single_folder_eval']
        
        self.eval_parameters = [Parameter().from_dict(p) for p in settings['eval_parameters']]
        self.objectives = [Parameter().from_dict(p) for p in settings['objectives']]
        self.performance_parameters = [Parameter().from_dict(p) for p in settings['perf_parameters']]
        self.read_calculation_folder()