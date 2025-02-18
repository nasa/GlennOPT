from glennopt.base import Parameter
import re

def read_input_to_dict(input_file:str="input.dat"):
    x = dict()
    with open(input_file, "r") as f: 
        parameter_name = ""
        for line in f:
            split_val = line.split('=')
            if len(split_val)==2: # x1 = 2 # Grab the 2
                matches = re.findall("(\w+)_(\d+$)",split_val[0])
                if len(matches)>0:
                    parameter_name = matches[0][0]
                    if parameter_name not in x.keys():
                        x[parameter_name] = []
                    x[parameter_name].append(float(split_val[1]))
                else:
                    parameter_name = split_val[0]
                    x[parameter_name] = float(split_val[1])

    return x

    
