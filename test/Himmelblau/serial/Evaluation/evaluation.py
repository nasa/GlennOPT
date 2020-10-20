def read_input(input_filename):
    x = []
    with open(input_filename, "r") as f: 
        for line in f:
            split_val = line.split('=')
            if len(split_val)==2: # x1 = 2 # Grab the 2
                x.append(float(split_val[1]))
    return x
 
def print_output(y,perf=None):
    with open("output.txt", "w") as f:        
        f.write('objective1 = {0:.6f}\n'.format(y)) # Output should contain [Name of the Objective/Parameter] = [value] This is read by the optimizer 
        f.write('p1 = {0:.6f}\n'.format(perf[0]))
        f.write('p2 = {0:.6f}\n'.format(perf[1]))
        
 
if __name__ == '__main__':
    x = read_input("input.dat")
    # Call Himmelblau test function 
    import himmelblau as hb
    y = hb.himmelblau(x)

    p1 = x[0] + x[1]
    p2 = x[0]**2 + x[1]**2
    print_output(y,[p1,p2])
    print('done')
    
    