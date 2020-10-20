def read_input(input_filename):
    x = []
    with open(input_filename, "r") as f: 
        for line in f:
            split_val = line.split('=')
            if len(split_val)==2: # x1 = 2 # Grab the 2
                x.append(float(split_val[1]))
    return x
 
def print_output(y):
    with open("output.txt", "w") as f:        
        f.write('objective1 = {0:.6f}\n'.format(y[0])) # Output should contain [Name of the Objective/Parameter] = [value] This is read by the optimizer 
        f.write('objective2 = {0:.6f}\n'.format(y[1]))
        f.write('p1 = {0:.6f}\n'.format(y[2]))
        f.write('p2 = {0:.6f}\n'.format(y[3]))
        f.write('p3 = {0:.6f}\n'.format(y[4]))
        # f.write('Objective2 = {0:.6f}\n'.format(y))
 
if __name__ == '__main__':
    x = read_input("input.dat")
    # Call Rosebrock test function 
    import kur as kur
    import time
    y = kur.KUR(x[0],x[1],x[2])
    print_output(y)
    