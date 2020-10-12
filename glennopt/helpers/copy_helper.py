import errno
import os,shutil
 
def copy(src, dest,symlinks=False, ignore=None):    
    for item in os.listdir(src):
        if '__pycache__' not in item:
            s = os.path.join(src, item)
            d = os.path.join(dest, item)
            try:
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)
            except OSError as e:
                print('Directory not copied. Error: %s' % e)