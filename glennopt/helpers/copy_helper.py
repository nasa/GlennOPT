import errno
import os,shutil
 
def copy(src:str, dest:str,symlinks=False, ignore=None):
    """Copies a directory

    Args:
        src (str): source directory
        dest (str): destination directory
        symlinks (bool, optional): This parameter accepts True or False, depending on which the metadata of the original links or linked links will be copied to the new tree. Defaults to False.
        ignore ([type], optional): If ignore is given, it must be a callable that will receive as its arguments the directory being visited by copytree(), and a list of its contents, as returned by os.listdir(). Defaults to None.
    """
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