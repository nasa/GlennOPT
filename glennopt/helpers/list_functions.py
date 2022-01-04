
from typing import List


def check_if_duplicates(listOfElems:List[str]) -> bool:
    """Checks if list has any duplicates

    Args:
        listOfElems (List[str]): List of strings 

    Returns:
        bool: True = duplicates present, False = no duplicates
    """
    
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True