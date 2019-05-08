import re as re

def fix_filenames(somestr):
    """ Fix terrible spaces in filenames
    """

    def replacenth(string, sub, wanted, n):
        where = [m.start() for m in re.finditer(sub, string)][n - 1]
        before = string[:where]
        after = string[where:]
        after = after.replace(sub, wanted)
        newString = before + after
        return newString

    fn2=somestr.replace(' ','',3)
    fn3=fn2.replace(' ','-',1)
    fn4 = fn3.replace(' ','')
    replacenth(fn4,'-',' ',6)

    return(fn4)
