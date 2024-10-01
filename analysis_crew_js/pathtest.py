from pathlib import Path
import os
# home would contain something like "/Users/myname"
home = str(Path.home())

current_directory = os.getcwd()
path = home + "/Desktop/crew_docs"
if not os.path.exists(path): 
    
    os.makedirs(path)

print(home)
print(current_directory)