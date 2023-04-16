from datetime import datetime
import os

# print(type(datetime.now().strftime("%Y-%m-%d %H:%M")))
# if !(os.path.isdir(datetime.now().strftime("%Y-%m-%d %H:%M"))):
try :
    os.makedirs(datetime.now().strftime("%Y-%m-%d-%H_%M"))
except:
    print("foder already exist")
