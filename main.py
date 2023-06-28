
## ------------------------------------------------------------------------
#       Use BASE AI 
## ------------------------------------------------------------------------

import time
#time.sleep(20)

import sys
import os
sys.path.insert(0, '../osais_ai_base')

print("from dir: "+os.getcwd()) 
print(os.environ) 

import cv2

from main_fastapi import app

## For debuging VAI locally ...
# from main_fastapi import initializeApp
# initializeApp("env_vai")
