
## ------------------------------------------------------------------------
#       Use BASE AI 
## ------------------------------------------------------------------------

import time
#time.sleep(20)

## 
# NOTE :    Could NOT make this work with univcorn... cv2 does not load... 
#           therefore testing with inference_gfpgan or directly in docker
## 

import sys
sys.path.insert(0, '../osais_ai_base')
from main_fastapi import app

## For debuging VAI locally ...
# from main_fastapi import initializeApp
# initializeApp("env_vai")
