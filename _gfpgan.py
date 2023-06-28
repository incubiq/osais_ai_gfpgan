
##
##      Entry of the AI_GFPGAN prog
##

import sys
import os
subdirectory = os.path.join(os.path.dirname(__file__), 'ai')
sys.path.append(subdirectory)

from ai.runai import fnRun

##
##      ENTRY POINTS
##

## WARMUP Data
def getWarmupData(_id):
    try:
        import time
        from werkzeug.datastructures import MultiDict
        ts=int(time.time())
        sample_args = MultiDict([
            ('-u', 'test_user'),
            ('-uid', str(ts)),
            ('-t', _id),
            ('-cycle', '0'),
            ('-o', 'warmup.jpg'),
            ('-filename', 'warmup.jpg')
        ])
        return sample_args
    except:
        print("Could not call warm up!\r\n")
        return None

## Run Warmup
def runWarmup(_id, fn_osais_runWarmup): 
    _args=getWarmupData(_id)
    fn_osais_runWarmup(fnRun, _args)

## Run AI
def runAI(_args, fn_osais_runAI): 
    fn_osais_runAI(fnRun, _args)

