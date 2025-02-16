import os
import assemblyai as aai

from typing import Optional
from abc import abstractmethod
import threading
import struct
import math

def read_env():
    f = open(".env", "r")
    try:

        lines = f.readlines()

        for line in lines:

            env_vars = line.split("=")
            if len(env_vars) < 2:
                continue

            key = env_vars[0].strip()
            value = env_vars[1].strip()

            os.environ[key] = value # set the environemnt varables
    except:
        print("Error reading .env file")
    
    f.close()






class m_MicrohoneStream(aai.extras.MicrophoneStream):
    
    def __init__(self, sample_rate = 44100, device_index:Optional[int] = None):
        super().__init__(sample_rate, device_index)
        self.activate = False
        
    
    def vad(self):
        for chunk in self:
            rms = self._cmpt_rms(chunk)
            if rms > 0.40:
                print("voice detected")
                self.activate = True
            else:
                self.activate = False

    def _cmpt_rms(self, chunk):
        
        cnt = len(chunk)//2

        if cnt == 0:
            return 0.0

        sum_squares = 0.0
        shrts = struct.unpack('>' + 'h'*cnt, chunk)
        for s in shrts:
            n = s/32768.0
            sum_squares += n * n

        rms = math.sqrt(sum_squares/cnt) 

        return rms