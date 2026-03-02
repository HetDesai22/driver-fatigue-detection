# save as create_alarm.py and run once
import numpy as np
import wave
import struct

frequency = 1000
duration  = 1.0
volume    = 0.5
framerate = 44100
samples   = int(framerate * duration)

with wave.open(r"E:\driver-fatigue-detection\alarm.wav", 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(framerate)
    for i in range(samples):
        value = int(volume * 32767 * 
                   np.sin(2 * np.pi * frequency * i / framerate))
        f.writeframes(struct.pack('<h', value))

print("✅ alarm.wav created!")