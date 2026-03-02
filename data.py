import os
ti = len(os.listdir(r'E:\driver-fatigue-detection\dataset\faces\images\train'))
vi = len(os.listdir(r'E:\driver-fatigue-detection\dataset\faces\images\val'))
tl = len(os.listdir(r'E:\driver-fatigue-detection\dataset\faces\labels\train'))
vl = len(os.listdir(r'E:\driver-fatigue-detection\dataset\faces\labels\val'))
print(f'Train images: {ti}')
print(f'Val images:   {vi}')
print(f'Train labels: {tl}')
print(f'Val labels:   {vl}')