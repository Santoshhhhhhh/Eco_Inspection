'''
First Step:
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt


copy and paste custom_data.yaml file in yolov5/data
'''
import subprocess
import torch
# import utils
# display = utils.notebook_init()  # checks

try:
    command = f"python3 yolov5/train.py --img 416 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --cache"
    result = subprocess.run(command, shell=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}: {e.stderr}")


# Continue to train the YOLOv5s on Custom_data further using the last saved weights(change the path of last weight)
try:
    command = f"python3 yolov5/train.py --img 416 --batch 16 --epochs 50 --data custom_data.yaml --weights runs/train/exp2/weights/last.pt --cache"
    result = subprocess.run(command, shell=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}: {e.stderr}")


# Run this cell to save the model as a zip file (Change the path of the model as per the latest exp in the run/train directory)
try:
    command = f"zip -r trained_model_new.zip yolov5/runs/train/exp3"
    result = subprocess.run(command, shell=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}: {e.stderr}")


