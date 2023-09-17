from flask import Flask, render_template,request,json
from werkzeug.exceptions import abort
import cv2
import os
import numpy as np
import tensorflow as tf
import subprocess
import glob
import string,random,datetime

############################    EXTERIOR    ################################
CATEGORIES = ['Front_View', 'Rear_View', 'Left_Side_View', 'Right_Side_View', 'Left_Front_Corner_View',
              'Left_Rear_Corner_View', 'Right_Front_Corner_View', 'Right_Rear_Corner_View',
              'Engine_Compartment', 'Under_Hood', 'Roof_View', 'Others']

classes_front = ["Hood", "Grille", "Front Windshield", "Front Bumper","Right Headlight", "Left Headlight"]
classes_right = ["Fender", "Front Wheel", "Quarter Panel","Rear Wheel", "Front Door", "Rear Door"]
classes_left = ["Fender", "Front Wheel", "Quarter Panel","Rear Wheel", "Front Door", "Rear Door"]
classes_rear = ['Windshield','Left Brake Lights','Trunk','Right Brake Lights','Bumper']

# Returns the most frequent prediction in the combination of three Models
# max1 = prediction of first model
# max2 = prediction of second model
# max3 = prediciton of third model
def find_most_frequent_number(max_1, max_2, max_3):

    frequency_dict = {max_1: 0, max_2: 0, max_3: 0}
    frequency_dict[max_1] += 1
    frequency_dict[max_2] += 1
    frequency_dict[max_3] += 1

    max_frequency = max(frequency_dict.values())

    most_frequent_numbers = [number for number, frequency in frequency_dict.items() if frequency == max_frequency]

    return most_frequent_numbers[0]

# Returns the prediction of the image
# model_bin = used to classify cars and others
# model_1,model_2,model_3 are the three models used to predict the view 
# using majority voting algorithm
def classify_image(model_1,model_2,model_3,image):

    # Load and preprocess the image
    image = cv2.resize(image, (128, 128))
    image_array = image / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    '''
    Making predictions using the majority voting
    Getting predictions from each model

    '''
    pred_1 = model_1.predict(image_array)
    pred_2 = model_2.predict(image_array)
    pred_3 = model_3.predict(image_array)
    # Perform majority voting
    max_1 = np.argmax(pred_1)
    max_2 = np.argmax(pred_2)
    max_3 = np.argmax(pred_3)
    predicted_ans = find_most_frequent_number(max_1, max_2, max_3)
    class_sum = np.sum([pred_1, pred_2, pred_3], axis=0)
    prob_sum = class_sum[0][predicted_ans]
    return predicted_ans,prob_sum

# Used for extracting different frames from our video
# Returns 2 arrays classlists1 which contains all the frames and classlists2 which 
# contains the predicted images of each class
# video_path =  path of video stored
# sample_interval = interval between each frame to be considered
def classify_video_frames(video_path,dir,sample_interval):
    # Three Multi-class Classification models for Majority Voting for Exterior Views
    model_ext1 = tf.keras.models.load_model('../aiml/models/sr_vgg16.keras')
    model_ext2 = tf.keras.models.load_model('../aiml/models/hl_inceptionresnetv2.keras')
    model_ext3 = tf.keras.models.load_model('../aiml/models/vv_mobilenetv2.keras')

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = int(sample_interval * fps)  # Number of frames to skip based on sample_interval
    
    frame_index = 0
    save_dir = 'static/frames_ext'
    alt_dir = dir+'frames_ext'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(alt_dir, exist_ok=True)
    class_lists1 = []
    class_lists2 = []
    frame_arr = [-1 for _ in range(11)]
    frame_prob_sum = [-1e8] * 11
    i=0
    while frame_index < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Set the frame index

        ret, frame = video.read()
        
        if not ret:
            break
        prediction,prob_sum = classify_image(model_ext1,model_ext2,model_ext3, frame)  # Call your classify_image function here
        if(prediction>=0 and prediction<=10 and frame_prob_sum[prediction]<prob_sum):
          frame_arr[prediction] = i
          frame_prob_sum[prediction]=prob_sum
        frame_path = os.path.join(save_dir, f'frame_{i}.jpg')
        alt_path = os.path.join(alt_dir, f'frame_{i}.jpg')
        # frame = cv2.resize(frame, (200,200))
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(alt_path, frame)
        class_lists1.append(f'frame_{i}.jpg')

        # Skipping frame_index based on frame_duration
        frame_index += frame_duration
        i=i+1
    video.release()
    check = [0 for _ in range(4)]
    for j in range(11):
        if frame_arr[j]<0:
            continue
        class_lists2.append([CATEGORIES[j],f'frame_{frame_arr[j]}.jpg'])
        if j==0:
            check[j]=1
            weights_file = "../aiml/models/model_front_parts/best.pt"
            image_file = f"static/frames_ext/frame_{frame_arr[j]}.jpg"
            command = f"python3 ../aiml/yolov5/detect.py --weights {weights_file} --source {image_file} --img 416 --conf 0.3 --save-txt"
            subprocess.run(command, shell=True)
        elif j==1:
            check[j]=1
            weights_file = "../aiml/models/model_rear_parts/best.pt"
            image_file = f"static/frames_ext/frame_{frame_arr[j]}.jpg"
            command = f"python3 ../aiml/yolov5/detect.py --weights {weights_file} --source {image_file} --img 416 --conf 0.3 --save-txt"
            subprocess.run(command, shell=True)
        elif j==2:
            check[j]=1
            weights_file = "../aiml/models/model_left_parts/best.pt"
            image_file = f"static/frames_ext/frame_{frame_arr[j]}.jpg"
            command = f"python3 ../aiml/yolov5/detect.py --weights {weights_file} --source {image_file} --img 416 --conf 0.5 --save-txt"            
            subprocess.run(command, shell=True)
        elif j==3:
            check[j]=1
            weights_file = "../aiml/models/model_right_parts/best.pt"
            image_file = f"static/frames_ext/frame_{frame_arr[j]}.jpg"
            command = f"python3 ../aiml/yolov5/detect.py --weights {weights_file} --source {image_file} --img 416 --conf 0.5 --save-txt"
            subprocess.run(command, shell=True)

    return class_lists1,class_lists2,check,frame_arr

# Function returns an array containing parts of each view and its corresponding images
# Input is the path of image, path of labels file which contains the coordinates and a
# variable which determines the view
def classify_parts(img_path,annotation_path,c,dir):
  if c==0:
      classes_parts = classes_front
      save_dir = 'static/front'
      alt_dir = dir+'front'
  elif c==1:
      classes_parts = classes_rear
      save_dir = 'static/rear'
      alt_dir = dir+'rear'
  elif c==2:
      classes_parts = classes_left
      save_dir = 'static/left'
      alt_dir = dir+'left'
  elif c==3:
      classes_parts = classes_right
      save_dir = 'static/right'
      alt_dir = dir+'right'
  num_classes = len(classes_parts)
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(alt_dir, exist_ok=True)
  # Load the image
  image = cv2.imread(img_path)

  # Create separate lists for each class
  parts = []

  # Load the annotation file
  with open(annotation_path, "r") as f:
      annotations = f.readlines()
  vis = [0 for _ in range(num_classes)]
  # Process each annotation
  for annotation in annotations:
      class_label, x, y, width, height = map(float, annotation.split())

      # Convert class label to integer
      class_label = int(class_label)

      # Extract the coordinates
      x_min = int((x - width / 2) * image.shape[1])
      y_min = int((y - height / 2) * image.shape[0])
      x_max = int((x + width / 2) * image.shape[1])
      y_max = int((y + height / 2) * image.shape[0])
      

      # Extract the region of interest
      image_2 = cv2.imread(img_path)
      roi = image_2[y_min:y_max, x_min:x_max]
      frame_path = os.path.join(save_dir, f'{class_label}.jpg')
      alt_path = os.path.join(alt_dir, f'{class_label}.jpg')
      roi = cv2.resize(roi, (200,200))
      cv2.imwrite(frame_path, roi)
      cv2.imwrite(alt_path, roi)
      row = [classes_parts[class_label],f'{class_label}.jpg']
      parts.append(row)
        

  return parts

# Function doesnot return anything. It makes sure the left and rights are detected 
# by checking their coordinates
def relabel_annotations(input_file_path,left,right):
    annotations = []
    label_4_lines = []
    label_5_lines = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                class_label, x_center, y_center, width, height = map(float, line.split())
                label = int(class_label)

                annotations.append((label, x_center, y_center, width, height))
                if label == left:
                    label_4_lines.append((x_center, line))
                elif label == right:
                    label_5_lines.append((x_center, line))

    if label_4_lines:
        min_x_center_label_4 = min(label_4_lines, key=lambda x: x[0])[0]
        for i in range(len(annotations)):
            if annotations[i][0] == left and annotations[i][1] != min_x_center_label_4:
                annotations[i] = (right, *annotations[i][1:])
    if label_5_lines:
        max_x_center_label_5 = max(label_5_lines, key=lambda x: x[0])[0]
        for i in range(len(annotations)):
            if annotations[i][0] == right and annotations[i][1] != max_x_center_label_5:
                annotations[i] = (left, *annotations[i][1:])

    with open(input_file_path, 'w') as output_file:
        for annotation in annotations:
            line = ' '.join(str(value) for value in annotation)
            output_file.write(line + '\n')

# Function returns the array containg the parts and the corresponding images
# It takes a variable c as input which determines the view of the image
def get_output_label_file_path(frame_arr,c,p,dir):

    # Get a list of all subdirectories matching the pattern '../aiml/yolov5/runs/detect/exp*'
    subdirectories = glob.glob('../aiml/yolov5/runs/detect/exp*')

    # Sort the subdirectories based on their suffix (i.e., the numerical part of the folder name)
    sorted_subdirectories = sorted(subdirectories, key=lambda x: int(x.split('exp')[-1]) if x.split('exp')[-1].isdigit() else -1)
    latest_folder = sorted_subdirectories[p]
    label_path = os.path.join(latest_folder, 'labels', f"frame_{frame_arr[c]}.txt")
    img_path = f'static/frames_ext/frame_{frame_arr[c]}.jpg'
  
    if c==0:
        relabel_annotations(label_path,4,5)
    elif c==1:
        relabel_annotations(label_path,1,3)
    elif c==2:
        relabel_annotations(label_path,1,3)
    elif c==3:
        relabel_annotations(label_path,3,1)

    parts = classify_parts(img_path,label_path,c,dir)
    return parts

MAXFILESIZE=30*1024*1024*30
app = Flask(__name__)
# Custom error handler
@app.errorhandler(404)
def not_found_error(error):
    response = app.response_class(
        response=json.dumps({'status':'Failure','data':'Invalid file extension. Only mp4 files are allowed.'}, sort_keys=False),
        mimetype='application/json'
    )
    response.status_code = 404
    return response

# Main function
@app.route('/', methods=['POST', 'GET'])
def exterior():
    if request.method == 'POST':
        file = request.files['video']
        file_extension = os.path.splitext(file.filename)[1]
        file_size = len(file.read())
        file.seek(0)
        if(file_extension != '.mp4'):
            return render_template("exterior.html",Statement='Invalid file extension. Only mp4 files are allowed.')
        if(file_size > MAXFILESIZE):
            return render_template("exterior.html",Statement='File size exceeds the allowed limit')
        # Save the video in your project directory by mentioning the path
        # Path may vary for different computer
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date = str(current_date)
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=8))
        dir = date+'/'+str(res)+'/'
        os.makedirs(dir, exist_ok=True)
        file.save('' + file.filename)
        file.save(dir + file.filename)  

        path = '' + file.filename
        sample_interval = 1  # Set the desired sampling interval in seconds
        class_lists1,class_lists2,check,frame_arr = classify_video_frames(path,dir,sample_interval)
        d=-1
        for i in range(4):
            if check[3-i]==1:
                check[3-i]=d
                d=d-1
        if check[0]<0:
           front = get_output_label_file_path(frame_arr,0,check[0],dir)
        else:
            front = []
        if check[1]<0:
           rear = get_output_label_file_path(frame_arr,1,check[1],dir)
        else:
            rear=[]
        if check[2]<0:
           left = get_output_label_file_path(frame_arr,2,check[2],dir)
        else:
            left=[]
        if check[3]<0:
           right = get_output_label_file_path(frame_arr,3,check[3],dir)
        else:
            right=[]
        # On getting a post request it will render a file image.html which displays all the views
        return render_template("image.html",answers=class_lists1,outputs=class_lists2,front=front,rear=rear,left=left,right=right)
    else:
        # Initial page index.html
        return render_template("exterior.html",Statement='')

############################    INTERIOR    ################################
CATEGORIES1 = ["Rear_View_Mirror", "Rear_Seat_Covers", "Power_Windows", "Front_Seat_Covers", "Floor_Mats", "Central_Lock", "Dashboard", "Door_Panel"]

# Returns the most frequent prediction in the combination of three Models
# max1 = prediction of first model
# max2 = prediction of second model
# max3 = prediciton of third model
def find_most_frequent_number1(max_1, max_2, max_3):

    frequency_dict = {max_1: 0, max_2: 0, max_3: 0}
    frequency_dict[max_1] += 1
    frequency_dict[max_2] += 1
    frequency_dict[max_3] += 1

    max_frequency = max(frequency_dict.values())

    most_frequent_numbers = [number for number, frequency in frequency_dict.items() if frequency == max_frequency]

    return most_frequent_numbers[0]

# Returns the prediction of the image
# model_1,model_2,model_3 are the three models used to predict the view 
# using majority voting algorithm
def classify_image1(model_1,model_2,model_3,image):

    # Load and preprocess the image
    image = cv2.resize(image, (128, 128))
    image_array = np.expand_dims(image, axis=0)  # Add batch dimension

    
    # Making predictions using the majority voting
    # Getting predictions from each model

    pred_1 = model_1.predict(image_array)
    pred_2 = model_2.predict(image_array)
    pred_3 = model_3.predict(image_array)

    # Perform majority voting
    max_1 = np.argmax(pred_1)
    max_2 = np.argmax(pred_2)
    max_3 = np.argmax(pred_3)
    predicted_ans = find_most_frequent_number1(max_1, max_2, max_3)
    class_sum = np.sum([pred_1, pred_2, pred_3], axis=0)
    prob_sum = class_sum[0][predicted_ans]
    return predicted_ans,prob_sum

# Used for extracting different frames from our video
# Returns 2 arrays classlists1 which contains all the frames and classlists2 which 
# contains the predicted images of each class
# video_path =  path of video stored
# sample_interval = interval between each frame to be considered
def classify_video_frames1(video_path, sample_interval,dir):
    # Three Multi-class Classification models for Majority Voting for Exterior Views
    model_int1 = tf.keras.models.load_model('../aiml/models/rmodel_resnet50.keras')
    model_int2 = tf.keras.models.load_model('../aiml/models/rmodel_resnet152.keras')
    model_int3 = tf.keras.models.load_model('../aiml/models/rmodel_vgg16.keras')

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = int(sample_interval * fps)  # Number of frames to skip based on sample_interval
    
    frame_index = 0
    save_dir = 'static/frames_int'
    alt_dir = dir+'frame_int'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(alt_dir, exist_ok=True)
    class_lists1 = []
    class_lists2 = []
    frame_arr = [-1 for _ in range(8)]
    frame_prob_sum = [-1e8] * 8
    i=0
    while frame_index < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Set the frame index

        ret, frame = video.read()
        
        if not ret:
            break
        prediction,prob_sum = classify_image1(model_int1,model_int2,model_int3, frame)  # Call your classify_image function here
        if(prediction>=0 and prediction<=7 and frame_prob_sum[prediction]<prob_sum):
          frame_arr[prediction] = i
          frame_prob_sum[prediction]=prob_sum
        frame_path = os.path.join(save_dir, f'frame_{i}.jpg')
        alt_path = os.path.join(alt_dir, f'frame_{i}.jpg')
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(alt_path, frame)
        class_lists1.append(f'frame_{i}.jpg')
        # Skipping frame_index based on frame_duration
        frame_index += frame_duration
        i=i+1
    video.release()
    for j in range(8):
        if frame_arr[j]<0:
            continue
        class_lists2.append([CATEGORIES1[j],f'frame_{frame_arr[j]}.jpg'])
    return class_lists1,class_lists2

@app.route('/interior', methods=['POST', 'GET'])
def interior():
    if request.method == 'POST':
        file = request.files['video']
        file_extension = os.path.splitext(file.filename)[1]
        file_size = len(file.read())
        file.seek(0)
        if(file_extension != '.mp4'):
            return render_template("interior.html",Statement='Invalid file extension. Only mp4 files are allowed.')
        if(file_size > MAXFILESIZE):
            return render_template("interior.html",Statement='File size exceeds the allowed limit')
        # Save the video in your project directory by mentioning the path
        # Path may vary for different computer
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date = str(current_date)
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=8))
        dir = date+'/'+str(res)+'/'
        os.makedirs(dir, exist_ok=True)
        file.save('' + file.filename)
        file.save(dir + file.filename)
        path = '' + file.filename
        sample_interval = 1  # Set the desired sampling interval in seconds
        class_lists1,class_lists2 = classify_video_frames1(path, sample_interval,dir)

        # On getting a post request it will render a file image.html which displays all the views
        return render_template("picture.html",answers=class_lists1,outputs=class_lists2)
    else:
        # Initial page index.html
        return render_template("interior.html",Statement='')


#testing api in postman
# Exterior
@app.route('/ext_img', methods=['POST', 'GET'])
def ext_print():
    if request.method == 'POST':
        file = request.files['file']
        file_extension = os.path.splitext(file.filename)[1]
        file_size = len(file.read())
        file.seek(0)
        if(file_extension != '.mp4'):
            abort(404)
        if(file_size > MAXFILESIZE):
            response = app.response_class(
            response=json.dumps({'status':'Failure','data':'File size exceeds the allowed limit'}, sort_keys=False),
            mimetype='application/json'
            )
            return response
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date = str(current_date)
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=8))
        dir = date+'/'+str(res)+'/'
        os.makedirs(dir, exist_ok=True)
        file.save('' + file.filename)
        file.save(dir + file.filename)  

        path = '' + file.filename
        sample_interval = 1  # Set the desired sampling interval in seconds
        class_lists1,class_lists2,check,frame_arr = classify_video_frames(path,dir,sample_interval)
        d=-1
        for i in range(4):
            if check[3-i]==1:
                check[3-i]=d
                d=d-1
        if check[0]<0:
           front = get_output_label_file_path(frame_arr,0,check[0],dir)
        else:
            front = []
        if check[1]<0:
           rear = get_output_label_file_path(frame_arr,1,check[1],dir)
        else:
            rear=[]
        if check[2]<0:
           left = get_output_label_file_path(frame_arr,2,check[2],dir)
        else:
            left=[]
        if check[3]<0:
           right = get_output_label_file_path(frame_arr,3,check[3],dir)
        else:
            right=[]
        response = app.response_class(
        response=json.dumps({'status':'Success','views_data':class_lists2,
                             'front_data':front,'rear_data':rear,'left_data':left,
                             'right_data':right}, sort_keys=False),
        mimetype='application/json'
        )
        return response

# Interior
@app.route('/int_img', methods=['POST', 'GET'])
def int_print():
    if request.method == 'POST':
        file = request.files['file']
        file_extension = os.path.splitext(file.filename)[1]
        file_size = len(file.read())
        file.seek(0)
        if(file_extension != '.mp4'):
            abort(404)
        if(file_size > MAXFILESIZE):
            response = app.response_class(
            response=json.dumps({'status':'Failure','data':'File size exceeds the allowed limit'}, sort_keys=False),
            mimetype='application/json'
            )
            return response
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date = str(current_date)
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=8))
        dir = date+'/'+str(res)+'/'
        os.makedirs(dir, exist_ok=True)
        file.save('' + file.filename)
        file.save(dir + file.filename)
        path = '' + file.filename
        sample_interval = 1  # Set the desired sampling interval in seconds
        class_lists1,class_lists2 = classify_video_frames1(path, sample_interval,dir)
        response = app.response_class(
        response=json.dumps({'status':'Success','views_data':class_lists2
                             }, sort_keys=False),
        mimetype='application/json'
        )
        return response

if __name__ == "__main__":
    app.run(debug=True)
