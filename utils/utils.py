import os
from tqdm import tqdm
import pickle
from mtcnn import MTCNN
import cv2

detector = MTCNN()

def extract_face(image):
    face = detector.detect_faces(image)
    faces_num = len(face)
    resized_face_crop = 0
    if faces_num != 0:
    	box = face[0]['box']
    
    if faces_num != 0:
    	if not((box[0] < 0) or (box[1] < 0) or (box[2] < 0) or (box[3] < 0)): 
    		face_crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    		face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    		resized_face_crop = cv2.resize(face_crop, (220, 220))
    	else:
    		faces_num = 0
    else:
        faces_num = 0
    return faces_num, resized_face_crop


def make_dict(data_dir):
    cwd = os.getcwd()
    os.chdir(data_dir)
    classes = os.listdir()
    face_dict = dict()
    for folder in tqdm(classes):
        face_dict[str(folder)] = os.listdir('./' + str(folder))
    os.chdir(cwd)
    f = open("./data/face_dict.pkl", "wb")
    pickle.dump(face_dict, f)
    f.close()
    return face_dict
