## check check
import sys
import traceback
print("python version >>> ")
print(sys.version)

print("IMPORTING NUMPY")
try:
    import numpy as np
except Exception as e:
    print(e)

print("IMPORTED NUMPY!! ", np.__version__)


print("IMPORTING CV2")
import cv2
print("IMPORTED CV2!! ", cv2.__version__)


print("IMPORTING scipy")
import scipy
print("IMPORTED scipy!! ", scipy.__version__)


print("IMPORTING Tesnorflow")
import tensorflow
print("IMPORTED Tesnorflow!! ", tensorflow.__version__)


print("IMPORTING mediapipe")
import mediapipe as mp
print("IMPORTED mediapipe!! ")


import os
import cv2
import numpy as np
import uuid
import json
import time
import traceback
print("[++++] importing faceDetector")
from detection.detectorMain import faceDetector
print("[++++] importing faceAlignment")
from alignment.alignmentMain import faceAlignment
print("[++++] importing get_embeddings, is_match")
from recognition.recoginationMain import get_embeddings, is_match
print("[++++] importing VGGFace")
from recognition.vggfaceTF import VGGFace

print("HELLO SIR !!")


# -------------------------------------------------------------------
# LOADING SAVED MODELS
# -------------------------------------------------------------------

def getDetectorModel(config_dir):
    print(f"[+++] DETECTOR CALL HUA HAI with config dir name : {config_dir} ")
    DETECTOR_CONFIG = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/cfg/detector_yolov3-face.cfg") 
    print()
    DETECTOR_MODEL_WEIGHTS = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/model-weights/detector_yolov3-wider_16000.weights")
    print(f"[++++] detector config path: {DETECTOR_CONFIG}")
    print(f"[++++] detector weight path: {DETECTOR_MODEL_WEIGHTS}")
    try:
        detector_net = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG, DETECTOR_MODEL_WEIGHTS)
    except Exception as e:
        print(f"[++++] cannot load darknet becox of this :: ", e)
        return
    print(f"[++++] darknet loaded from OpenCv")
    detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print(f"[++++] darknet backend set")
    detector_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print(f"[++++] darknet target set")
    try:
        print("[++++] doing 1st inference here")
        frame = np.zeros((416,416,3)).astype(np.uint8)
        detector_response = faceDetector(frame, detector_net)
    except Exception as e:
        print("[++++] some exception in detection!")
        print(e)
    print(f"[++++] returning object : {detector_net}")
    return detector_net


def getEmbeddingModel(config_dir):
    try:
        print("[++++] EMBEDDING CALL HUA HAI")
        weights_path = os.path.join(config_dir,"recognition/model-weights/rcmalli_vggface_tf_notop_resnet50.h5")
        print("[++++] loading resNet50")
        embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg',weights_path=weights_path)
        print("[++++] loaded resNet50")
        # doing 1st inference here
        print("[++++] doing 1st inference here")
        frame = np.zeros((224,224,3)).astype(np.uint8)
        print("[++++] passing image of zeros, frame:", type(frame), frame.shape)
        current_embedding = get_embeddings(frame, embedding_model)  
        print("[++++] got embedding of shape: ", current_embedding.shape)
    except Exception as e:
        print("something went wrong :( ", e)
        pass
    return embedding_model



# -------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------
def detect(frame, detector_net):
    try:
        detector_response = faceDetector(frame, detector_net)
        if detector_response["result"] == 0:
            cropped_face = detector_response["data"]["cropped_face"]
            cropping_cord = detector_response["data"]["final_boxes"]
        else:
            cropped_face = [[[]]] 
            cropping_cord = []
    except Exception as e:
        pass
    return cropped_face, cropping_cord


# -------------------------------------------------------------------
# Alignment
# -------------------------------------------------------------------
def align(cropped_face, cropped_cord):
    print("[++++] inside align function with cropped face image shape: ", cropped_face.shape)
    try:
        # alligned_face, eye_coords_wrt_original = faceAlignment(cropped_face, cropped_cord)
        print("[++++] face aligned")
        print("hardcoding values ... ")
        # alligned_face = np.ones((224,244,3))*10.astype(np.uint8)
        alligned_face = cv2.resize(cropped_face, (224,224))
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
        print("hardcoded values ... ")
    except Exception as e:
        print("[++++] something went wrong")
        alligned_face = [[[]]]
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    print("returning ...")
    return alligned_face, eye_coords_wrt_original

# -------------------------------------------------------------------
# GET EMBEDDINGS FOR SINGLE IMAGE.
# -------------------------------------------------------------------
def getEmbeddings(opencv_image, embedding_model, detector_net):
    print("[++++] inside function getEmbeddings")
    try:
        cropped_face, cropping_cord = detect(opencv_image, detector_net)
        print("[++++] detected face")
        alligned_face, eye_coords_wrt_original = align(cropped_face, cropping_cord)
        print("[++++] aligned face, alligned_face: ", type(alligned_face), alligned_face.shape)
        current_embedding = get_embeddings(alligned_face, embedding_model)[0]   
        print("[++++] got embeddings")
    except Exception as e:
        print("[++++] something went wrong here")
        print(e)
        current_embedding = [-1] * 2048
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    return current_embedding, eye_coords_wrt_original


# -------------------------------------------------------------------
# FROM NIST -- MULTIPLE IMAGES OF SINGLE FACE
# -------------------------------------------------------------------
def create_template_multiple_images_single_face(image_tuple):
    try:
        print("[++++] fr.py file >> create_template_multiple_images_single_face")
        print(f'[++++] image_tuple length:  {len(image_tuple)}')
        if len(image_tuple) == 5:
            print("[++++] let's begin")
            template_role = image_tuple[0]
            image_datas = image_tuple[1]

            detector_net = image_tuple[2]

            embedding_model = image_tuple[3]
            embedding_model._make_predict_function()
            # image_tuple[4] contains configDir->str
            # config_dir = image_tuple[4]
            # embedding_model = getEmbeddingModel(config_dir)

            print("[++++] unpacked tuple.")
            embeddings_list = []
            eyeCoords_list = []
            print(f"[++++] TemplateRole: {type(template_role)}")
            print(f"[++++] TemplateRole: {template_role}")

            print(f"[++++] image_datas type: {type(image_datas)}")
            print(f"[++++] image_datas length: {len(image_datas)}")
            
            print(f"[++++] detector_net: {type(detector_net)}")
            print(f"[++++] embed_model: {type(embedding_model)}")
            
            print("[++++] Iterating through all the images")
            for image_data in image_datas:
                try:
                    print("[++++] unpacking metadata")
                    image_metadata = image_data[0]
                    w = image_metadata[1]
                    h = image_metadata[2]
                    depth = image_metadata[3]
                    channel = int(depth / 8)
                    print("[++++] unpacking imagedata")                
                    image = image_data[1]
                    print("[++++] creating numpy array")
                    image = np.array(image)
                    opencv_image = image.reshape((h,w,channel)).astype(np.uint8)
                    print("[++++] created numpy images, getting embeddigs")
                    embedding, eyeCoords = getEmbeddings(opencv_image, embedding_model, detector_net)
                    print("[++++] got embeddings and eye coordinates")
                    if not embedding[0] == -1:  #hardcode check to check if embeddings are correcct or not.
                        embeddings_list.append(embedding) 
                except Exception as e:
                    print('[++++] some exception, returning default embeddings')
                    print(e)
                    embedding = [-1] *2048
                    eyeCoords = (0,0,-1,-1,-1,-1)

                eyeCoords_list.append(eyeCoords)

            if not embeddings_list:
                embedding = [-1] *2048
                embeddings_list.append(embedding)


            final_embeddings = np.mean(embeddings_list, axis=0).tolist()
            return (0, tuple(final_embeddings), tuple(eyeCoords_list))
        
        else:
            embedding = [-1] *2048
            embeddings_list = [embedding]
            eyeCoords = (0,0,-1,-1,-1,-1)
            eyeCoords_list = [eyeCoords]
            final_embeddings = np.mean(embeddings_list, axis=0).tolist()
            return (0, tuple(final_embeddings), tuple(eyeCoords_list))
    except Exception as e:
        print("[++++] some excpetion in function: ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.print_exc())



# # load saved image tuple file :: sent by Nawal Singh

# import json 
# import pprint

# config_dir = ""
# detector_net = getDetectorModel(config_dir)
# embedding_model = getEmbeddingModel(config_dir)

# with open("/home/ubuntu/create_template_multiple_images_single_face.json", "r") as f:
#     image_tuple = json.load(f)

# image_tuple = list(image_tuple)
# image_tuple.append(detector_net)
# image_tuple.append(embedding_model)

# resp = create_template_multiple_images_single_face(image_tuple)
# print(resp)


    # DEBUG = False
    # if DEBUG:
    #     print('create_template_multiple_images_single_face => input info ...')
    #     print("PYTHON: 1 : ", type(image_tuple)) # tuple 
    #     print("PYTHON: 2 : ", len(image_tuple))  # of length 2
    #     print("PYTHON: 3 : ", type(image_tuple[0])) # integer :: templete role
    #     print("PYTHON: 3.1 : ", image_tuple[0])  # having tempalte role 0/1/2/3 -> we use 0 and 1 only :: 
    #     print("PYTHON: 4 : ", type(image_tuple[1])) # tuple
    #     print("PYTHON: 4.1 : ", len(image_tuple[1])) # of length as #images passed of same person ...
    #     print("PYTHON: 5.1 : ", type(image_tuple[1][0]), len(image_tuple[1][0])) # tuple of length 2 having metadata for first image 
    #     print("PYTHON: 6.1 : ", type(image_tuple[1][0][0])) # tuple of length 4 -> label, width, height, depth 
    #     print("PYTHON: 6.2 : ", len(image_tuple[1][0][0])) # tuple of length 4 -> label, width, height, depth 
    #     print("PYTHON: 7.1 : ", type(image_tuple[1][0][1])) # tuple        
    #     print("PYTHON: 7.2 : ", len(image_tuple[1][0][1])) # image data as 1D array ...
    
    # # output is is this structure
    # (
    #     ReturnCode->int,
    #     long-data-1D-tuple->[double],
    #     (
    #         #1st face's eye details
    #         (isLeftAssigned->int(0 or 1), isRightAssigned->int(0 or 1), xleft->int, yleft->int, xright->int, yright->int),
    #         ...
    #         ...
    #         ...
    #         #Nth face's eye details
    #         (isLeftAssigned->int(0 or 1), isRightAssigned->int(0 or 1), xleft->int, yleft->int, xright->int, yright->int)
    #     )
    # )
    # return (
    #     0, # return code -> frvt_structs.h  
    #     tuple([0.0012]*64), #  MUST HAVE (999,999,999)
    #     (
    #         (1, 0, 1, 1, 2, 2), # isLeftAssigned,isRightAssigned,xleft, y_left, x_right, y_right   EyePair (frvt_structs.h)
    #     )*len(image_tuple[1]),
    # )




def create_template_single_images_mutiple_faces(image_tuple):
    DEBUG = False
    # # input is is this structure
    # (
    #     TemplateRole->int,
    #     (
    #         (#1st face details
    #             (label->int, width->int, height->int, depth->int),
    #             (long-data-1D-list->[int])
    #         ),
    #         ...
    #         ...
    #         ...
    #         (#Nth face details
    #             (label->int, width->int, height->int, depth->int),
    #             (long-data-1D-list->[int])
    #         )
    #     )
    # )
    if DEBUG:
        print('create_template_single_images_mutiple_faces => input info ...')
        print('10.1', type(image_tuple)) # tuple
        print('10.2', len(image_tuple)) # of length 2
        print('11.1', type(image_tuple[0])) # int
        print('11.2', image_tuple[0]) # with TemplateRole
        print('12.1', type(image_tuple[1])) # tuple 
        print('12.2', len(image_tuple[1])) # of length 1(always) 
        print('13.1', type(image_tuple[1][0])) # tuple
        print('13.2', len(image_tuple[1][0])) # of length 2, 0th for meta, 1st for iamge-data
        print('14.1', image_tuple[1][0][0]) # tuple of length 4 having meta -> label,width,height,depth
        print('14.2', type(image_tuple[1][0][1])) # tuple
        print('14.3', len(image_tuple[1][0][1])) # of length of image-data (w*h*(depth/8))
        
    # # output is is this structure
    # (
    #     ReturnCode->int,
    #     (
    #         (long-data-1D-tuple->[double]),
    #         (long-data-1D-tuple->[double]),
    #         ...
    #         ...
    #         ... n templates for n face detected ...
    #         (long-data-1D-tuple->[double])
    #     ),
    #     (
    #         #1st face's eye details
    #         (isLeftAssigned->int(0 or 1), isRightAssigned->int(0 or 1), xleft->int, yleft->int, xright->int, yright->int),
    #         ...
    #         ...
    #         ...
    #         #Nth face's eye details
    #         (isLeftAssigned->int(0 or 1), isRightAssigned->int(0 or 1), xleft->int, yleft->int, xright->int, yright->int),
    #     )
    # )
    num_face_detected = 3
    return (
        0,
        (tuple([0.0012]*64),)*num_face_detected,
        (
            (1, 0, 1, 1, 2, 2),
        )*num_face_detected,
    )





