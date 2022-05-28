print("fr.py file")
DEBUG = True
def dprint(*args):
    if DEBUG:
        print(args)

print("check check")
import sys
dprint("[++++] importing traceback")
import traceback

dprint("[++++] python version >>> ")
dprint(sys.version)

dprint("IMPORTING NUMPY")
try:
    import numpy as np
except Exception as e:
    dprint(e)

dprint("IMPORTED NUMPY!! ", np.__version__)


dprint("IMPORTING CV2")
import cv2
dprint("IMPORTED CV2!! ", cv2.__version__)


dprint("IMPORTING scipy")
import scipy
dprint("IMPORTED scipy!! ", scipy.__version__)


dprint("IMPORTING Tesnorflow")
import tensorflow as tf
dprint("IMPORTED Tesnorflow!! ", tf.__version__)


dprint("IMPORTING mediapipe")
import mediapipe as mp
dprint("IMPORTED mediapipe!! ")


import os
import cv2
import numpy as np
import uuid
import json
import time
import traceback
dprint("[++++] importing faceDetector")
from detection.detectorMain import faceDetector
dprint("[++++] importing faceAlignment")
from alignment.alignmentMain import faceAlignment
dprint("[++++] importing get_embeddings, is_match")
try:
    from recognition.recoginationMain import get_embeddings, is_match
except Exception as e:
    dprint(e)

dprint("[++++] importing VGGFace")
from recognition.vggfaceTF import VGGFace

dprint("HELLO SIR !!")


# -------------------------------------------------------------------
# LOADING SAVED MODELS
# -------------------------------------------------------------------

def getDetectorModel(config_dir):
    dprint(f"[+++] DETECTOR CALL HUA HAI with config dir name : {config_dir} ")
    DETECTOR_CONFIG = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/cfg/detector_yolov3-face.cfg") 
    dprint()
    DETECTOR_MODEL_WEIGHTS = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/model-weights/detector_yolov3-wider_16000.weights")
    dprint(f"[++++] detector config path: {DETECTOR_CONFIG}")
    dprint(f"[++++] detector weight path: {DETECTOR_MODEL_WEIGHTS}")
    try:
        detector_net = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG, DETECTOR_MODEL_WEIGHTS)
    except Exception as e:
        dprint(f"[++++] cannot load darknet becox of this :: ", e)
        return
    dprint(f"[++++] darknet loaded from OpenCv")
    detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    dprint(f"[++++] darknet backend set")
    detector_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    dprint(f"[++++] darknet target set")
    try:
        dprint("[++++] doing 1st inference here")
        frame = np.zeros((416,416,3)).astype(np.uint8)
        detector_response = faceDetector(frame, detector_net)
    except Exception as e:
        dprint("[++++] some exception in detection!")
        dprint(e)
    dprint(f"[++++] returning object : {detector_net}")
    return detector_net


def getEmbeddingModel(config_dir):
    try:

        # dprint("[++++] EMBEDDING CALL HUA HAI")
        # weights_path = os.path.join(config_dir,"recognition/model-weights/rcmalli_vggface_tf_notop_resnet50.h5")
        
        # dprint("[++++] loading resNet50, weigts_path: ", weights_path)
        # embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg',weights_path=weights_path)
        # dprint("[++++] loaded resNet50")
        
        model_path = os.path.join(config_dir,"embedding_model")
        dprint("[++++] model_path: ", model_path)
        embedding_model = tf.keras.models.load_model(model_path)
        dprint("[++++] loaded embedding model")
        # # doing 1st inference here
        dprint("[++++] doing 1st inference here")
        frame = np.zeros((224,224,3)).astype(np.uint8)
        dprint("[++++] passing image of zeros, frame:", type(frame), frame.shape)
        current_embedding = get_embeddings(frame, embedding_model)  
        dprint("[++++] got embedding of shape: ", current_embedding.shape)
        

    except Exception as e:
        dprint("something went wrong :( ", e)
        pass
    dprint("[++++] _obj_reference_counts_dict: ", embedding_model._obj_reference_counts_dict)
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
    dprint("[++++] inside align function with cropped face image shape: ", cropped_face.shape)
    try:
        alligned_face, eye_coords_wrt_original = faceAlignment(cropped_face, cropped_cord)
        dprint("[++++] face aligned")
        
        # dprint("hardcoding values ... ")
        # alligned_face = cv2.resize(cropped_face, (224,224))
        # eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
        # dprint("hardcoded values ... ")
    except Exception as e:
        dprint("[++++] something went wrong")
        alligned_face = [[[]]]
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    dprint("returning ...")
    return alligned_face, eye_coords_wrt_original

# -------------------------------------------------------------------
# GET EMBEDDINGS FOR SINGLE IMAGE.
# -------------------------------------------------------------------
def getEmbeddings(opencv_image, embedding_model, detector_net):
    dprint("[++++] inside function getEmbeddings")
    try:
        cropped_face, cropping_cord = detect(opencv_image, detector_net)
        dprint("[++++] detected face")
        alligned_face, eye_coords_wrt_original = align(cropped_face, cropping_cord)
        dprint("[++++] aligned face, alligned_face: ", type(alligned_face), alligned_face.shape)
        current_embedding = get_embeddings(alligned_face, embedding_model)[0]   
        dprint("[++++] got embeddings")
    except Exception as e:
        dprint("[++++] something went wrong here")
        dprint(e)
        current_embedding = [-1] * 2048
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    return current_embedding, eye_coords_wrt_original


# -------------------------------------------------------------------
# FROM NIST -- MULTIPLE IMAGES OF SINGLE FACE
# -------------------------------------------------------------------
def create_template_multiple_images_single_face(image_tuple):
    try:
        dprint("[++++] fr.py file >> create_template_multiple_images_single_face")
        dprint(f'[++++] image_tuple length:  {len(image_tuple)}')
        if len(image_tuple) == 5:
            dprint("[++++] let's begin")
            template_role = image_tuple[0]
            image_datas = image_tuple[1]

            detector_net = image_tuple[2]

            # embedding_model = image_tuple[3]
            # dprint("[++++] size of model recieved from c++", sys.getsizeof(object))
            ############ image_tuple[4] contains configDir->str
            config_dir = image_tuple[4]
            embedding_model = getEmbeddingModel(config_dir)

            dprint("[++++] unpacked tuple.")
            embeddings_list = []
            eyeCoords_list = []
            dprint(f"[++++] TemplateRole: {type(template_role)}")
            dprint(f"[++++] TemplateRole: {template_role}")

            dprint(f"[++++] image_datas type: {type(image_datas)}")
            dprint(f"[++++] image_datas length: {len(image_datas)}")
            
            dprint(f"[++++] detector_net: {type(detector_net)}")
            dprint(f"[++++] embed_model: {type(embedding_model)}")
            
            dprint("[++++] Iterating through all the images")
            for image_data in image_datas:
                try:
                    dprint("[++++] unpacking metadata")
                    image_metadata = image_data[0]
                    w = image_metadata[1]
                    h = image_metadata[2]
                    depth = image_metadata[3]
                    channel = int(depth / 8)
                    dprint("[++++] unpacking imagedata")                
                    image = image_data[1]
                    dprint("[++++] creating numpy array")
                    image = np.array(image)
                    opencv_image = image.reshape((h,w,channel)).astype(np.uint8)
                    dprint("[++++] created numpy images, getting embeddigs")
                    embedding, eyeCoords = getEmbeddings(opencv_image, embedding_model, detector_net)
                    dprint("[++++] got embeddings and eye coordinates")
                    if not embedding[0] == -1:  #hardcode check to check if embeddings are correcct or not.
                        embeddings_list.append(embedding) 
                except Exception as e:
                    dprint('[++++] some exception, returning default embeddings')
                    dprint(e)
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
        dprint("[++++] some excpetion in function: ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # dprint(exc_type, fname, exc_tb.tb_lineno)
        dprint(traceback.print_exc())



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
# dprint(resp)


    # DEBUG = False
    # if DEBUG:
    #     dprint('create_template_multiple_images_single_face => input info ...')
    #     dprint("PYTHON: 1 : ", type(image_tuple)) # tuple 
    #     dprint("PYTHON: 2 : ", len(image_tuple))  # of length 2
    #     dprint("PYTHON: 3 : ", type(image_tuple[0])) # integer :: templete role
    #     dprint("PYTHON: 3.1 : ", image_tuple[0])  # having tempalte role 0/1/2/3 -> we use 0 and 1 only :: 
    #     dprint("PYTHON: 4 : ", type(image_tuple[1])) # tuple
    #     dprint("PYTHON: 4.1 : ", len(image_tuple[1])) # of length as #images passed of same person ...
    #     dprint("PYTHON: 5.1 : ", type(image_tuple[1][0]), len(image_tuple[1][0])) # tuple of length 2 having metadata for first image 
    #     dprint("PYTHON: 6.1 : ", type(image_tuple[1][0][0])) # tuple of length 4 -> label, width, height, depth 
    #     dprint("PYTHON: 6.2 : ", len(image_tuple[1][0][0])) # tuple of length 4 -> label, width, height, depth 
    #     dprint("PYTHON: 7.1 : ", type(image_tuple[1][0][1])) # tuple        
    #     dprint("PYTHON: 7.2 : ", len(image_tuple[1][0][1])) # image data as 1D array ...
    
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
        dprint('create_template_single_images_mutiple_faces => input info ...')
        dprint('10.1', type(image_tuple)) # tuple
        dprint('10.2', len(image_tuple)) # of length 2
        dprint('11.1', type(image_tuple[0])) # int
        dprint('11.2', image_tuple[0]) # with TemplateRole
        dprint('12.1', type(image_tuple[1])) # tuple 
        dprint('12.2', len(image_tuple[1])) # of length 1(always) 
        dprint('13.1', type(image_tuple[1][0])) # tuple
        dprint('13.2', len(image_tuple[1][0])) # of length 2, 0th for meta, 1st for iamge-data
        dprint('14.1', image_tuple[1][0][0]) # tuple of length 4 having meta -> label,width,height,depth
        dprint('14.2', type(image_tuple[1][0][1])) # tuple
        dprint('14.3', len(image_tuple[1][0][1])) # of length of image-data (w*h*(depth/8))
        
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


# if __name__ == '__main__':
#     weights_path = os.path.join("","recognition/model-weights/rcmalli_vggface_tf_notop_resnet50.h5")
#     dprint("[++++] loading resNet50, weigts_path: ", weights_path)
#     embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg',weights_path=weights_path)
#     dprint("[++++] saving model", )
#     embedding_model.save('embedding_model')
#     dprint("[++++] saced model" )



