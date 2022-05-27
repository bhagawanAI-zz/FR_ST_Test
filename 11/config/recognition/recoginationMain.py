# face verification with the VGGFace2 model
from scipy.spatial.distance import cosine
import numpy as np


from recognition.utils import preprocess_input


def get_embeddings(alligned_face, model):
    	
	samples = alligned_face[:,:,::-1].astype(np.float32)
	print("[++++] samples size: ", samples.shape)
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input([samples], version=2)	
	print("[++++] processed samples size: ", samples.shape)
	try:
		print("[++++] performing prediction")
		print("[++++] model type: ", type(model))
		# print("[++++] model dir(): ", dir(model))
		yhat = model.predict(samples, verbose=2)
	except Exception as e:
		print("[++++] keras exception ", e)
	return yhat


 
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return {"isMatch": 1, "matchScore":score, "thresh":thresh}
	else:
		return {"isMatch": 0, "matchScore":score, "thresh":thresh}