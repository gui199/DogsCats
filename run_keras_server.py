# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import model_from_json
from flask import Flask, request, render_template, jsonify, abort, url_for, redirect
from PIL import Image
import numpy as np
import io


# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None
MODELJSON = './checkpoint/model_X.json'
MODELJSON_WEIGHTS = './checkpoint/model_X_weights.h5'


def load_modelo():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	# load json and create model
	with open(MODELJSON, 'r') as json_file:
		loaded_model_json = json_file.read()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(MODELJSON_WEIGHTS)


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image


def its_cat_or_dog(predict):
	''' Função que recebe uma probabilidade
	e retorna uma string com o rótulo indicado.'''
	print(predict[0])
	if predict[0] > 0.5:
		return "is a dog"
	else:
		return "is a cat"


@app.route("/")
@app.route("/base")
def base():
	# página de boas vindas
    return render_template('dashboard.html')


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST":
		if request.files.get("image"):
			# read the image in PIL format
			image = request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = its_cat_or_dog(preds)

			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			r = {"label": results, "probability": float(preds[0])}
			data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))
	#load_modelo()
    #iniciar aplicativo
	app.run(host='0.0.0.0',)
