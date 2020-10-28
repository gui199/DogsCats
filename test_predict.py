
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import VGG16
from keras.models import model_from_json
# caregamento de constantes
MODELJSON = './checkpoint/model_X.json'
MODELJSON_WEIGHTS = './checkpoint/model_X_weights.h5'
IMAGE_PATH = './test_dog.jpg'

# make a prediction for a new image.

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image(IMAGE_PATH)
	# load json and create model
	with open(MODELJSON, 'r') as json_file:
		loaded_model_json = json_file.read()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(MODELJSON_WEIGHTS)
	# predict the class
	result = model.predict(img)
	print(result)
	if result[0] > 0.5:
		print("is a dog")
	else:
		print("is a cat")


if __name__ == '__main__':
    run_example()
