# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "https://work-in-my-machine.herokuapp.com/predict"
IMAGE_PATH = './test_dog.jpg'


def main():
	# load the input image and construct the payload for the request
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}

	# submit the request	
	r = requests.post(KERAS_REST_API_URL, files=payload)

	# ensure the request was sucessful
	if r.status_code == 200:
		# loop over the predictions and display them
		# for (i, result) in enumerate(r["predictions"]):
		# 	print("{}. {}: {:.4f}".format(i + 1, result["label"],
		# 		result["probability"]))
		print(r.json())

	elif r.status_code == 500:
		print("App failed.")

	# otherwise, the request failed
	else:
		print("Request failed.")


if __name__ == '__main__':
    main()
