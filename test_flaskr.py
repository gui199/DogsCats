import unittest
import run_keras_server
from PIL import Image
import numpy as np
import io
import flask
import simple_request
#import cv2

IMAGE_TEST = './test_dog.jpg'

# Código para ignorar o warning/aviso
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class FlaskrTestCase(unittest.TestCase):

    def setUp(self):
        ''' Iniciar o App do Flask '''
        run_keras_server.app.testing = True
        self.app = run_keras_server.app.test_client()

    def test_home(self):
        ''' Checar Home '''
        result = self.app.get('/')
        # Make your assertions
        self.assertEquals(result.status, "200 OK")

    def test_base(self):
        ''' Checar Base '''
        result = self.app.get('/base')
        # Make your assertions
        self.assertEquals(result.status, "200 OK")

    def test_load_model(self):
        ''' Testar se o modelo carrega '''
        result = run_keras_server.load_modelo()
        self.assertEqual(result, None)  # result != None

    def test_prepare_image(self):
        ''' Testar a preparação de imagens '''
        # image = flask.request.files[IMAGE_TEST].read()
        # image = Image.open(io.BytesIO(image))
        # image = cv2.imread(IMAGE_TEST)
        image = Image.open(IMAGE_TEST)
        result = run_keras_server.prepare_image(image, target=(224, 224))
        image.close()

    def test_predict_url(self):
        ''' Testar se a url existe '''
        response = self.app.post('/predict')
        self.assertEquals(response.status, "200 OK")

    def test_predict_send_image(self):
        ''' Testar a predição de imagens '''
        # image = open(IMAGE_TEST, "rb").read()
        image = Image.open(IMAGE_TEST)
        payload = {"image": image}
        response = self.app.post('/predict', data=payload,
                                content_type='multipart/form-data',)
        self.assertEquals(response.status, "200 OK")
        image.close()

if __name__ == '__main__':
    unittest.main()
