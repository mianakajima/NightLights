from random import randrange
import glob
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from model import CNN


classes = {0: 'Class 1',
           1: 'Class 2',
           2: 'Class 3',
           3: 'Class 4',
           4: 'Class 5'}

def get_random_image_path(class_name: str, num_items=10):
    """Returns path of random image in specified class"""

    random_image_number = randrange(0, num_items)
    images = glob.glob(f'static/scaled_images/{class_name}/*.jpg')

    chosen_path = images[random_image_number]

    return chosen_path

def transform_image(image_path):

    image = Image.open(image_path)
    # from training data
    image_mean = torch.tensor(0.0277)
    image_std = torch.tensor(0.0504)

    transform = transforms.Compose(
        [transforms.Resize(size=(430, 430)),
         transforms.Grayscale(1),
         transforms.ToTensor(),
         transforms.Normalize(image_mean, image_std)
         ]
    )

    transformed_image = transform(image)

    return transformed_image


def get_naive_bayes_prediction(image):
    """Returns class prediction of image using Naive Bayes"""

    nb_model = pickle.load(open('static/models/gnb_model.sav', 'rb'))
    sum_image_pixels = torch.sum(transform_image(image)).reshape(-1, 1)
    prediction = nb_model.predict(sum_image_pixels)

    predicted_class = classes[prediction[0]]

    return predicted_class



def get_CNN_prediction(im_path):
    """ Returns class prediction of image using trained CNN model"""
    CNN_model = CNN()
    CNN_model.load_state_dict(torch.load('static/models/lights_inital2.pth'))
    CNN_model.eval()

    transformed_image = transform_image(im_path).reshape((1, 1, 430, 430))

    model_output = CNN_model(transformed_image)
    prediction = torch.argmax(model_output, 1)

    predicted_class = classes[prediction.item()]

    return predicted_class

