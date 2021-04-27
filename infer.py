
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from config import MODEL_PATH
from Model import VehicleColorModel

# Setting Up the Labels 
labels = ['black', 'blue' , 'cyan' , 'gray' , 'green' , 'red' , 'white' , 'yellow']
def decode_label(index):
    return  labels[index]

def encode_label_from_path(path):
    for index,value in enumerate(labels):
        if value in path:
            return  index

model = VehicleColorModel()
model.load_state_dict(torch.load(MODEL_PATH , map_location=torch.device("cpu")))
transforms = Compose([Resize(224), CenterCrop(224), ToTensor()])

def infer(image): 
	image = transforms(image)
	image = image.unsqueeze(0)
	pred = model.forward(image).argmax(dim = 1)
	class_label = decode_label(pred)
	return class_label
	

if __name__ == "__main__":
	image = Image.open('/path/to/an/image').convert('RGB')
	print(infer(image))
