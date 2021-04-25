
from torchvision.transforms  import ToTensor, Resize, CenterCrop , Compose
import torch.nn as nn
import torch
from Model import VehicleColorModel
from PIL import Image

# Setting Up the Labels 
labels = ['black', 'blue' , 'cyan' , 'gray' , 'green' , 'red' , 'white' , 'yellow']
def decode_label(index):
    return  labels[index]

def encode_label_from_path(path):
    for index,value in enumerate(labels):
        if value in path:
            return  index

# print(torch.cuda.is_available())
MODEL_PATH = '/home/jeetu/Project/VehicleColorA/Exp1/model_3.pt'
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
	image = Image.open('/home/jeetu/Desktop/AA.jpg').convert('RGB')
	print(infer(image))
