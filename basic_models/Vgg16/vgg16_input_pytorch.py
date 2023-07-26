import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import sys
import numpy as np

from torchsummary import summary as summary

model = models.vgg16(pretrained=True)

input_image = Image.open("bus.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
sys.stdout = open('input_bus1.txt','w')
a = input_tensor.tolist()
for i in range(224):
    for j in range(224):
        print(a[2][i][j])
        print(a[1][i][j])
        print(a[0][i][j])
sys.stdout.close()
# input_batch = torch.unsqueeze(input_tensor, 0) # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)

# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)


# # Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top_prob, top_catid = torch.topk(probabilities, 1)
# print(categories[top_catid[0]])

# summary(model, (3,224,224))
