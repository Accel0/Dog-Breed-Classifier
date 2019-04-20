# Dog-Breed-Classifier
**This project was made for Udacity Nano degree Deep Learning**
This project can be used to classifiy dog breeds and even detect humans faces!

# Requirements:
This project is written in Python so you need some packages for it to run:
1. torch
2. torchvision
3. numpy
4. matplotlib
5. cv2
6. tqdm
7. PIL
8. collections




# Steps:
1. First you have to download the images data dog images [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and the human images data [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
2. Load the images and included their location on your device ```import numpy as np
from glob import glob
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))```

3. Run the face_detector ```def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0```
4. Run the Dog_detector ```def dog_detector(img_path):
    if(VGG16_predict(img_path) >= 151 and VGG16_predict(img_path) <= 268):
        is_a_dog = True
    else:
        is_a_dog = False
    return is_a_dog # true/false
dog_detector('/data/dog_images/train/001.Affenpinscher/Affenpinscher_00001.jpg')```
5. Run the Data Loaders for the images in Step 4 in the notebook
6. Load the model ```import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
model_transfer = models.vgg19(pretrained = True)
for param in model_transfer.parameters():
    param.requires_grad = False```
7. Load your own classifier to use in the model and change the output to 133
`classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(25088, 1000, bias = True)),
                                        ("relu1", nn.ReLU()),
                                        ("Dropout", nn.Dropout(0.3)),
                                        ("fc2", nn.Linear(1000, 500, bias = True)),
                                        ("relu2", nn.ReLU()),
                                        ("Dropout2", nn.Dropout(0.3)),
                                        ("output", nn.Linear(500, 133, bias = True))]))
model_transfer.classifier = classifier
for param in model_transfer.classifier.parameters():
    param.requires_grad = True
if use_cuda:
    model_transfer = model_transfer.cuda()
print(model_transfer)`
and then load the criterion and the optimizer
`criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr = 0.01)
print(model_transfer) `
9. Train the model and the best model will be saved
10. Run predict_breed_transfer and see how it performs on your own dog image ```class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]
def predict_breed_transfer(img_path):
    image = Image.open(img_path)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    img = transform(image)
    img = img.unsqueeze(0)
    if use_cuda:
        img = img.cuda()
    model_transfer.eval()
    input_img = model_transfer(img)
    if use_cuda:
        input_img = input_img.cpu()
    output = input_img.data.numpy().argmax()
    return class_names[output]
predict_breed_transfer('/data/dog_images/train/001.Affenpinscher/Affenpinscher_00001.jpg')```
11. Run the run_app and give it a picture in your own device and see how good it performs! ```def run_app(img_path):
    if(dog_detector(img_path)):
        print("Hello Dog! I think you are a {}".format(predict_breed_transfer(img_path)))
    elif(face_detector(img_path)):
        print("Hello Human! I think you look like a {}".format(predict_breed_transfer(img_path)))
    else:
        print("Error I am sorry but you dont look like a dog or a human\n Maybe the picture is corrputed?")
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()```
