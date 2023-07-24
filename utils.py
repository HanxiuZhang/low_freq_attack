import cv2
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torchattacks import FGSM, PGD, CW, AutoAttack
from torch_dct import dct, idct

def read_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img).cuda()
    return img_tensor

def show_tensor(tensor,width=200,height=200):
    print(tensor.size())
    tensor_img = transforms.ToPILImage()(tensor)
    display(tensor_img.resize((width,height)))

def combine_model(model,input_size):
    c_model = nn.Sequential(transforms.Resize(input_size,antialias=True),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      model)
    return c_model

def predict_1_img(img_tensor,model):
    res = model(img_tensor.unsqueeze(0).cuda())
    label = res[0].argmax().item()
    prob = F.softmax(res,dim=1)[0].max().item()
    return label, prob

def attack_1_img(model,img_tensor,label,atk_name,**kwargs):
    atk = eval(atk_name)(model,**kwargs)
    adv_img_tensor = atk(img_tensor.unsqueeze(0).cuda(),labels = torch.Tensor([label]).cuda().type(torch.LongTensor))
    return adv_img_tensor[0]

def low_mask(input,n):
    res = torch.zeros_like(input)
    res[:,:n,:n] = input[:,:n,:n]
    return res

def low_freq_mask(input,n):
    return idct(low_mask(dct(input),512))