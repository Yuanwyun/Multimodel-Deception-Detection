import os
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
import torch.nn as nn
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
 
        
class FSDATA(Dataset):
    def __init__(self, annotations_file, img_dir, spec_dir,num_frames):
        self.annos = pd.read_csv(annotations_file)  # columns as [file_name, label]
        self.img_dir = img_dir  # face_frames_folder with video clips as sub folders
        self.spec_dir = spec_dir  
        self.frame_size = 224 # any multiple of 32
        self.transforms_face = T.Compose([
                                    T.ToTensor(),
                                    T.Resize(224, antialias=True),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        self.transforms_image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   #统一标准化
        self.number_of_target_frames = num_frames

    def __len__(self):        #多少个图
        return len(self.annos)  #return 长度
    
    def __getitem__(self, idx):   
        file_name = self.annos.iloc[idx, 0]  # clip_name 
        split = file_name.split('_')  #每遇见一次下划线，就分割  eg. document_name_time_...,, output is document, name , time
        file_path = self.img_dir + split[0] + '_' + split[1] + '/' + split[2] + '_' + split[3] + '/face_frames/'
        spec_path = self.spec_dir + '/' +  file_name +'.png'
        image = read_image(spec_path).float()/255.0
        image = self.transforms_image(image)
        # list all images (no numpy facial affect features) in the video_clip folder
        frame_names = [ i for i in os.listdir(file_path) if i.split('.')[-1] == 'jpg']  
        
        # sample exactly 64 images 
        target_frames = np.linspace(start=0,stop=len(frame_names)-1,num=self.number_of_target_frames,dtype=np.int32)

        # now apply transforms and stack all images together
        face_frames = []
        for i in target_frames:
            img = np.asarray(Image.open(file_path + frame_names[i]))/ 255.0
            face_frames.append(self.transforms_face(img))

        face_frames = torch.stack(face_frames, 0) # shape = (64,3,224,224) = (num_frames, 3 channels, H, W)

        face_frames = face_frames.permute(1,0,2,3) # include this line if you want to use the slow_r50 model in pytorch
        # it swaps the dimensions to (3 channels, num_frames, H, W)
        
        # labels
        label = self.annos.iloc[idx, 1]       
        label = torch.tensor(label)

        return face_frames.float(),image, label
    
train_dataset = FSDATA(
                    annotations_file="path", 
                    spec_dir="/spectrograms/",
                    img_dir="CropFace_MTCNN/",
                    num_frames=64
                    )

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for spec, labels in train_loader:
    print(spec.shape, labels.shape)
for face_frames, labels in train_loader:
    print(face_frames.shape, labels.shape)
    break

#################################################################################################################


device = torch.device("cuda:0")

batch_size = 8
num_epochs = 10
protocols = [['train1.csv','test1.csv'],['train2.csv','test2.csv'],['train3.csv','test3.csv']]



class ResNet(nn.Module):
    def __init__(self,):
        super(ResNet, self).__init__()

        r50 = torchvision.models.resnet50(pretrained=True)
        r50.fc = nn.Linear(r50.fc.in_features,2) 
        self.resnet = r50

    def forward(self,x):
        features = self.resnet(x)
        return features


# 处理face的模型：r18_3d
class r18_3d(nn.Module):
    def __init__(self, ):
        super(r18_3d, self).__init__()
        r3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        r3d.blocks[5].proj = torch.nn.Identity()
        # for p in r3d.parameters(): p.requires_grad = False
        self.cnn_3d = r3d  
        self.classifier = nn.Sequential(
         nn.Flatten(),
         nn.Linear(2048,2),
         )

    def forward(self, x):
        x = x.float() # for face frames
        return self.classifier(self.cnn_3d(x))


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()


        r50 = torchvision.models.resnet50(pretrained=True)
        r50.fc = nn.Linear(r50.fc.in_features, 2)
        self.resnet = r50

        # Create the r18_3d model
        r3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        r3d.blocks[5].proj = torch.nn.Identity()
        r3d.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 2)
        )
        self.cnn_3d = r3d

    def forward(self, x_resnet, x_r18_3d):
        # Forward pass through the ResNet model
        resnet_features = self.resnet(x_resnet)
        # Forward pass through the r18_3d model
        r18_3d_output = self.cnn_3d(x_r18_3d)

        # For example, you can concatenate the features or perform other operations
        combined_features = torch.cat((resnet_features, r18_3d_output), dim=1)

        return combined_features

# Example of using the combined model


        
def train_one_epoch(train_data_loader,model,optimizer,loss_fn):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0   
    model.train()
    for face, image, labels in train_data_loader:       
        face = face.to(device)
        image = image.to(device)
        labels = labels.to(device) 
        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(image,face)
        _loss = loss_fn(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)      
        #Backward
        _loss.backward()
        optimizer.step()
        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)

    acc = round(sum_correct_pred/total_samples,4)*100
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, acc

def val_one_epoch(val_data_loader, model,loss_fn):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
      for face,image, labels in val_data_loader:       
        face = face.to(device)
        image = image.to(device)
        labels = labels.to(device)

        preds = model(image,face)
        _loss = loss_fn(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)
    acc = round(sum_correct_pred/total_samples,4)*100
    epoch_loss = np.mean(epoch_loss)    
    return epoch_loss, acc


annotations_file="train1.csv"
img_dir="CropFace_MTCNN/"
spec_dir="spectrograms/"

num_frames = 64
fullpath ='path/'

for P in protocols:

    print("\n\nCurrent protocol.....................", P)
    train,test = P  

    train_dataset = FSDATA(fullpath + train , img_dir, spec_dir, num_frames)
    
    test_dataset = FSDATA(fullpath + test ,img_dir, spec_dir, num_frames)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, num_workers=16)
    print("\t Dataset Loaded")
    

    model = CombinedModel()
    model = torch.nn.DataParallel(model)
    model.to(device)
    print("\t Model Loaded")
    
# Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)
#Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []

    for epoch in range(num_epochs):
        print('\tEpoch...........',epoch+1)      
        ###Training
        loss, acc = train_one_epoch(train_loader,model,optimizer,loss_fn)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn)
        best_val_acc.append(val_acc)

    print("\n\tBest Accuracy........", round(np.max(np.asarray(best_val_acc)),2))
