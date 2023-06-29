import cv2
import time
import torch
import numpy as np
import hdf5storage as hd

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


net_num = int( input("(option: 0, 1, 2, 3)Train on CUDA: ") )
flag_all = int( input("0 for half and 1 for full: ") )

####################################################################################################

msf_np   = np.load('../dataset/msf_f.npy')               
pan_np   = np.load('../dataset/pan_f.npy')               
lbl_np   = hd.loadmat("../dataset/label.mat")['label'].astype('uint8')  
lbl_np   = lbl_np-1
row = lbl_np.shape[0]
col = lbl_np.shape[1]

print('The shape and dtype of origin msf and pan are:')
print(msf_np.dtype, msf_np.shape)
print(pan_np.dtype, pan_np.shape)
print("\n")

####################################################################################################

Patch_size = 32                               
img_fill = int(Patch_size/2)
Interpolation = cv2.BORDER_REFLECT_101
msf_np = cv2.copyMakeBorder(msf_np, img_fill, img_fill, img_fill, img_fill, Interpolation)
pan_np = cv2.copyMakeBorder(pan_np, img_fill, img_fill, img_fill, img_fill, Interpolation)

msf_np = msf_np.transpose( (2,0,1) )
pan_np = pan_np.transpose( (2,0,1) )

print('The shape and dtype of msf and pan are:')
print(msf_np.dtype, msf_np.shape)
print(pan_np.dtype, pan_np.shape)
print("\n")

label_element, element_count = np.unique(lbl_np, return_counts=True) 
kind_sum  = len(label_element) - 1
label_sum = sum(element_count)
print('Total number of categories: ', kind_sum)
print('Total number of samples:', label_sum)
print('Total number of samples for each category: ')
for i in range(kind_sum+1):
    print(str(label_element[i]).ljust(3), str(element_count[i]).ljust(8), "%.1f%%"%(100*element_count[i]/label_sum))
print("\n")

####################################################################################################
class MyData(Dataset):
    def __init__(self, msf, pan, xyz):
        self.msf = msf
        self.pan = pan
        self.xyz = xyz


    def __getitem__(self, index):
        x, y, z = self.xyz[index]
        msf_x, msf_y = img_fill+2*x  , img_fill+2*y
        pan_x, pan_y = img_fill+2*x  , img_fill+2*y

        image_msf = self.msf[:, msf_x-img_fill:msf_x+img_fill, msf_y-img_fill:msf_y+img_fill].astype('float32')
        image_pan = self.pan[:, pan_x-img_fill:pan_x+img_fill, pan_y-img_fill:pan_y+img_fill].astype('float32')


        return image_msf, image_pan, (x, y)

    def __len__(self):
        return len(self.xyz)


####################################################################################################


all_labeled_xyz = np.array([ (index[0], index[1], z) for index, z in np.ndenumerate(lbl_np) ])

if not flag_all:
    all_labeled_xyz = all_labeled_xyz[all_labeled_xyz[:,2]!=255,:]
	


all_dataset = MyData(msf_np, pan_np, all_labeled_xyz)
all_iter = DataLoader(all_dataset, batch_size=64, shuffle=False, num_workers = 32)


####################################################################################################

class RBlock(nn.Module):
    def __init__(self, ch, com):
        super().__init__()
        self.ch        = ch
        self.com       = com
        self.margin    = 0.3

        self.com_conv  = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_conv0 = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_conv0 = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.com_bn1   = nn.BatchNorm2d(2*ch)
        self.com_bn2   = nn.BatchNorm2d(2*ch)

        self.msf_conv1 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_bn1   = nn.BatchNorm2d(2*ch)
        self.msf_conv2 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_bn2   = nn.BatchNorm2d(2*ch)

        self.pan_conv1 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_bn1   = nn.BatchNorm2d(2*ch)
        self.pan_conv2 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_bn2   = nn.BatchNorm2d(2*ch)


        self.extra1 = nn.Sequential(
            nn.Conv2d(ch, 2*ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*ch)
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(ch, 2*ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*ch)
        )
          
    def forward(self, msf, pan):
        if self.com==1:
            msf_mid   = self.com_bn1( self.com_conv(msf)  )
            pan_mid   = self.com_bn2( self.com_conv(pan)  )
        else:
            msf_mid   = self.com_bn1( self.msf_conv0(msf) )
            pan_mid   = self.com_bn2( self.pan_conv0(pan) )

        msf_self  = self.msf_bn1( self.msf_conv1(msf_mid) )
        msf_other = self.msf_bn2( self.msf_conv2(msf_mid) )

        pan_self  = self.pan_bn1( self.pan_conv1(pan_mid) )
        pan_other = self.pan_bn2( self.pan_conv2(pan_mid) )

        M_pos = torch.pow( torch.sum( torch.pow( nn.Flatten()(msf_other - pan_self), 2) , axis=1 ), 0.5)
        M_neg = torch.pow( torch.sum( torch.pow( nn.Flatten()(msf_other - msf_self), 2) , axis=1 ), 0.5)
        P_pos = torch.pow( torch.sum( torch.pow( nn.Flatten()(pan_other - msf_self), 2) , axis=1 ), 0.5)
        P_neg = torch.pow( torch.sum( torch.pow( nn.Flatten()(pan_other - pan_self), 2) , axis=1 ), 0.5)

        msf_loss = torch.mean(  F.relu( M_pos-M_neg+self.margin )  ) 
        pan_loss = torch.mean(  F.relu( P_pos-P_neg+self.margin )  ) 

        return  F.relu(0.5*msf_self+0.5*pan_other+self.extra1(msf)),  F.relu(0.5*pan_self+0.5*msf_other+self.extra2(pan)), 0.5*(msf_loss+pan_loss)

class YBlock(nn.Module):
    def __init__(self, ch, com):
        super().__init__()
        self.ch        = ch
        self.com       = com
        self.margin    = 0.3

        self.com_conv  = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_conv0 = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_conv0 = nn.Conv2d(ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.com_bn1   = nn.BatchNorm2d(2*ch)
        self.com_bn2   = nn.BatchNorm2d(2*ch)

        self.msf_conv1 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_bn1   = nn.BatchNorm2d(2*ch)
        self.msf_conv2 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.msf_bn2   = nn.BatchNorm2d(2*ch)

        self.pan_conv1 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_bn1   = nn.BatchNorm2d(2*ch)
        self.pan_conv2 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1)
        self.pan_bn2   = nn.BatchNorm2d(2*ch)

        self.extra1 = nn.Sequential(
            nn.Conv2d(ch, 3*ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(3*ch)
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(ch, 3*ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(3*ch)
        )

    def forward(self, msf, pan):
        if self.com==1:
            msf_mid   = self.com_bn1( self.com_conv(msf) )
            pan_mid   = self.com_bn2( self.com_conv(pan) )
        else:
            msf_mid   = self.com_bn1( self.msf_conv0(msf) )
            pan_mid   = self.com_bn2( self.pan_conv0(pan) )

        msf_self  = self.msf_bn1( self.msf_conv1(msf_mid) )
        msf_other = self.msf_bn2( self.msf_conv2(msf_mid) )

        pan_self  = self.pan_bn1( self.pan_conv1(pan_mid) )
        pan_other = self.pan_bn2( self.pan_conv2(pan_mid) )


        M_pos_0 = torch.pow( msf_other - pan_self, 2)
        M_neg_0 = torch.pow( msf_other - msf_self, 2)
        P_pos_0 = torch.pow( pan_other - msf_self, 2)
        P_neg_0 = torch.pow( pan_other - pan_self, 2)

        M_pos = torch.pow( torch.sum( nn.Flatten()(M_pos_0) ,axis=1), 0.5)
        M_neg = torch.pow( torch.sum( nn.Flatten()(M_neg_0) ,axis=1), 0.5)
        P_pos = torch.pow( torch.sum( nn.Flatten()(P_pos_0) ,axis=1), 0.5)
        P_neg = torch.pow( torch.sum( nn.Flatten()(P_neg_0) ,axis=1), 0.5)

        msf_loss   =  torch.mean(  F.relu(M_pos-M_neg+self.margin)  )  
        pan_loss   =  torch.mean(  F.relu(P_pos-P_neg+self.margin)  )

        ############################## COMIX ##############################

        M_dif = M_neg_0-M_pos_0
        M_dif = torch.sum(torch.sum(M_dif, axis=-1), axis=-1)
        P_dif = P_neg_0-P_pos_0
        P_dif = torch.sum(torch.sum(P_dif, axis=-1), axis=-1)
    
        _, M_indices = M_dif.topk(k=self.ch, dim=1, largest=True, sorted=True)
        _, P_indices = P_dif.topk(k=self.ch, dim=1, largest=True, sorted=True)

        
        msf_other_common_list = []
        msf_other_diffrt_list = []
        pan_self_common_list  = []
        pan_self_diffrt_list  = []

        pan_other_common_list = []
        pan_other_diffrt_list = []
        msf_self_common_list  = []
        msf_self_diffrt_list  = []

        for j in range(msf_self.shape[0]):
            M_ini = M_indices[j].tolist()
            M_out = list(  set(range(2*self.ch))-set(M_ini)  )
            msf_other_common_list.append( msf_other[j, M_ini ,:,:] )
            msf_other_diffrt_list.append( msf_other[j, M_out ,:,:] )
            pan_self_common_list .append(  pan_self[j, M_ini ,:,:] )
            pan_self_diffrt_list .append(  pan_self[j, M_out ,:,:] )

            P_ini = P_indices[j].tolist()
            P_out = list(  set(range(2*self.ch))-set(P_ini)  )
            pan_other_common_list.append( pan_other[j, P_ini ,:,:] )
            pan_other_diffrt_list.append( pan_other[j, P_out ,:,:] )
            msf_self_common_list .append(  msf_self[j, P_ini ,:,:] )
            msf_self_diffrt_list .append(  msf_self[j, P_out ,:,:] )


        msf_other_common = torch.stack(msf_other_common_list, dim=0)
        msf_other_diffrt = torch.stack(msf_other_diffrt_list, dim=0)
        pan_self_common  = torch.stack( pan_self_common_list, dim=0)
        pan_self_diffrt  = torch.stack( pan_self_diffrt_list, dim=0)
        
        pan_other_common = torch.stack(pan_other_common_list, dim=0)
        pan_other_diffrt = torch.stack(pan_other_diffrt_list, dim=0)
        msf_self_common  = torch.stack( msf_self_common_list, dim=0)
        msf_self_diffrt  = torch.stack( msf_self_diffrt_list, dim=0)


        msf_fusion = torch.cat( (msf_self_diffrt, 0.5*(msf_self_common+pan_other_common), pan_other_diffrt) ,dim=1) 
        pan_fusion = torch.cat( (pan_self_diffrt, 0.5*(pan_self_common+msf_other_common), msf_other_diffrt) ,dim=1)
            
        return F.relu(msf_fusion+self.extra1(msf)),  F.relu(pan_fusion+self.extra2(pan)), 0.5*(msf_loss+pan_loss)
        

####################################################################################################

class YNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk1 = YBlock(4  ,0)
        self.blk2 = RBlock(12 ,0)
        self.blk3 = YBlock(24 ,0)
        self.blk4 = RBlock(72 ,1)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(),
            nn.Linear(288, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )


    def forward(self, msf0, pan0):                         
        msf1, pan1, loss1 = self.blk1(msf0, pan0)
        msf2, pan2, loss2 = self.blk2(msf1, pan1)
        msf3, pan3, loss3 = self.blk3(msf2, pan2)
        msf4, pan4, loss4 = self.blk4(msf3, pan3)


        feature = torch.cat((msf4, pan4), dim=1)
        out = self.fc(feature)

        return out, loss1+loss2+loss3+loss4

####################################################################################################

# Visualize
cnn = torch.load('best953831.pkl')
cnn.cuda(net_num)
cnn.eval()
class_count = np.zeros(kind_sum)
out_clour = np.zeros((row, col, 3))


def clour_model(cnn, all_data_loader):
    last_time = time.time()
    for step, (ms4, pan, gt_xy) in enumerate(all_data_loader):
        if step%10000==0:
            print(step, end=', ')
            print("%.2f"%(time.time()-last_time))
            last_time = time.time()

        with torch.no_grad():
            ms4 = ms4.cuda(net_num)
            pan = pan.cuda(net_num)
            output = cnn(ms4, pan)
            pred_y = torch.max(output[0], 1)[1].cuda(net_num).data.squeeze()
            pred_y_numpy = pred_y.cpu().numpy()

        color = [
            [ 255, 255, 0   ],
            [ 255, 0  , 0   ],
            [ 33 , 145, 237 ],
            [ 201, 252, 189 ],
            [ 0  , 0  , 230 ],
            [ 0  , 255, 0   ],
            [ 240, 32 , 160 ],
            [ 221, 160, 221 ],
            [ 140, 230, 240 ],
            [ 0  , 255, 255 ]
        ] 

        for k in range(len(pred_y_numpy)):
            kind = pred_y_numpy[k]
            class_count[kind]+=1
            out_clour[gt_xy[0][k]][gt_xy[1][k]] = color[kind]

    print(class_count)
    cv2.imwrite("10%s.png"%('_full' if flag_all else '_half'), out_clour)

clour_model(cnn,  all_iter)
