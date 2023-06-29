import cv2
import time
import torch
import random
import datetime
import numpy as np
import hdf5storage as hd

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



net_num = int( input("(option: 0, 1, 2, 3)Train on CUDA: ") )
max_result = [0, 0, 0] #OA, AA, KAPPA
max_OA = [0, 0, 0]   
max_AA = [0, 0, 0]   
max_KP = [0, 0, 0]  


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

all_labeled_xyz = []
for x in range(row):
    for y in range(col):
        if lbl_np[x][y] != 255:
            all_labeled_xyz.append( ( x, y, lbl_np[x][y] ) )


random.shuffle(all_labeled_xyz)
Train_Rate = 0.01
Train_Num  = int( Train_Rate*len(all_labeled_xyz) ) +1  # +1 to round up

train_xyz = all_labeled_xyz[:Train_Num]
test_xyz = all_labeled_xyz[Train_Num:]


####################################################################################################

look_xyz = np.array(train_xyz)

label_element, element_count = np.unique(look_xyz[:,2], return_counts=True)
kind_sum  = len(label_element)
label_sum = sum(element_count)
print('Total number of train categories: ', kind_sum)
print('Total number of train samples:', label_sum)
print('Total number of train samples for each category: ')
for i in range(kind_sum):
    print(str(label_element[i]).ljust(3), str(element_count[i]).ljust(8), "%.1f%%"%(100*element_count[i]/label_sum))
print("\n")

####################################################################################################
# supply to balance
'''
supply_num = [40, 50, 0, 0, 40, 0, 0, 0, 0, 50, 0]
for xyz in test_xyz:
    x, y, z = xyz
    if supply_num[z]>0:
        supply_num[z]-=1
        train_xyz.append(xyz)
    if sum(supply_num)==0:
        break
'''

####################################################################################################
# after supply
'''
look_xyz = np.array(train_xyz)

label_element, element_count = np.unique(look_xyz[:,2], return_counts=True) 
kind_sum  = len(label_element)
label_sum = sum(element_count)
print('Total number of supplied categories: ', kind_sum)
print('Total number of supplied samples:', label_sum)
print('Total number of supplied samples for each category: ')
for i in range(kind_sum):
    print(str(label_element[i]).ljust(3), str(element_count[i]).ljust(8), "%.1f%%"%(100*element_count[i]/label_sum))
print("\n")
'''



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
        image_lbl = z.astype('int64')

        return image_msf, image_pan, image_lbl

    def __len__(self):
        return len(self.xyz)

train_dataset = MyData(msf_np, pan_np, train_xyz)
test_dataset  = MyData(msf_np, pan_np, test_xyz)

bz = 64
train_iter = DataLoader(train_dataset, batch_size=bz, shuffle=False, num_workers = 32)
test_iter  = DataLoader(test_dataset , batch_size=bz, shuffle=False, num_workers = 32)

print("Trianing rate is %.2f, batch size is %d"%(Train_Rate, bz))
print("%d batch for trian, %d batch for test" %(len(train_iter),len(test_iter)))
print("\n")


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


for A, B, y in train_iter:
    a, b = YNet()(A,B)
    print(a.shape, b.dtype)
    break

look = YNet()
total = sum([param.nelement() for param in look.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
del look

####################################################################################################

def batch_correct(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp   = y_hat.type(y.dtype) == y
    return cmp.sum().item()


def test_model(net, test_iter): 
    net.eval() 
    Mat = np.zeros((kind_sum, kind_sum))
    now_c = 0

    for A, B, y in test_iter:
        A, B, y = A.cuda(net_num), B.cuda(net_num), y.cuda(net_num)
        now_c  +=1
        ########################################################
        y_hat   = (net(A, B)[0]).argmax(axis=1)
        for i in range(len(y)):
            Mat[y[i], y_hat[i]]+=1

        if(now_c%5000==0):
            right_vector = Mat.diagonal()
            OA = np.sum(right_vector) / np.sum(Mat)
            GL = right_vector / np.sum(Mat, axis=1)
            AA = np.mean( GL )
            print("Detail", end=': ')
            print(np.sum(Mat, axis=1))
            
            PE = 0
            for i in range(kind_sum):
                PE += np.sum(Mat[i,:])*np.sum(Mat[:,i])
            PE = PE / ( np.sum(Mat)**2)
            KAPPA = (OA-PE)/(1-PE)

        #########################################################
            global max_result
            global max_OA
            global max_AA
            global max_KP
            if(OA>max_result[0]):
                max_result[0]=OA
                max_OA = [OA, AA, KAPPA]
            if(AA>max_result[1]):
                max_result[1]=AA
                max_AA = [OA, AA, KAPPA]
            if(KAPPA>max_result[2]):
                max_result[2]=KAPPA
                max_KP = [OA, AA, KAPPA]
            break

    return [OA, AA, KAPPA, GL]


def train_model(net, train_iter, test_iter, num_epochs, lr):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    net.cuda(net_num)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    
    for epoch in range(1, num_epochs+1):
        net.train()
        now = time.time()
        ret = [0,0,0,0]
        for A, B, y in train_iter:
            optimizer.zero_grad()
            A, B, y = A.cuda(net_num), B.cuda(net_num), y.cuda(net_num)
            y_hat, Tloss = net(A, B)
            ls = loss(y_hat, y)+0.25*Tloss
            ls.backward()
            optimizer.step()
            with torch.no_grad():
                ret[0] += ls * bz
                ret[1] += Tloss * bz
                ret[2] += batch_correct(y_hat, y)
                ret[3] += bz

        with torch.no_grad():
            train_l   = ret[0] / ret[3]
            train_t   = ret[1] / ret[3]
            train_acc = ret[2] / ret[3]
        train_info = f'epoch{epoch:2d}, loss {train_l.item():.4f}, Tloss {train_t.item():.4f}, train acc {train_acc:.4f}, time{time.time()-now:.1f}\n'
        
        print(train_info,end='')
        with open('../model/log/log.txt', mode='a+') as f:
            f.write(train_info)
        
        if(epoch%3==0):
            now_t = time.time()
            test_res = test_model(net, test_iter)
            with open('../model/log/log.txt', mode='a+') as f:
                f.write("result:%s\n"%(str(test_res)))
            print("Runing Time: %.2f"%(time.time()-now_t))
            print("Test Result: "+ str(test_res))
            print("Best OA is: "+ str(max_OA))
            print("Best AA is: "+ str(max_AA))
            print("Best KP is: "+ str(max_KP))
            AA = test_res[1]
            if AA>0.9:
                torch.save(net, '../model/%s.pkl'%AA)
            

####################################################################################################
with open('../model/log/log.txt', mode='a+') as f:
    f.write(f'\n############# %s ##############\n'%(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')))

lr, num_epochs = 3e-4, 985
train_model(YNet(), train_iter, test_iter, num_epochs, lr)


