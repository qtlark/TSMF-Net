import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF

############# 要用到的函数 #############
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def split(pan, size):
    st = []
    for i in range(size):
        for j in range(size):
            st.append(pan[i::size,j::size])
    return np.array(st)


def image_gradient(img):
    H, W = img.shape
    gradient = np.zeros([H, W])
    gx = 0
    gy = 0
    for i in range(H - 1):
        for j in range(W - 1):
            gx = img[i + 1, j] - img[i, j]
            gy = img[i, j + 1] - img[i, j]
            gradient[i, j] = np.sqrt(gx**2 + gy**2)

    return gradient


def edge_dect(img):
    nam=1e-09
    apx=1e-10
    return np.exp( -nam / ( (image_gradient(img)**4)+apx ) )


def get_gram(msf, pan):
    res = np.array(range(16)).reshape((4,4)).astype('float32')
    for i in range(4):
        for j in range(4):
            res[i][j] = np.sum(msf[i]*pan[j])
    return res

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sign(x):
    x[x>=0] =  1
    x[x< 0] = -1
    return x 


chk = np.zeros(128).astype('int32')
ans = np.zeros(128).astype('int32')
plzh= []
def find_plzh(x):
    if x>3:
        plzh.append( [ans[0],ans[1],ans[2],ans[3]] )
        return 
    
    for i in range(4):
        if chk[i]==0:
            ans[x] = i
            chk[i] = 1
            find_plzh(x+1)
            chk[i] = 0
############# 要用到的函数 #############        
    


# 开始
msf = to_tensor( TIFF.open('../../Dataset/image6/ms4.tif', mode='r').read_image() )     
msf = cv2.resize(msf, (msf.shape[1]*2, msf.shape[0]*2), interpolation=cv2.INTER_LINEAR)      
msf = msf.transpose( (2,0,1) )      #(800,830,4)->(4,800,830)

pan = to_tensor( TIFF.open('../../Dataset/image6/pan.tif', mode='r').read_image() )     
pan = split(pan, 2)






# 通道对齐
gram = get_gram(msf, pan)
find_plzh(0)

nowmin  = 985e6
nowbest = ()
for pl in plzh:
    nowsum = 0
    for i in range(4):
        nowsum+=gram[i][pl[i]]
    if nowsum<nowmin:
        nowmin  = nowsum
        nowbest = pl

print(nowbest)
#panc = np.copy(pan)
pan[[0,1,2,3],:,:] = pan[nowbest,:,:]       #调换顺序
#print(  np.sum(pan[0]==panc[0]), np.sum(pan[1]==panc[1]), np.sum(pan[2]==panc[2]), np.sum(pan[3]==panc[3])   )
#print(  np.sum(pan[0]==panc[0]), np.sum(pan[2]==panc[1]), np.sum(pan[1]==panc[2]), np.sum(pan[3]==panc[3])   )


# 强度分量
alpha = np.array( [0.0000, 0.4266, 0.0388, 0.5347] )
beta  = np.array( [0.4352, 0.0486, 0.0545, 0.2652] )
beta[[0,1,2,3]] = beta[nowbest]             #调换顺序

I_m = alpha[0]*msf[0] + alpha[1]*msf[1] + alpha[2]*msf[2] + alpha[3]*msf[3]
I_p =  beta[0]*pan[0] +  beta[1]*pan[1] +  beta[2]*pan[2] +  beta[3]*pan[3]

I_mean = 0.5*(I_m+I_p)
mu  = np.mean(I_mean)
gamma = sigmoid( (I_mean-mu)*( sign(I_m-I_p) ) )


I_c = gamma*I_m + (1-gamma)*I_p

print(np.sum(gamma>0.5)/np.sum(gamma>-1))

print("I_m和I_p之间的bias为", end=': ')
print(np.sum((I_m-I_p)**2))         #2.88有误差，1.5的那个没误差，记得再检查
print("I_m和I_c之间的bias为", end=': ')
print(np.sum((I_m-I_c)**2))
print("I_p和I_c之间的bias为", end=': ')
print(np.sum((I_p-I_c)**2))
print("I_mean和I_c之间的bias为", end=': ')
print(np.sum((I_mean-I_c)**2))


# 检测算子
W_mi = [ edge_dect(msf[0]), edge_dect(msf[1]), edge_dect(msf[2]), edge_dect(msf[3]) ]
W_pi = [ edge_dect(pan[0]), edge_dect(pan[1]), edge_dect(pan[2]), edge_dect(pan[3]) ]


alpha_mean = np.mean(alpha)
beta_mean  = np.mean(beta)
W_m  = []
W_p  = []
for i in range(4):
    W_m.append( alpha[i]/alpha_mean*( gamma*W_mi[i] + (1-gamma)*W_pi[i] ) )
    W_p.append(  beta[i]/ beta_mean*( gamma*W_mi[i] + (1-gamma)*W_pi[i] ) )


# 融合
for i in range(4):
    msf[i] = msf[i] + W_m[i]*(I_c-I_m) 
    pan[i] = pan[i] + W_p[i]*(I_c-I_p) 

msf = msf.transpose( (1,2,0) )      #(4,1600,1660) -> (1600,1660,4)
pan = pan.transpose( (1,2,0) )      #(4,1600,1660) -> (1600,1660,4)

print(msf.dtype, msf.shape)
print(pan.dtype, pan.shape)

np.save("msf_f6.npy", msf)
np.save("pan_f6.npy", pan)
