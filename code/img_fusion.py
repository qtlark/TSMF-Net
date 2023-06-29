import cv2
import numpy as np
from libtiff import TIFF

# Functions
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
    return np.stack(st, axis=-1)


def image_gradient(img):
    H, W = img.shape
    gx = np.pad(np.diff(img, axis=0), ((0,1),(0,0)), 'constant')
    gy = np.pad(np.diff(img, axis=1), ((0,0),(0,1)), 'constant')
    gradient = abs(gx) + abs(gy)
    return gradient


def edge_dect(img):
    nam=1e-9
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
  
    


# begin
msf = TIFF.open('../dataset/ms4.tif', mode='r').read_image().astype("float32")
msf = cv2.resize(msf, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
msf = to_tensor( msf )  
msf = msf.transpose( (2,0,1) ) 


pan = TIFF.open('../dataset/pan.tif', mode='r').read_image().astype("float32")
pan = split(pan, 2)
pan = to_tensor( pan )  
pan = pan.transpose( (2,0,1) ) 






# align the channel
gram = get_gram(msf, pan)
find_plzh(0)
print(gram)

nowmax  = 0
nowbest = ()
for pl in plzh:
    nowsum = 0
    for i in range(4):
        nowsum+=gram[i][pl[i]]
    if nowsum>nowmax:
        nowmax  = nowsum
        nowbest = pl

print(nowbest)
print(np.sum(np.sum(pan, axis=-1),axis=-1))
pan[[0,1,2,3],:,:] = pan[nowbest,:,:]       # switch the order
print(np.sum(np.sum(pan, axis=-1),axis=-1))



# Intensity component
alpha = np.array( [0.3707, 0.0000, 0.0174, 0.4367] )
beta  = np.array( [0.3937, 0.0055, 0.0929, 0.5079] )
beta[[0,1,2,3]] = beta[nowbest]             # switch the order
print("<== Please check parameters every time you run the code ==>")
print("alpha is" + str(alpha))
print("beta  is" + str(beta))
print("<== Please check parameters every time you run the code ==>")

I_m = alpha[0]*msf[0] + alpha[1]*msf[1] + alpha[2]*msf[2] + alpha[3]*msf[3]
I_p =  beta[0]*pan[0] +  beta[1]*pan[1] +  beta[2]*pan[2] +  beta[3]*pan[3]

I_mean = 0.5*(I_m+I_p)
mu  = np.mean(I_mean)
gamma = sigmoid( (I_mean-mu)*( sign(I_m-I_p) ) )
I_c = gamma*I_m + (1-gamma)*I_p

print("The bias between I_m and I_p is", end=': ')
print(np.sum((I_m-I_p)**2))         
print("The bias between I_m and I_c is", end=': ')
print(np.sum((I_m-I_c)**2))
print("The bias between I_p and I_c is", end=': ')
print(np.sum((I_p-I_c)**2))
print("The bias between I_mean and I_c is", end=': ')
print(np.sum((I_mean-I_c)**2))


# edge detection operator
W_mi = [ edge_dect(msf[i]) for i in range(4) ]
W_pi = [ edge_dect(pan[i]) for i in range(4) ]


alpha_mean = np.mean(alpha)
beta_mean  = np.mean(beta)
W_m  = [alpha[i]/alpha_mean*( gamma*W_mi[i] + (1-gamma)*W_pi[i] ) for i in range(4)]
W_p  = [beta[i] /beta_mean* ( gamma*W_mi[i] + (1-gamma)*W_pi[i] ) for i in range(4)]


# fusion
for i in range(4):
    msf[i] = msf[i] + W_m[i]*(I_c-I_m) 
    pan[i] = pan[i] + W_p[i]*(I_c-I_p) 

msf = msf.transpose( (1,2,0) )      
pan = pan.transpose( (1,2,0) )      

print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))

np.save("../dataset/msf_f.npy", msf)
np.save("../dataset/pan_f.npy", pan)
