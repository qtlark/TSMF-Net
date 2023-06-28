import cv2
import numpy as np
from libtiff import TIFF
from scipy.io import savemat

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


msf = TIFF.open('../dataset/ms4.tif', mode='r').read_image().astype("float32")
msf = cv2.resize(msf, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
msf = to_tensor( msf )  
msf = msf.reshape((-1, 4))
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
savemat('../dataset/msf.mat', {'msf': msf})


pan = TIFF.open('../dataset/pan.tif', mode='r').read_image().astype("float32")
pan = split(pan, 2)
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
pan = to_tensor( pan )   
pan = pan.reshape((-1, 4))
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
savemat('../dataset/pan.mat', {'pan': pan})

