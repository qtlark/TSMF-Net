import cv2
import scipy.io
import numpy as np
from libtiff import TIFF


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


msf = to_tensor( TIFF.open('../ms4.tif', mode='r').read_image() )     
msf = cv2.resize(msf, (msf.shape[1]*2, msf.shape[0]*2), interpolation=cv2.INTER_LINEAR)         
msf0 = msf[:,:,0].flatten()
msf1 = msf[:,:,1].flatten()
msf2 = msf[:,:,2].flatten()
msf3 = msf[:,:,3].flatten()


pan = to_tensor( TIFF.open('../pan.tif', mode='r').read_image() )     
pan = split(pan, 2)
pan0 = pan[0,:,:].flatten()
pan1 = pan[1,:,:].flatten()
pan2 = pan[2,:,:].flatten()
pan3 = pan[3,:,:].flatten()



X = np.stack((msf0,msf1,msf2,msf3,pan0,pan1,pan2,pan3), axis=1)
print(X.dtype, X.shape)

scipy.io.savemat('X.mat', {'arr': X})
#np.savetxt('X.txt',X)


'''
out_tiff = TIFF.open('X6.tif', mode = 'w')
out_tiff.write_image('', compression = None)
out_tiff.close()
'''
