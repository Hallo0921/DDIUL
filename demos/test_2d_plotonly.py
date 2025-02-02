import scipy.io as sio
import numpy as np

seismic = sio.loadmat('../data/2Dsyn_data.mat')
dn = seismic['DataNoisy']

mask=np.load('../data/2d_syn_mask.npy')
d0 = dn*mask

# after you run 2d_data.py, taking around 20 minutes (on CPU)
d1=np.load('DDIUL-2d-syn-d1.npy')

from pyseistr import cseis
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(12,4))
plt.imshow(np.concatenate([dn,d0,d1,dn-d1],axis=1),cmap=cseis(),aspect='auto')
plt.title("Noisy|Incomplete|Reconstructed|Noise")
plt.savefig('2Dresult.png')
plt.show()