import scipy.io as sio
import numpy as np

seismic = sio.loadmat('../data/3Dsyn_data.mat')
dn = seismic['DataNoisy']
n1,n2,n3=dn.shape
mask=np.load('../data/3d_syn_mask.npy')
d0 = dn*mask

# after you run 3d_data.py
d1=np.load('DDIUL-3d-syn-d1.npy')
from pyseistr import cseis
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(12,4))
plt.imshow(np.concatenate([dn.reshape(n1,n2*n3),d0.reshape(n1,n2*n3),d1.reshape(n1,n2*n3),dn.reshape(n1,n2*n3)-d1.reshape(n1,n2*n3)],axis=1),cmap=cseis(),vmin=-0.5,vmax=0.5,aspect='auto')
plt.title("Noisy|Incomplete|Reconstructed|Noise")
plt.tight_layout()
plt.savefig('3Dresult.png')
plt.show()