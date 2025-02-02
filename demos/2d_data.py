import numpy as np
import torch
import scipy.io as sio
from sklearn.decomposition import SparseCoder, MiniBatchDictionaryLearning
from torch import nn
import sys
sys.path.append('../model')
from model.DL import DL
# from DL import DL
# import DL as DL
from pyseistr.patch import patch2d, patch2d_inv

seismic = sio.loadmat('../data/2Dsyn_data.mat')
dn = seismic['DataNoisy']
mask=np.load('../data/2d_syn_mask.npy')
d0 = dn*mask
n1, n2=dn.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DL(48*48)
net.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
min_loss = float('inf')
best_model_state = None
run_loss=[]
epoch = 1
niter = 10
a = (niter - np.arange(niter)) / (niter - 1)
dbef = np.zeros((niter, n1, n2))
Pafter = np.zeros((niter, n1, n2))
Pafter_d2 = np.zeros((niter, n1, n2))
d1 = d0.copy()
for iter in range(niter):
    print(f"Iteration {iter + 1}")

    dbef[iter, :, :] = d1
    # Patch Process
    data = patch2d(d1, 48, 48, 1, 1, 1)

    # Reconstruction Error-Based Patch Selection (REBPS)
    normalized_data = data
    n_components = 1000
    dictionary_learner = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, n_iter=100)
    dictionary = dictionary_learner.fit(normalized_data).components_
    sparse_coder = SparseCoder(dictionary=dictionary, transform_algorithm='omp', transform_n_nonzero_coefs=10)
    sparse_codes = sparse_coder.transform(normalized_data)
    reconstructed_data = np.dot(sparse_codes, dictionary)
    reconstruction_errors = np.linalg.norm(normalized_data - reconstructed_data, axis=1)
    num_representatives = int(np.round(data.shape[0] * 0.75))
    representative_indices = np.argsort(reconstruction_errors)[-num_representatives:]
    representative_patches = normalized_data[representative_indices]
    # print(representative_patches.shape)

    slices_array = np.array(representative_patches)
    slices_tensor = torch.from_numpy(slices_array).float()
    data = torch.from_numpy(data).float()
    data = data.view(data.shape[0], 1, data.shape[1])
    train_loader = torch.utils.data.DataLoader(dataset=slices_tensor, batch_size=2048, shuffle=True, drop_last=True)

    for i in range(epoch):
        running_loss = 0.0
        for input in train_loader:
            input = input.view(2048, 1, 48*48)
            input = input.to(device)

            output = net(input)
            loss = loss_fn(output, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        run_loss.append(running_loss/len(train_loader))
        print('train:{}, loss:{}'.format(i+1, run_loss[i]))

    net.eval()
    with torch.no_grad():
        data = data.to(device)
        output = net(data)
    output = output.cpu().detach().squeeze().numpy()
    # Unpatch Process
    d2 = patch2d_inv(output, n1, n2, 48, 48, 1, 1, 1)
    Pafter_d2[iter, :, :] = d2
    # POCS
    d1 = a[iter] * d0 * mask + (1 - a[iter]) * d2 * mask + d2 * (1 - mask)
    # Recording result
    Pafter[iter, :, :] = d1

# Save reconstruction data
np.save('DDIUL-2d-syn-d1.npy', d1)