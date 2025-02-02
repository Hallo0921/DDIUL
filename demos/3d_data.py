import numpy as np
import torch
import scipy.io as sio
from sklearn.decomposition import SparseCoder, MiniBatchDictionaryLearning
from torch import nn
import sys
sys.path.append('../model')
# from model.DL import DL
from DL import DL
# import DL as DL
from pyseistr.patch import patch3d, patch3d_inv

seismic = sio.loadmat('../data/3Dsyn_data.mat')
dn = seismic['DataNoisy']
n1,n2,n3=dn.shape
mask=np.load('../data/3d-syn-mask.npy')
d0=dn*mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=DL(15*15*15)
net.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
min_loss = float('inf')
best_model_state = None
run_loss=[]
epoch = 50
niter = 10
a = (niter - np.arange(niter)) / (niter - 1)
dbef = np.zeros((niter, n1, n2, n3))
Pafter = np.zeros((niter, n1, n2, n3))
Pafter_d2 = np.zeros((niter, n1, n2, n3))
d1 = d0.copy()
for iter in range(niter):
    print(f"Iteration {iter + 1}")
    # 记录当前数据状态
    dbef[iter, :, :, :] = d1
    data = patch3d(d1, 15, 15, 15, 2, 2, 2, 1)
    data=data.T

    # 数据
    normalized_data = data
    # 设置字典大小
    n_components = 1000  # 字典的大小，可以根据数据量调整
    # 创建稀疏编码模型
    dictionary_learner = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, n_iter=100)
    # 学习字典
    dictionary = dictionary_learner.fit(normalized_data).components_
    # 创建稀疏编码器
    sparse_coder = SparseCoder(dictionary=dictionary, transform_algorithm='omp', transform_n_nonzero_coefs=10)
    # 编码数据
    sparse_codes = sparse_coder.transform(normalized_data)
    # 计算重建误差
    reconstructed_data = np.dot(sparse_codes, dictionary)
    reconstruction_errors = np.linalg.norm(normalized_data - reconstructed_data, axis=1)
    # 选择高重建误差的补丁
    num_representatives = int(np.round(data.shape[0] * 0.75))  # 选择的代表性补丁数量
    representative_indices = np.argsort(reconstruction_errors)[-num_representatives:]
    representative_patches = normalized_data[representative_indices]
    print(len(representative_patches))

    slices_array = np.array(representative_patches)
    slices_tensor = torch.from_numpy(slices_array).float()
    data = torch.from_numpy(data).float()
    data = data.view(data.shape[0], 1, data.shape[1])
    train_loader = torch.utils.data.DataLoader(dataset=slices_tensor, batch_size=2048, shuffle=True, drop_last=True)

    for i in range(epoch):
        running_loss = 0.0
        for input in train_loader:
            input = input.view(2048, 1, 15*15*15)
            input = input.to(device)
            # 前向传播
            output = net(input)
            loss = loss_fn(output, input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        run_loss.append(running_loss/len(train_loader))
        print('训练第:{}轮, loss为:{}'.format(i+1, run_loss[i]))

    net.eval()
    with torch.no_grad():
        data = data.to(device)
        output = net(data)
    output = output.cpu().detach().squeeze().numpy()
    output = np.transpose(output)
    d2 = patch3d_inv(output, n1, n2, n3, 15, 15, 15, 2, 2, 2, 1)
    Pafter_d2[iter, :, :, :] = d2
    # 插值
    d1 = a[iter] * d0 * mask + (1 - a[iter]) * d2 * mask + d2 * (1 - mask)

    # 记录结果
    Pafter[iter, :, :, :] = d1

# Save reconstruction data
np.save('DDIUL-3d-syn-d1.npy', d1)