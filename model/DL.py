import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, head_num, bias=True):
        """ Multi-head self attention layer """
        super(MultiHeadSelfAttention, self).__init__()
        self.head_num = head_num
        self.in_features = in_features
        self.out_features = in_features // head_num
        self.linear_q = nn.Linear(in_features, self.out_features * head_num, bias=bias)
        self.linear_k = nn.Linear(in_features, self.out_features * head_num, bias=bias)
        self.linear_v = nn.Linear(in_features, self.out_features * head_num, bias=bias)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.linear_q(x).view(batch_size, -1, self.head_num, self.out_features)
        k = self.linear_k(x).view(batch_size, -1, self.head_num, self.out_features)
        v = self.linear_v(x).view(batch_size, -1, self.head_num, self.out_features)

        attention = torch.einsum('bihd,bjhd->bhij', q, k) / (self.out_features ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.einsum('bhij,bjhd->bihd', attention, v)
        out = out.contiguous().view(batch_size, -1, self.head_num * self.out_features)
        return out
class Encoder(nn.Module):
    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features//2)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features//2, in_features//2)
        self.bn2 = nn.BatchNorm1d(1)
        self.relu2 = nn.ReLU()
        self.mlp = nn.Sequential(
                    nn.Linear((in_features//2)*2, (in_features//2)*4),
                    nn.ReLU(),
                    nn.Linear((in_features//2)*4, (in_features//2))
                )
    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x2 = self.linear2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        out1 = torch.cat([x1, x2], dim=-1)
        out1 = self.mlp(out1)
        return out1

class Decoder(nn.Module):
    def __init__(self, in_features):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(1)
        self.relu2 = nn.ReLU()
        self.mlp = nn.Sequential(
                    nn.Linear(in_features*2, in_features*4),
                    nn.ReLU(),
                    nn.Linear(in_features*4, in_features)
                )
    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x2 = self.linear2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        out1 = torch.cat([x1, x2], dim=-1)
        out1 = self.mlp(out1)
        return out1
class SkipBlock(nn.Module):
    def __init__(self, in_features):
        super(SkipBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x
class DL(nn.Module):
    def __init__(self, input_size):
        super(DL, self).__init__()
        self.p1 = nn.Linear(input_size, 512)

        self.encoder1 = Encoder(512)
        self.attention1 = MultiHeadSelfAttention(256, 4)
        self.skip1 = SkipBlock(256)

        self.encoder2 = Encoder(256)
        self.attention2 = MultiHeadSelfAttention(128, 4)
        self.skip2 = SkipBlock(128)

        self.encoder3 = Encoder(128)
        self.attention3 = MultiHeadSelfAttention(64, 4)
        self.skip3 = SkipBlock(64)

        self.decoder3=Decoder(64)
        self.decoder2=Decoder(128)
        self.decoder1=Decoder(256)
        self.out=nn.Linear(512, input_size)

    def forward(self, x):
        x1 = self.p1(x)
        e1 = self.encoder1(x1)
        e1 = self.attention1(e1)
        s1 = self.skip1(e1)
        e2 = self.encoder2(e1)
        e2 = self.attention2(e2)
        s2 = self.skip2(e2)
        e3 = self.encoder3(e2)
        e3 = self.attention3(e3)
        s3 = self.skip3(e3)

        d3 = self.decoder3(s3)
        d3 = torch.cat((d3, s3), dim=-1)


        d2 = self.decoder2(d3)
        d2 = torch.cat((d2, s2), dim=-1)


        d1 = self.decoder1(d2)
        d1 = torch.cat((d1, s1), dim=-1)


        decoded = self.out(d1)
        return decoded
if __name__ == '__main__':
    x = torch.randn(64, 1, 48*48)
    net = DL(48*48)
    print(net(x).shape)