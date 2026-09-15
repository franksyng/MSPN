import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, in_size, reduction_size=512, out_size=1):
        super(FCLayer, self).__init__()
        if in_size != reduction_size:
            self.fc_reduction = nn.Linear(in_size, reduction_size)
        else:
            self.fc_reduction = None
        self.fc = nn.Sequential(nn.Linear(reduction_size, out_size))
    def forward(self, feats):
        if self.fc_reduction != None:
            feats = self.fc_reduction(feats)
        x = self.fc(feats)
        return feats, x


class BClassifier(nn.Module):
    def __init__(self, input_size, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        # self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
    
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        # print(V.shape, Q.shape)
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A_raw = A
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        return A_raw, B
    
class DSMIL(nn.Module):
    def __init__(self, in_channels, n_classes, i_classifier, b_classifier, surv=False, reduction_size=512):
        super(DSMIL, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=reduction_size)
        self.surv = surv
        
    def forward(self, x):
        # print('x shape', x.shape)
        x = x.squeeze(0)
        feats, classes = self.i_classifier(x)
        A_raw, B = self.b_classifier(feats, classes)
        prediction_bag = self.fcc(B).view(1,-1) # 1 x C x 1
        # C = C.view(1, -1)
        # return C, A_raw, B 
        # print(prediction_bag.shape)
        if self.surv:
            return prediction_bag, None
        else:
            return classes, prediction_bag, A_raw, B


class DSMILMS(nn.Module):
    def __init__(self, n_classes, i_classifier, b_classifier, mil_hidden_1=512, mil_hidden_2=64):
        super(DSMILMS, self).__init__()
        self.i_classifier_5x = i_classifier
        self.b_classifier_5x = b_classifier

        self.i_classifier_10x = i_classifier
        self.b_classifier_10x = b_classifier

        self.i_classifier_20x = i_classifier
        self.b_classifier_20x = b_classifier

        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(int(mil_hidden_2*2), n_classes),
        # )
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=mil_hidden_2)
        
    def forward(self, x):
        # print('x shape', x.shape)
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        feats_5x, classes_5x = self.i_classifier_5x(x_5x)
        A_raw_5x, B_5x = self.b_classifier_5x(feats_5x, classes_5x)
        feats_10x, classes_10x = self.i_classifier_10x(x_10x)
        A_raw_10x, B_10x = self.b_classifier_10x(feats_10x, classes_10x)
        feats_20x, classes_20x = self.i_classifier_20x(x_20x)
        A_raw_20x, B_20x = self.b_classifier_20x(feats_20x, classes_20x)
        # print(B_5x.shape)

        ms_feats = torch.permute(torch.concat((self.ms_encoder(B_5x), self.ms_encoder(B_10x), self.ms_encoder(B_20x)), dim=0), (1,2,0))
        # print(ms_feats.shape)
        cs_attn = self.cs_attn(ms_feats.view(ms_feats.shape[0], ms_feats.shape[1], ms_feats.shape[2], 1))
        cs_attn = F.softmax(cs_attn.view(cs_attn.shape[0], cs_attn.shape[1], cs_attn.shape[2]), dim=-1)
        cs_attn_feats = torch.matmul(cs_attn, torch.permute(ms_feats, (0, 2, 1))).view(1,cs_attn.shape[0],-1)
        prediction_bag = self.fcc(cs_attn_feats).view(1,-1)

        return prediction_bag, None

class DSMILMSCat(nn.Module):
    def __init__(self, n_classes, i_classifier, b_classifier, mil_hidden_1=512, mil_hidden_2=64):
        super(DSMILMSCat, self).__init__()
        self.i_classifier_5x = i_classifier
        self.b_classifier_5x = b_classifier

        self.i_classifier_10x = i_classifier
        self.b_classifier_10x = b_classifier

        self.i_classifier_20x = i_classifier
        self.b_classifier_20x = b_classifier

        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        # self.cs_attn = nn.Sequential(
        #     nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        # )
        # # self.classifier = nn.Sequential(
        # #     nn.Linear(int(mil_hidden_2*2), n_classes),
        # # )
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=int(mil_hidden_2*3))
        
    def forward(self, x):
        # print('x shape', x.shape)
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        feats_5x, classes_5x = self.i_classifier_5x(x_5x)
        A_raw_5x, B_5x = self.b_classifier_5x(feats_5x, classes_5x)
        feats_10x, classes_10x = self.i_classifier_10x(x_10x)
        A_raw_10x, B_10x = self.b_classifier_10x(feats_10x, classes_10x)
        feats_20x, classes_20x = self.i_classifier_20x(x_20x)
        A_raw_20x, B_20x = self.b_classifier_20x(feats_20x, classes_20x)
        # print(B_5x.shape)


        ms_feats = torch.concat((self.ms_encoder(B_5x), self.ms_encoder(B_10x), self.ms_encoder(B_20x)), dim=0).view(B_5x.shape[1], -1)
        prediction_bag = self.fcc(ms_feats).view(1,-1)

        return prediction_bag, None
