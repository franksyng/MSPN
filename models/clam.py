import torch
import torch.nn as nn
import torch.nn.functional as F

class CLAM_SB_Head(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super(CLAM_SB_Head, self).__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # k must not exceed the bag size. PANDA needle biopsies have as few as
        # 6 patches at 5x (22 slides < 8), and the multi-scale CLAM heads run
        # a 5x branch, so topk(A, 8) raised 'selected index k out of range'.
        # No-op for every other cohort here (bags are thousands of patches).
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # k must not exceed the bag size. PANDA needle biopsies have as few as
        # 6 patches at 5x (22 slides < 8), and the multi-scale CLAM heads run
        # a 5x branch, so topk(A, 8) raised 'selected index k out of range'.
        # No-op for every other cohort here (bags are thousands of patches).
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        # if attention_only:
            # return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        total_inst_loss = 0.0
        if instance_eval:
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        # logits = self.classifiers(M)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # Y_prob = F.softmax(logits, dim = 1)
        # if instance_eval:
            # results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            # 'inst_preds': np.array(all_preds)}
        # else:
            # results_dict = {}
        # if return_features:
            # results_dict.update({'features': M})
        # return logits, Y_prob, Y_hat, A_raw, results_dict
        return M, A_raw, total_inst_loss

class CLAM_MB_Head(CLAM_SB_Head):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        # nn.Module.__init__(self)
        super(CLAM_MB_Head, self).__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        # bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        # self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        total_inst_loss = 0.0
        if instance_eval:
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        return M, A_raw, total_inst_loss

        # logits = torch.empty(1, self.n_classes).float().to(M.device)
        # for c in range(self.n_classes):
        #     logits[0, c] = self.classifiers[c](M[c])

        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # Y_prob = F.softmax(logits, dim = 1)
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
        #     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        # if return_features:
        #     results_dict.update({'features': M})
        # return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_SB(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super(CLAM_SB, self).__init__()
        self.clamsb_head = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.classifier = nn.Linear(512, n_classes)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            h = h.squeeze(0)
        M, A_raw, total_inst_loss = self.clamsb_head(h, label, instance_eval)
        logits = self.classifier(M)
        return logits, total_inst_loss

class CLAM_MB(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super(CLAM_MB, self).__init__()
        self.clamsb_head = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.n_classes = n_classes
        bag_classifiers = [nn.Linear(512, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            h = h.squeeze(0)
        M, A_raw, total_inst_loss = self.clamsb_head(h, label, instance_eval)
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        return logits, total_inst_loss


class CLAM_SB_MS(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), mil_hidden_1=512, mil_hidden_2=64, subtyping=False, embed_dim=1024):
        super(CLAM_SB_MS, self).__init__()
        self.clamsb_head_5x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.clamsb_head_10x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.clamsb_head_20x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Linear(mil_hidden_2, n_classes)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            # h = h.squeeze(0)
            h_5x, h_10x, h_20x = h
            h_5x = h_5x.squeeze(0)
            h_10x = h_10x.squeeze(0)
            h_20x = h_20x.squeeze(0)
            
        M_5x, A_raw_5x, total_inst_loss_5x = self.clamsb_head_5x(h_5x, label, instance_eval)
        M_10x, A_raw_10x, total_inst_loss_10x = self.clamsb_head_10x(h_10x, label, instance_eval)
        M_20x, A_raw_20x, total_inst_loss_20x = self.clamsb_head_20x(h_20x, label, instance_eval)

        ms_feats = torch.concat((self.ms_encoder(M_5x), self.ms_encoder(M_10x), self.ms_encoder(M_20x)), dim=0).T
        cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        cs_attn_feats = torch.mm(cs_attn, ms_feats.T)
        # print(cs_attn_feats.shape)
        logits = self.classifier(cs_attn_feats)

        # logits = self.classifier(M)
        return logits, total_inst_loss_5x + total_inst_loss_10x + total_inst_loss_20x


class CLAM_MB_MS(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), mil_hidden_1=512, mil_hidden_2=64, subtyping=False, embed_dim=1024):
        super(CLAM_MB_MS, self).__init__()
        self.clammb_head_5x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.clammb_head_10x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.clammb_head_20x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.n_classes = n_classes
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        bag_classifiers = [nn.Linear(mil_hidden_2, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.bag_classifiers = nn.ModuleList(bag_classifiers)
        # self.classifier = nn.Linear(mil_hidden_2, n_classes)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            # h = h.squeeze(0)
            h_5x, h_10x, h_20x = h
            h_5x = h_5x.squeeze(0)
            h_10x = h_10x.squeeze(0)
            h_20x = h_20x.squeeze(0)
            
        M_5x, A_raw_5x, total_inst_loss_5x = self.clammb_head_5x(h_5x, label, instance_eval)
        M_10x, A_raw_10x, total_inst_loss_10x = self.clammb_head_10x(h_10x, label, instance_eval)
        M_20x, A_raw_20x, total_inst_loss_20x = self.clammb_head_20x(h_20x, label, instance_eval)

        logits = torch.empty(1, self.n_classes).float().to(M_5x.device)
        for c in range(self.n_classes):
            ms_feats = torch.concat((self.ms_encoder(M_5x[c]).unsqueeze(0), self.ms_encoder(M_10x[c]).unsqueeze(0), self.ms_encoder(M_20x[c]).unsqueeze(0)), dim=0).T
            cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
            cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
            cs_attn_feats = torch.mm(cs_attn, ms_feats.T)
            logits[0, c] = self.bag_classifiers[c](cs_attn_feats)

        return logits, total_inst_loss_5x + total_inst_loss_10x + total_inst_loss_20x
    
class CLAM_SB_MSCat(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), mil_hidden_1=512, mil_hidden_2=64, subtyping=False, embed_dim=1024):
        super(CLAM_SB_MSCat, self).__init__()
        self.clamsb_head_5x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.clamsb_head_10x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.clamsb_head_20x = CLAM_SB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.classifier = nn.Linear(int(mil_hidden_2*3), n_classes)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            # h = h.squeeze(0)
            h_5x, h_10x, h_20x = h
            h_5x = h_5x.squeeze(0)
            h_10x = h_10x.squeeze(0)
            h_20x = h_20x.squeeze(0)
            
        M_5x, A_raw_5x, total_inst_loss_5x = self.clamsb_head_5x(h_5x, label, instance_eval)
        M_10x, A_raw_10x, total_inst_loss_10x = self.clamsb_head_10x(h_10x, label, instance_eval)
        M_20x, A_raw_20x, total_inst_loss_20x = self.clamsb_head_20x(h_20x, label, instance_eval)

        ms_feats = torch.concat((self.ms_encoder(M_5x), self.ms_encoder(M_10x), self.ms_encoder(M_20x)), dim=1)
        logits = self.classifier(ms_feats)

        # logits = self.classifier(M)
        return logits, total_inst_loss_5x + total_inst_loss_10x + total_inst_loss_20x

class CLAM_MB_MSCat(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), mil_hidden_1=512, mil_hidden_2=64, subtyping=False, embed_dim=1024):
        super(CLAM_MB_MSCat, self).__init__()
        self.clammb_head_5x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                        subtyping=subtyping, embed_dim=embed_dim)
        self.clammb_head_10x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.clammb_head_20x = CLAM_MB_Head(size_arg=size_arg, dropout=dropout, k_sample=k_sample, n_classes=n_classes, instance_loss_fn=instance_loss_fn,
                                subtyping=subtyping, embed_dim=embed_dim)
        self.n_classes = n_classes
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        # self.classifier = nn.Linear(int(mil_hidden_2*3), n_classes)
        bag_classifiers = [nn.Linear(int(mil_hidden_2*3), 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.bag_classifiers = nn.ModuleList(bag_classifiers)
    
    def forward(self, h, label=None, instance_eval=False, vis_heatmap=False):
        if not vis_heatmap:
            # h = h.squeeze(0)
            h_5x, h_10x, h_20x = h
            h_5x = h_5x.squeeze(0)
            h_10x = h_10x.squeeze(0)
            h_20x = h_20x.squeeze(0)
            
        M_5x, A_raw_5x, total_inst_loss_5x = self.clammb_head_5x(h_5x, label, instance_eval)
        M_10x, A_raw_10x, total_inst_loss_10x = self.clammb_head_10x(h_10x, label, instance_eval)
        M_20x, A_raw_20x, total_inst_loss_20x = self.clammb_head_20x(h_20x, label, instance_eval)

        logits = torch.empty(1, self.n_classes).float().to(M_5x.device)
        for c in range(self.n_classes):
            ms_feats = torch.concat((self.ms_encoder(M_5x[c]).unsqueeze(0), self.ms_encoder(M_10x[c]).unsqueeze(0), self.ms_encoder(M_20x[c]).unsqueeze(0)), dim=1)
            logits[0, c] = self.bag_classifiers[c](ms_feats)
        return logits, total_inst_loss_5x + total_inst_loss_10x + total_inst_loss_20x



class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x