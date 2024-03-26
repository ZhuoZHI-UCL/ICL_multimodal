
from ....tools.tools_model import Transformer_cross_attention, Transformer_mask_encoder, Transformer_self_regression
import pytorch_lightning as pl
import numpy as np
import torch
import random
import torch.nn as nn
from sklearn import metrics
import socket
import os
import shutil
from torch.optim.lr_scheduler import  MultiStepLR
import pandas as pd
import time
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=975, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):  
        if x.size(1) != self.pe[:x.size(1)].size(1):
            print("Position embedding size mismatch!")
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)


class Medfuse_task1_ICL_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() 
        #params
        x = config.ehr_max_pooled 
        y = config.cxr_max_pooled
        z = 1
        m = config.num_NN + 1 
        
        self.label_index = torch.tensor((np.arange(m)[:, None] * (x + y + z) + np.arange(x+y, x+y+z)).ravel().tolist())
        self.ehr_index = torch.tensor((np.arange(m)[:, None] * (x + y + z) + np.arange(x)).ravel().tolist())
        self.cxr_index = torch.tensor((np.arange(m)[:, None] * (x + y + z) + np.arange(x, x+y)).ravel().tolist())
        #select the model 
        if self.hparams.config.Model_type == 'self_regression':
            self.segment_embedding = nn.Embedding(m, self.hparams.config.hidden_size)
            self.Position_embedding = PositionalEncoding(self.hparams.config.hidden_size, (x+y+z)*m)
            self.modality_embedding = nn.Embedding(3, self.hparams.config.hidden_size)
            self.ICL_transformer = Transformer_self_regression(hidden_size = self.hparams.config.hidden_size, 
                                                               num_layers = self.hparams.config.num_layers, 
                                                               num_heads = self.hparams.config.nheads,
                                                               max_seq_length=(x+y+z)*m)
        elif self.hparams.config.Model_type == 'encoder_mask':
            self.ICL_transformer = Transformer_mask_encoder(self.hparams.config.num_layers, 768,self.hparams.config.nheads)
            self.ehr_layernorm = nn.Sequential(
                    nn.LayerNorm(self.hparams.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hparams.config.hidden_size , self.hparams.config.hidden_size), 
                )
            self.cxr_layernorm = nn.Sequential(
                    nn.LayerNorm(self.hparams.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hparams.config.hidden_size , self.hparams.config.hidden_size), 
                )
            self.segment_embedding = nn.Embedding(m, self.hparams.config.hidden_size)
            self.Position_embedding = PositionalEncoding(self.hparams.config.hidden_size, x+y+z)
            self.modality_embedding = nn.Embedding(3, self.hparams.config.hidden_size)
            self.mask_ehr = nn.Parameter(torch.rand(1, self.hparams.config.hidden_size))
            self.mask_cxr = nn.Parameter(torch.rand(1, self.hparams.config.hidden_size))
            self.mask_label = nn.Parameter(torch.rand(1, self.hparams.config.hidden_size))
        elif self.hparams.config.Model_type == 'cross_attention':
            self.layer_norm_input = nn.LayerNorm(self.hparams.config.hidden_size)
            self.ICL_transformer = Transformer_cross_attention(768, self.hparams.config.nheads, self.hparams.config.num_layers, 8,8,1)
            self.segment_embedding = nn.Embedding(m, self.hparams.config.hidden_size)
            self.Position_embedding = PositionalEncoding(self.hparams.config.hidden_size, x+y+z)
            self.modality_embedding = nn.Embedding(3, self.hparams.config.hidden_size)
            
        #classifier
        if self.hparams.config.cal_margin:
            self.mimic_classifier = nn.Sequential(
                    nn.LayerNorm(self.hparams.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hparams.config.hidden_size, self.hparams.config.num_classes),
                    # nn.Sigmoid()
                )
        else:
            self.mimic_classifier = nn.Sequential(
                    nn.LayerNorm(self.hparams.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hparams.config.hidden_size, self.hparams.config.num_classes),
                    nn.Sigmoid()
                )   

            
            
        #all embedding layers
        self.label_embeddings = nn.Linear(1, self.hparams.config.hidden_size)
        #different label positions for different tasks
        self.position_label = 1
        self.position_flag = 2
        if self.hparams.config.Model_type == 'CLF':
            self.pooler = Pooler(self.hparams.config.hidden_size)
        
        #metric 
        self.all_preds_train = []
        self.all_labels_train = []
        self.complete_preds_train = []
        self.complete_labels_train = []
        self.missed_ehr_preds_train = []
        self.missed_ehr_labels_train = []
        self.missed_cxr_preds_train = []
        self.missed_cxr_labels_train = []
        self.ehr_paired_preds_train = [] 
        self.ehr_paired_labels_train = []

        self.all_preds_val = []
        self.all_labels_val = []
        self.complete_preds_val = []
        self.complete_labels_val = []
        self.missed_ehr_preds_val = []
        self.missed_ehr_labels_val = []
        self.missed_cxr_preds_val = []
        self.missed_cxr_labels_val = []
        self.ehr_paired_preds_val = [] 
        self.ehr_paired_labels_val = []
        self.best_final_matrix = 0

        self.all_preds_test = []
        self.all_labels_test = []
        self.complete_preds_test = []
        self.complete_labels_test = []
        self.missed_ehr_preds_test = []
        self.missed_ehr_labels_test = []
        self.missed_cxr_preds_test = []
        self.missed_cxr_labels_test = []
        self.ehr_paired_preds_test = [] 
        self.ehr_paired_labels_test = []
        self.step_count = 0
    
    def forward(self, batch):
        ICL_feature, ehr_label,  flag_pair, patient_id, all_ground_truth = batch
        ICL_feature_backup = ICL_feature.clone()
        m = self.hparams.config.num_NN + 1 #number of all samples
        x = self.hparams.config.ehr_max_pooled #token number for ehr feature
        y = self.hparams.config.cxr_max_pooled #token number for cxr feature
        z = 1 # label length
        self.ehr_index = self.ehr_index.to(self.device)
        self.cxr_index = self.cxr_index.to(self.device)
        self.label_index = self.label_index.to(self.device)
        
        #deal with label---------------------------------------------------------------------------------------------------------------------------------------------
        if self.hparams.config.use_original_label:
            self.all_labels_groundtruth = torch.zeros_like(ICL_feature[..., 0]) 
            self.all_labels_groundtruth[:, self.label_index]  = ICL_feature[:,self.label_index,0]
            selected_tokens = ICL_feature[:, self.label_index[:-1], 0]
            selected_tokens_flattened = selected_tokens.view(-1, self.hparams.config.num_classes)  
            all_tokens_embedded = self.label_embeddings(selected_tokens_flattened.int())
            reshaped_embeddings = all_tokens_embedded.view(len(ICL_feature), len(self.label_index[:-1]), -1)  # shape: [batch_size, num_indices, 768]
            ICL_feature[:, self.label_index[:-1], :] = reshaped_embeddings
   
        else:
            self.all_labels_groundtruth = torch.zeros_like(ICL_feature[..., 0]).to(self.device) 
            for i in range(len(self.label_index)):
                self.all_labels_groundtruth[:, self.label_index[i]]  = all_ground_truth[i]
            self.all_labels_groundtruth = self.all_labels_groundtruth.unsqueeze(-1)
        
     
        if self.hparams.config.Model_type == 'self_regression':        
            #all the embedding----------------------------------------------------------------------------------------------------------------------------------------------------
            #position embedding
            ICL_feature = self.Position_embedding(ICL_feature)
            #segment embedding
            segment_ids = torch.arange(m).unsqueeze(0).repeat(ICL_feature.shape[0], 1).to(ICL_feature.device)
            segment_embedded = self.segment_embedding(segment_ids)
            segment_embedded = segment_embedded.repeat_interleave(x+y+z, dim = 1)
            ICL_feature = ICL_feature + segment_embedded
            #modality embedding
            ehr_tensor = self.modality_embedding(torch.tensor(0).to(self.device)).expand(1, x, 768)
            cxr_tensor = self.modality_embedding(torch.tensor(1).to(self.device)).expand(1, y, 768)
            label_tensor = self.modality_embedding(torch.tensor(2).to(self.device)).expand(1, z, 768)
            single_sample_pattern = torch.cat([ehr_tensor, cxr_tensor, label_tensor], dim=1)
            modality_embedded = single_sample_pattern.repeat(ICL_feature.shape[0], m, 1)
            modality_embedded = modality_embedded[:, :(x+y+z)*m, :]
            ICL_feature = ICL_feature + modality_embedded
            output = self.ICL_transformer ( ICL_feature_backup,ICL_feature)
            pre_label = output[:,self.label_index,:]
            pre_label_flattened = pre_label.view(-1, self.hparams.config.hidden_size) 
            cls_fts = output[:,-1,:] 
            pre_label_remapped = self.mimic_classifier(pre_label_flattened) 
            #return-------------------------------------------------------------------------------------------------------------------------------------------------------------
            return None, None,None,None, pre_label_remapped, self.all_labels_groundtruth[:,self.label_index].view(-1, self.hparams.config.num_classes), cls_fts
        elif self.hparams.config.Model_type == 'encoder_mask':
            #random generate mask
            mask_prob_feature = self.hparams.config.mask_prob_feature
            mask_prob_label = self.hparams.config.mask_prob_label
            #two kinds of mask
            self.ehr_mask_index =  torch.tensor(random.sample(list(self.ehr_index), int(len(self.ehr_index)*mask_prob_feature)),device=self.device)
            self.cxr_mask_index = torch.tensor(random.sample(list(self.cxr_index), int(len(self.cxr_index)*mask_prob_feature)),device=self.device)               
            self.label_mask_index = torch.tensor(random.sample(list(self.label_index[:-1]), int(len(self.label_index[:-1])*mask_prob_label)),device=self.device) 
            self.label_mask_index = torch.cat((self.label_mask_index, torch.tensor([self.label_index[-1]]).to(self.device)), dim=0)

            self.mask_ehr_extend = self.mask_ehr.repeat(len(ICL_feature), len(self.ehr_mask_index), 1)
            self.mask_cxr_extend = self.mask_cxr.repeat(len(ICL_feature), len(self.cxr_mask_index), 1)
            self.mask_label_extend = self.mask_label.repeat(len(ICL_feature), len(self.label_mask_index), 1)

            ICL_feature[:,self.ehr_mask_index,:] = self.mask_ehr_extend 
            ICL_feature[:,self.cxr_mask_index,:] = self.mask_cxr_extend 
            ICL_feature[:,self.label_mask_index,:] = self.mask_label_extend
            
            #all the embeddings---------------------------------------------------
            ICL_feature_flattened = ICL_feature.reshape(-1, x+y+1,ICL_feature.shape[-1]) 
            #modality embedding
            ICL_feature_flattened[:,:x,:] = ICL_feature_flattened[:,:x,:] +  self.modality_embedding(torch.zeros(ICL_feature_flattened.shape[0],x).long().to(self.device))
            ICL_feature_flattened[:,x:x+y,:] = ICL_feature_flattened[:,x:x+y,:] +  self.modality_embedding(torch.ones(ICL_feature_flattened.shape[0],y).long().to(self.device))
            #position embedding
            ICL_feature_flattened = self.Position_embedding(ICL_feature_flattened) 
         
            ICL_feature_new = ICL_feature_flattened.reshape(ICL_feature.shape[0],-1, ICL_feature.shape[2])
            #segment embedding
            self.segment_ids = torch.arange(m).unsqueeze(0).repeat(ICL_feature_new.shape[0], 1).to(self.device)
            segment_embedded = self.segment_embedding(self.segment_ids)
            segment_embedded = segment_embedded.repeat_interleave(x+y+z, dim = 1)
            ICL_feature_new = ICL_feature_new + segment_embedded
            
  
            pre_ehr_feature = output[:,self.ehr_mask_index,:].view(-1, self.hparams.config.hidden_size) #(256*49)*768
            pre_ehr_feature = self.ehr_layernorm(pre_ehr_feature)
            pre_cxr_feature = output[:,self.cxr_mask_index,:].view(-1, self.hparams.config.hidden_size) #(256*49)*768
            pre_cxr_feature = self.cxr_layernorm(pre_cxr_feature)

            pre_label = output[:,self.label_index,:]
            cls_fts = output[:,-1,:]
            pre_label_flattened = pre_label.view(-1, self.hparams.config.hidden_size)
            pre_label_remapped = self.mimic_classifier(pre_label_flattened)
            #return------------------------------------------------------------------------------------------
            return pre_ehr_feature, pre_cxr_feature, ICL_feature_backup[:,self.ehr_mask_index,:].view(-1, 768), ICL_feature_backup[:,self.cxr_mask_index,:].view(-1, 768), pre_label_remapped, self.all_labels_groundtruth[:,self.label_index].view(-1, self.hparams.config.num_classes), cls_fts
        elif self.hparams.config.Model_type == 'cross_attention':
            #all the embeddings----------------------------------------------------------------------------------------------------------------------

            ICL_feature_flattened = ICL_feature.reshape(-1, x+y+1,ICL_feature.shape[-1]) 
            #modality embedding
            ICL_feature_flattened[:,:x,:] = ICL_feature_flattened[:,:x,:] +  self.modality_embedding(torch.zeros(ICL_feature_flattened.shape[0],x).long().to(self.device))
            ICL_feature_flattened[:,x:x+y,:] = ICL_feature_flattened[:,x:x+y,:] +  self.modality_embedding(torch.ones(ICL_feature_flattened.shape[0],y).long().to(self.device))
            #position embedding
            ICL_feature_flattened = self.Position_embedding(ICL_feature_flattened) 

            ICL_feature_new = ICL_feature_flattened.reshape(ICL_feature.shape[0],-1, ICL_feature.shape[2])

            self.segment_ids = torch.arange(m-1).unsqueeze(0).repeat(ICL_feature_new.shape[0], 1).to(self.device)
            segment_embedded = self.segment_embedding(self.segment_ids)
            segment_embedded = segment_embedded.repeat_interleave(x+y+1, dim = 1)
            ICL_feature_new[:,:-(x+y+1),:] = ICL_feature_new[:,:-(x+y+1),:] + segment_embedded
     
            ICL_feature_new = self.layer_norm_input(ICL_feature_new)
            src_current = ICL_feature_new[:,-(x+y+1):,:]
            src_NN = ICL_feature_new[:,:-(x+y+1),:]
            output = self.ICL_transformer(src_current, src_current)
            #return
            pre_label = self.mimic_classifier(output[:,-1,:])
            cls_fts = output[:,-1,:]
            return None, None, None, None,pre_label, self.all_labels_groundtruth[:,-1], cls_fts
        elif self.hparams.config.Model_type == 'CLF':
            output = torch.mean(ICL_feature[:,-(x+y+z):,:], dim = 1)
            cls_fts = self.pooler(output)
            pre_label = self.mimic_classifier(cls_fts)
            return None, None,None,None, pre_label, self.all_labels_groundtruth[:,-1], cls_fts
        
        
    def training_step(self, batch, batch_idx):

        pre_ehr_mask, pre_cxr_mask, truth_ehr_mask, truth_cxr_mask, pre_label, truth_label,_ = self(batch)
        self.step_count +=1
        #loss
        criterion = nn.BCELoss()
        loss_label = criterion(pre_label, truth_label)
        total_loss = loss_label
        loss_ehr = loss_cxr = 0.0
        if self.hparams.config.Model_type == 'encoder_mask':
            criterion = nn.MSELoss()
            loss_ehr = criterion(pre_ehr_mask, truth_ehr_mask)
            loss_cxr = criterion(pre_cxr_mask, truth_cxr_mask)
            total_loss = loss_label + self.hparams.config.ehr_cxr_loss_weight*(loss_ehr + loss_cxr)

        pre_label_reshapped = pre_label.view(batch[0].shape[0],-1 ,self.hparams.config.num_classes)
        y_hat = pre_label_reshapped[:,-1].squeeze()
        y_true = batch[1].float().to(y_hat.device)

        self.all_preds_train.append(y_hat)
        self.all_labels_train.append(y_true)
        self.complete_preds_train.append(y_hat[torch.tensor([x == "complete" for x in batch[self.position_flag]])]) 
        self.complete_labels_train.append(y_true[torch.tensor([x == "complete" for x in batch[self.position_flag]])])
        self.missed_cxr_preds_train.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_cxr_labels_train.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_ehr_preds_train.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])
        self.missed_ehr_labels_train.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])   
        #tensorboard
        self.log('train_loss/total_loss', total_loss)
        self.log('train_loss/loss_ehr', loss_ehr)
        self.log('train_loss/loss_cxr', loss_cxr)
        self.log('train_loss/loss_label', loss_label)
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': total_loss}
    
    def validation_step(self, batch,batch_idx):

        pre_ehr_mask, pre_cxr_mask, truth_ehr_mask, truth_cxr_mask, pre_label, truth_label,_ = self(batch)
        #loss
        criterion = nn.BCELoss()
        loss_label = criterion(pre_label, truth_label)
        total_loss = loss_label
        loss_ehr = loss_cxr = 0.0
        if self.hparams.config.Model_type == 'encoder_mask':
            criterion = nn.MSELoss()
            loss_ehr = criterion(pre_ehr_mask, truth_ehr_mask)
            loss_cxr = criterion(pre_cxr_mask, truth_cxr_mask)
            total_loss = loss_label + self.hparams.config.ehr_cxr_loss_weight*(loss_ehr + loss_cxr)

        pre_label_reshapped = pre_label.view(batch[0].shape[0],-1 ,self.hparams.config.num_classes)
        y_hat = pre_label_reshapped[:,-1].squeeze()
        y_true = batch[1].float().to(y_hat.device)

        self.all_preds_val.append(y_hat)
        self.all_labels_val.append(y_true)
        self.complete_preds_val.append(y_hat[torch.tensor([x == "complete" for x in batch[self.position_flag]])]) 
        self.complete_labels_val.append(y_true[torch.tensor([x == "complete" for x in batch[self.position_flag]])])
        self.missed_cxr_preds_val.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_cxr_labels_val.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_ehr_preds_val.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])
        self.missed_ehr_labels_val.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])   
        #tensorboard
        self.log('val_loss/total_loss', total_loss)
        self.log('val_loss/loss_ehr', loss_ehr)
        self.log('val_loss/loss_cxr', loss_cxr)
        self.log('val_loss/loss_label', loss_label)
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': total_loss}
    
    def test_step(self, batch,batch_idx):
        data_each_batch = []
        start_time = time.time()
        pre_ehr_mask, pre_cxr_mask, truth_ehr_mask, truth_cxr_mask, pre_label, truth_label, cls_fts = self(batch)
        end_time = time.time()  
        elapsed_time = (end_time - start_time) * 1000 
        
        
        print('The time cost of one batch is {} ms'.format(elapsed_time))

        pre_label_reshapped = pre_label.view(batch[0].shape[0],-1 ,self.hparams.config.num_classes)
        y_hat = pre_label_reshapped[:,-1].squeeze() 
        y_true = batch[1].float().to(y_hat.device)
        
        self.all_preds_test.append(y_hat)
        self.all_labels_test.append(y_true)
        self.complete_preds_test.append(y_hat[torch.tensor([x == "complete" for x in batch[self.position_flag]])]) 
        self.complete_labels_test.append(y_true[torch.tensor([x == "complete" for x in batch[self.position_flag]])])
        self.missed_cxr_preds_test.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_cxr_labels_test.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[self.position_flag]])])
        self.missed_ehr_preds_test.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])
        self.missed_ehr_labels_test.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[self.position_flag]])])   
        
        
        for i in range(len(y_hat)):
            single_sample_output = {
            "logits": y_hat[i].cpu().numpy(),
            "cls_feats": cls_fts[i].cpu().numpy(),
            "label": y_true[i].cpu().numpy(),
            "flag_pair": batch[self.position_flag][i],
            "patient_id": batch[3][i]
            }            

            data_each_batch.append(single_sample_output)
           

        full_path = self.config.load_path_test
        path = '/home/uceezzz/tool/CVPR2024_model_output' + '/result' + '/'.join(full_path.split('/')[full_path.split('/').index('result')+1:])
        if os.path.exists(path) and self.best_final_matrix==0:
            shutil.rmtree(path)
            self.best_final_matrix = 1
        if not os.path.exists(path):
            os.makedirs(path)

        df = pd.DataFrame(data_each_batch)
        file_path = os.path.join(path, f"{self.best_final_matrix}.pkl")
        df.to_pickle(file_path)
        self.best_final_matrix += 1
    
    
    def training_epoch_end(self, outputs):
        all_auroc = metrics.roc_auc_score( torch.cat(self.all_labels_train).data.cpu().numpy(),torch.cat(self.all_preds_train).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_train).data.cpu().numpy(),torch.cat(self.all_preds_train).data.cpu().numpy() )
        self.log('train_auroc/train_all_auroc', all_auroc)
        self.log('train_auprc/train_all_auprc', all_auprc)

        if len(torch.cat(self.complete_preds_train).data.cpu().numpy()) > 0:
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_train).data.cpu().numpy(),torch.cat(self.complete_preds_train).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_train).data.cpu().numpy(),torch.cat(self.complete_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_complete_auroc', complete_auroc)
            self.log('train_auprc/train_complete_auprc', complete_auprc)
        
        if len(torch.cat(self.missed_cxr_preds_train).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_train).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_train).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score( torch.cat(self.missed_cxr_labels_train).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_missed_cxr_auroc', missed_cxr_auroc)
            self.log('train_auprc/train_missed_cxr_auprc', missed_cxr_auprc)

        if len(torch.cat(self.missed_ehr_preds_train).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_train).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_train).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_train).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_missed_ehr_auroc', missed_ehr_auroc)
            self.log('train_auprc/train_missed_ehr_auprc', missed_ehr_auprc)
        
        final_matrix = all_auroc + all_auprc
        self.log('train_auroc/train_final_matrix', final_matrix)

        self.all_preds_train = []
        self.all_labels_train = []
        self.complete_preds_train = []
        self.complete_labels_train = []
        self.missed_cxr_preds_train = []
        self.missed_cxr_labels_train = []
        self.missed_ehr_preds_train = []
        self.missed_ehr_labels_train = []

        screen = os.environ.get('STY', 'Not available')
        print('server: {} || gpu: No.{} || screen: {}'.format(socket.gethostname(), self.hparams.config.gpu_id, screen))
        
    def validation_epoch_end(self, outputs):

        all_auroc = metrics.roc_auc_score( torch.cat(self.all_labels_val).data.cpu().numpy(),torch.cat(self.all_preds_val).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_val).data.cpu().numpy(),torch.cat(self.all_preds_val).data.cpu().numpy() )
        self.log('val_auroc/val_all_auroc', all_auroc)
        self.log('val_auprc/val_all_auprc', all_auprc)

        if len(torch.cat(self.complete_preds_val).data.cpu().numpy()) > 0:
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_val).data.cpu().numpy(),torch.cat(self.complete_preds_val).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_val).data.cpu().numpy(),torch.cat(self.complete_preds_val).data.cpu().numpy())
            self.log('val_auroc/val_complete_auroc', complete_auroc)
            self.log('val_auprc/val_complete_auprc', complete_auprc)
        
        if len(torch.cat(self.missed_cxr_preds_val).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_val).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_val).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score(torch.cat(self.missed_cxr_labels_val).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_val).data.cpu().numpy() )
            self.log('val_auroc/val_missed_cxr_auroc', missed_cxr_auroc)
            self.log('val_auprc/val_missed_cxr_auprc', missed_cxr_auprc)

        if len(torch.cat(self.missed_ehr_preds_val).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_val).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_val).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_val).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_val).data.cpu().numpy())
            self.log('val_auroc/val_missed_ehr_auroc', missed_ehr_auroc)
            self.log('val_auprc/val_missed_ehr_auprc', missed_ehr_auprc)
        
        current_final_matrix = all_auroc + all_auprc
        self.log('val_auroc/val_final_matrix', current_final_matrix)

        if current_final_matrix > self.best_final_matrix:
            self.best_final_matrix = current_final_matrix
            if not self.hparams.config.save_best:
                print('get new best final matrix: {}'.format(self.best_final_matrix))
            with open(f'{self.trainer.logger.log_dir}' + '/best_result.txt', 'w') as f:              
                f.write('best_final_matrix: ' + (format(self.best_final_matrix, '.3f')) + '\n')
                f.write('auroc_all: ' + (format(all_auroc, '.3f') if 'all_auroc' in locals() else 'None') + '\n')
                f.write('auprc_all: ' + (format(all_auprc, '.3f') if 'all_auprc' in locals() else 'None') + '\n')
                f.write('auroc_complete: ' + (format(complete_auroc, '.3f') if 'complete_auroc' in locals() else 'None') + '\n')
                f.write('auprc_complete: ' + (format(complete_auprc, '.3f') if 'complete_auprc' in locals() else 'None') + '\n')
                f.write('auroc_missed_cxr: ' + (format(missed_cxr_auroc, '.3f') if 'missed_cxr_auroc' in locals() else 'None') + '\n')
                f.write('auprc_missed_cxr: ' + (format(missed_cxr_auprc, '.3f') if 'missed_cxr_auprc' in locals() else 'None') + '\n')
                f.write('auroc_missed_ehr: ' + (format(missed_ehr_auroc, '.3f') if 'missed_ehr_auroc' in locals() else 'None') + '\n')
                f.write('auprc_missed_ehr: ' + (format(missed_ehr_auprc, '.3f') if 'missed_ehr_auprc' in locals() else 'None') + '\n')
        else:
            if not self.hparams.config.save_best:
                print('please wait for new best final matrix...')

        

        self.all_preds_val = []
        self.all_labels_val = []
        self.complete_preds_val = []
        self.complete_labels_val = []
        self.missed_cxr_preds_val = []
        self.missed_cxr_labels_val = []
        self.missed_ehr_preds_val = []
        self.missed_ehr_labels_val = []
        
    def test_epoch_end(self, outputs):
        all_auroc = metrics.roc_auc_score( torch.cat(self.all_labels_test).data.cpu().numpy(),torch.cat(self.all_preds_test).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_test).data.cpu().numpy(),torch.cat(self.all_preds_test).data.cpu().numpy() )
        self.log('val_auroc/val_all_auroc', all_auroc)
        self.log('val_auprc/val_all_auprc', all_auprc)

        if len(torch.cat(self.complete_preds_test).data.cpu().numpy()) > 0:
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_test).data.cpu().numpy(),torch.cat(self.complete_preds_test).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_test).data.cpu().numpy(),torch.cat(self.complete_preds_test).data.cpu().numpy())
            self.log('val_auroc/val_complete_auroc', complete_auroc)
            self.log('val_auprc/val_complete_auprc', complete_auprc)
        
        if len(torch.cat(self.missed_cxr_preds_test).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_test).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_test).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score(torch.cat(self.missed_cxr_labels_test).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_test).data.cpu().numpy() )
            self.log('val_auroc/val_missed_cxr_auroc', missed_cxr_auroc)
            self.log('val_auprc/val_missed_cxr_auprc', missed_cxr_auprc)

        if len(torch.cat(self.missed_ehr_preds_test).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_test).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_test).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_test).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_test).data.cpu().numpy())
            self.log('val_auroc/val_missed_ehr_auroc', missed_ehr_auroc)
            self.log('val_auprc/val_missed_ehr_auprc', missed_ehr_auprc)
        
        
        
        
        
        
        
        
        
    def configure_optimizers(self): 
        max_epoch = self.hparams.config.max_epoch
        max_steps = int(self.hparams.config.subsample_ratio*max_epoch*18845/(self.hparams.config.per_gpu_batchsize)) 
        step_per_epoch = int(max_steps/max_epoch) #2
        if step_per_epoch == 0:
            step_per_epoch = 1
        lr = self.hparams.config.learning_rate
        end_lr = self.hparams.config.end_lr
        wd = self.hparams.config.weight_decay
        lr_mult = self.hparams.config.lr_mult
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
            ]
        head_names = ["cxr_classifier", "mimic_classifier", "ehr_layernorm", "cxr_layernorm"]
        prompt_name = "prompt"
        names = [n for n, p in self.named_parameters()]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },            
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
            
       
        def lr_lambda(current_step):

            if current_step < 10*step_per_epoch:
                return 1.0
            else:
                return 0.5
            
            
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )
    