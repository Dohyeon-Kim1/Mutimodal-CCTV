import os
import json
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.vast.general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from torch.nn import LayerNorm as LayerNorm
from easydict import EasyDict as edict

class VAST(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.construct_vision_encoder()
        self.construct_audio_encoder()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio   
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len





   
    def construct_multimodal_encoder(self):    
        
        from transformers import BertConfig, BertForMaskedLM, BertTokenizer

        bertconfig = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
        self.multimodal_encoder = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased",config = bertconfig )
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        self.text_masker = TokenMasker(mask_token = self.multimodal_encoder.tokenizer.mask_token_id, range_start=106, range_end = 30522)

        

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]


        elif key == 'caption_tokens':

            caption_tokens = self.multimodal_encoder.tokenizer(batch.raw_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt").to(torch.device('cuda'))
         
            batch[key] = caption_tokens
        
        elif key == 'subtitle_tokens':
         
            subtitle_tokens = self.multimodal_encoder.tokenizer(batch.raw_subtitles,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_subtitle_len,
                                                    return_tensors="pt")
            subtitle_tokens = subtitle_tokens.to(torch.device('cuda'))
            batch[key] = subtitle_tokens
                                        

        elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        
        elif key == 'audio_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.audio_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens

        elif key == 'omni_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.omni_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_omni_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output



        elif key == 'vision_caption_output':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output


        elif key == 'audio_caption_output':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output

       
        elif key == 'subtitle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = subtitle_output
  

        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output

        elif key == 'audio_output':
            audio_output = self.forward_audio_encoder(batch.audio_spectrograms) 
            batch[key] = audio_output


        elif key == 'condition_feats_v':
            vision_output = self.batch_get(batch, 'vision_output')
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v

        elif key == 'condition_feats_a':
            audio_output = self.batch_get(batch, 'audio_output')
            condition_feats_a = self.get_multimodal_forward_input_audio(audio_output)
            batch[key] = condition_feats_a

        elif key == 'condition_feats_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle_output)
            batch[key] = condition_feats_s

        elif key == 'condition_feats_va':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_va = torch.cat((condition_feats_v, condition_feats_a),dim=1)
            batch[key] = condition_feats_va

        elif key == 'condition_feats_vs':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vs = torch.cat((condition_feats_v, condition_feats_s),dim=1)
            batch[key] = condition_feats_vs

        elif key == 'condition_feats_vas':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s),dim=1)
            batch[key] = condition_feats_vas


        elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v

        elif key == 'feat_a':
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output_pooled)
            feat_a = F.normalize(feat_a,dim=-1)
            batch[key] = feat_a       

        elif key == 'feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s(subtitle_output_pooled) 
            feat_s = F.normalize(feat_s,dim=-1)
            batch[key] = feat_s

        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_va':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_va = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
            feat_va = self.contra_head_va(feat_va)
            feat_va = F.normalize(feat_va,dim=-1)
            batch[key] = feat_va

        elif key == 'feat_vs': 
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vs = torch.cat((vision_output_pooled, subtitle_output_pooled), dim=1)
            feat_vs = self.contra_head_vs(feat_vs)
            feat_vs = F.normalize(feat_vs,dim=-1) 
            batch[key] = feat_vs

        elif key == 'feat_vas':     
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
            feat_vas = self.contra_head_vas(feat_vas)
            feat_vas = F.normalize(feat_vas,dim=-1)
            batch[key] = feat_vas  

        elif key == 'feat_t_omni_caption':
            caption_tokens = self.batch_get(batch, 'omni_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_audio_caption':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        return batch[key] 


    def forward(self, batch, task, compute_loss=True):
        batch = edict(batch)
        ### vast-27m pretraining
        if 'vision_captions' in batch or 'audio_captions' in batch or 'omni_captions' in batch:
            assert compute_loss
            return self.forward_vast27m(batch, task)


        ### other datasets pretraining or finetuning
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap(batch, task, compute_loss=compute_loss)
                output_ls.append(cap_dict)

            elif task.startswith('qa'):
                qa_dict = self.forward_qa(batch, task, compute_loss=compute_loss)
                output_ls.append(qa_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict


    def forward_vast27m(self, batch, task):
     
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret_vast27m(batch, task)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap_vast27m(batch, task)
                output_ls.append(cap_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
            
        slice_output = self.multimodal_encoder.bert(input_ids = slice_input_ids,
                                                    attention_mask = slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores

    def forward_cap(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]

        if compute_loss:

            loss_dict = {}
            loss_ls_cap = []

            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])

            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                                attention_mask = attention_mask,
                                                encoder_hidden_states=condition_feats,
                                                labels = txt_labels)
                loss_ls_cap.append(output.loss)

            loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
            loss_dict['loss_cap'] = loss_cap
            return loss_dict

        else:
            evaluation_dict = {}
            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')

                batch_size = condition_feats.shape[0]
                if self.config.captioner_mode:
                    batch_size *=self.config.generate_nums

                init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
                init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)
                
                if self.config.captioner_mode:
                    condition_feats = condition_feats.unsqueeze(1).expand(-1, self.config.generate_nums,-1,-1).reshape(-1,*condition_feats.shape[1:])
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            do_sample = True,
                                                            top_k = 10,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=self.max_caption_len,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 

                                                        
                else:
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                        attention_mask=init_attention_mask,
                                        encoder_hidden_states=condition_feats,
                                        max_new_tokens=self.max_caption_len,
                                        num_beams=self.beam_size,
                                        eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                        pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id,
                                        length_penalty=0.6) 
                                                        
                outputs_newgen = outputs[:,1:]
                captions = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                evaluation_dict[f'generated_captions_{task}'] = captions

            return evaluation_dict



    def forward_cap_vast27m(self, batch, task):
        subtasks = task.split('%')[1:]
        loss_dict = {}
        loss_ls_cap = []

        for task in subtasks:
            assert task in ['tv','ta','tva','tvs','tvas']
            if task == 'tv':
                caption_tokens = self.batch_get(batch, 'vision_caption_tokens')

            elif task == 'ta':
                caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                
            else:
                caption_tokens = self.batch_get(batch, 'omni_caption_tokens')

            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])
            condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            output = self.multimodal_encoder(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            encoder_hidden_states=condition_feats,
                                            labels = txt_labels)
            loss_ls_cap.append(output.loss)

        loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
        loss_dict['loss_cap'] = loss_cap
        return loss_dict
          