import os
import sys
import json
import random
import argparse
import numpy as np
from easydict import EasyDict as edict


def parse_with_config(args):

    file_cfg = edict(json.load(open(args.config)))
    cmd_cfg_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                        if arg.startswith('--')}

    ### load default run_cfg
    run_cfg = edict(json.load(open(file_cfg.run_cfg.default)))
    ### overwrite run_cfg by config file
    run_cfg.update(file_cfg.run_cfg)
    ### overwrite run_cfg by cmd
    for k in cmd_cfg_keys:
        if k in run_cfg:
            run_cfg[k] = getattr(args,k)


    # if file_cfg['model_cfg']: must have

    ### load default model_cfg
    model_cfg = edict(json.load(open(file_cfg.model_cfg.default)))
    ### overwrite model_cfg by config file
    model_cfg.update(file_cfg.model_cfg)



    if args.pretrain_dir:
        ### load pretrained model_cfg
        pretrain_model_cfg = edict(json.load(open(os.path.join(args.pretrain_dir,'log','hps.json')))).model_cfg
        ### overwite inherit_keys
        global_inherit_keys = ['vision_encoder_type','pool_video']
        inherit_keys = list(set(global_inherit_keys)|set(model_cfg.inherit_keys))
        inherit_model_cfg = edict({k:v for k,v in pretrain_model_cfg.items() if k in inherit_keys})
        model_cfg.update(inherit_model_cfg)

    # else:
    #     ### load from specific path
    #     assert args.model_cfg_file
    #     model_cfg = edict(json.load(open(args.model_cfg_file)))

    ### overwrite model_cfg by cmd
    for k in cmd_cfg_keys:
        if k in model_cfg:
            model_cfg[k] = getattr(args,k)


    ### load data_cfg from config file
    data_cfg = file_cfg['data_cfg']

    ### overwrite data_cfg by cmd, only valid when single dataset
    for k in cmd_cfg_keys:
        if k.startswith('train_'):
            assert len(data_cfg.train)==1 or k in ['train_batch_size','train_task']

            if k=='train_epoch':
                data_cfg.train[0].epoch = args.train_epoch
            elif k=='train_steps':
                data_cfg.train[0].steps = args.train_steps
            elif k=='train_vision_sample_num':
                data_cfg.train[0].vision_sample_num = args.train_vision_sample_num
            elif k=='train_batch_size':
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].batch_size = args.train_batch_size
            elif k=='train_task':
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].task = args.train_task
        elif k.startswith('test'):
            # assert len(data_cfg.val)==1
            for i in range(len(data_cfg.val)):
                if k=='test_batch_size':
                    data_cfg.val[i].batch_size = args.test_batch_size
                elif k=='test_vision_sample_num':
                    data_cfg.val[i].vision_sample_num = args.test_vision_sample_num
                elif k=='test_task':
                    data_cfg.val[i].task = args.test_task

        elif k=='vision_transforms':
            assert len(data_cfg.train)==1
            assert len(data_cfg.val)==1
            data_cfg.train[0]['vision_transforms'] = args.vision_transforms
            data_cfg.val[0]['vision_transforms'] = args.vision_transforms


    ### general configurations for different models, transmit directly from run_cfg

    # model_cfg.vision_resolution = run_cfg.vision_resolution
    # model_cfg.max_length = run_cfg.max_length
    # model_cfg.min_length = run_cfg.min_length
    # model_cfg.max_output_txt_len = run_cfg.max_output_txt_len
    # model_cfg.beam_size = run_cfg.beam_size
    # model_cfg.prompt = run_cfg.prompt
    # model_cfg.checkpointing = run_cfg.checkpointing
    # model_cfg.frozen_vision = run_cfg.frozen_vision
    # model_cfg.captioner_mode = run_cfg.captioner_mode
    # model_cfg.generate_nums = run_cfg.generate_nums


    ### special rules

    if model_cfg.checkpointing:
        run_cfg.use_ddp = False

    data_cfg.concatenated_nums = getattr(model_cfg,'concatenated_nums',1) ### for cosa training

    max_vision_sample_num = compute_max_vision_sample_num_for_position_embeddings(data_cfg)
    max_audio_sample_num = compute_max_audio_sample_num_for_position_embeddings(data_cfg)

    model_cfg.max_vision_sample_num = max_vision_sample_num
    model_cfg.max_audio_sample_num = max_audio_sample_num

    if run_cfg.bf16:
        run_cfg.fp16 = False
    ### output cfg

    output_cfg = edict({'run_cfg':run_cfg,
                        'model_cfg':model_cfg,
                        'data_cfg':data_cfg,
                        'local_rank':args.local_rank})

    return output_cfg


def get_args():

    class Args:
        vision_resolution=224
        local_rank=0
        checkpoint="models/vast/weights/vast.pth"
        output_dir="./"
        gradient_accumulation_steps=1
        learning_rate=1e-4
        clip_lr=5e-7
        clip_lr_text=5e-7
        optim='adam'
        betas=[0.9, 0.98]
        dropoutt=0.1
        weight_decay=0.01
        grad_norm=5.0
        warmup_ratio=0.1
        opt_model=None
        llm_model=None
        resume=False
        scheduler=42
        fp16=True
        bf16=False
        config="models/vast/config/caption-msrvtt.json"
        zero_shot=False
        scheduler='warmup_linear'
        max_generation_len=40
        max_length=30
        min_length=8
        max_output_txt_len=256
        amp='apex'
        train_id=''
        test_id=''
        train_task=''
        test_task=''
        test_batch_size=-1
        max_text_tokens=40
        train_batch_size=-1
        checkpointing=True
        frozen_vision=False
        scst_finetuning=False
        use_proposal_conv=True
        ret_bidirection_evaluation=False
        trainer_type=""
        itm_rerank_num=50
        itm_ratio=1.0
        save_best=True
        train_epoch=-1
        contra_ratio=1.0
        train_steps=-1
        train_vision_sample_num=-1
        test_vision_sample_num=-1
        train_audio_sample_num=-1
        log_steps=-1
        test_audio_sample_num=-1
        concatenated_nums=1
        vision_encoder_type='clip_vit_base_16'
        frame_embedding_type=''
        loss_type=''
        vision_transforms='none'
        multimodal_encoder_type='bert_base_uncased'
        num_train_steps=0
        huggingface_trainer=False
        pretrain_dir=None
        deepspeed=""
        prompt=None
        model_cfg_file=""
        llm_type=""
        dual_softmax=False
        pool_video=False
        use_flash_attn=False
        qformer_question=False
        frozen_llm=True
        use_deepspeed=False
        captioner_mode=False
        qformer_text_input=True
        evaluate_ret_text=False
        pool_vision=False
        first_eval=True
        vision_perceiver_query_num=-1
        remove_before_ckpt=True
        dataset_mix_type='random'
        valid_freq=10
        new_params_name=[]
        new_lr=0.0
        beam_size=3
        generate_nums=1
        beam_size_qa=1
        contra_dim=512
        mode='testing'
        perceiver_mode=''
        vision_cut_frames=-1

    return parse_with_config(Args)

def compute_max_vision_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    vision_sample_num_ls_train=[]
    for d_cfg in data_cfg_train:
        vision_sample_num = d_cfg.get('vision_sample_num',1)
        vision_sample_num_ls_train.append(vision_sample_num * data_cfg.concatenated_nums)


    data_cfg_val = data_cfg.val
    vision_sample_num_ls_val=[]
    for d_cfg in data_cfg_val:
        vision_sample_num = d_cfg.get('vision_sample_num',1)
        vision_sample_num_ls_val.append(vision_sample_num )


    max_vision_sample_num = max(vision_sample_num_ls_train) if vision_sample_num_ls_train else max(vision_sample_num_ls_val)

    assert max_vision_sample_num  > 0
    return max_vision_sample_num

def compute_max_audio_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    audio_sample_num_ls_train=[]
    for d_cfg in data_cfg_train:
        audio_sample_num = d_cfg.get('audio_sample_num',1)
        audio_sample_num_ls_train.append(audio_sample_num * data_cfg.concatenated_nums)


    data_cfg_val = data_cfg.val
    audio_sample_num_ls_val=[]
    for d_cfg in data_cfg_val:
        audio_sample_num = d_cfg.get('audio_sample_num',1)
        audio_sample_num_ls_val.append(audio_sample_num )


    max_audio_sample_num = max(audio_sample_num_ls_train) if audio_sample_num_ls_train else max(audio_sample_num_ls_val)

    assert max_audio_sample_num  > 0
    return max_audio_sample_num