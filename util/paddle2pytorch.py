import paddle
import torch
import numpy as np
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd', type=str, default="checkpoint-799.pd")
    parser.add_argument('--pth', type=str, default="mae_pretrain_vit_base.pth")
    args = parser.parse_args()

    pd_path = args.pd
    pt_path = args.pth
    pt_state = torch.load(pt_path)
    print(pt_state["model"].keys())
    print(len(pt_state["model"]))
    print("=========================")
    pd_state = paddle.load(pd_path)

    mirror = torch.load(pt_path)
    #print(pd_state["model"].keys())

    # if pt.keys all in pd.keys
    # n = 0
    # for k in pt_state["model"].keys():
    #     if k in pd_state["model"].keys():
    #         n = n + 1
    #     else:
    #         print(k)

    # print(n)

    # test
    # pt_fcs = pt_state["model"]["blocks.0.mlp.fc1.weight"]
    # pd_fcs = pd_state["model"]["blocks.0.mlp.fc1.weight"]

    # print(pt_fcs.shape, pd_fcs.shape)

    # try:
    #     pt_fcs = pt_fcs = pd_fcs
    # except:
    #     print("error!")    
    
    # print(pt_fcs.shape, pd_fcs.shape)
    # print("pt_fcs=", pt_fcs)
    # print("pd_fcs=", pd_fcs)

    for k in ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'blocks.0.norm1.weight', 'blocks.0.norm1.bias', 'blocks.0.attn.qkv.weight', 'blocks.0.attn.proj.weight', 'blocks.0.attn.proj.bias', 'blocks.0.norm2.weight', 'blocks.0.norm2.bias', 'blocks.0.mlp.fc1.weight', 'blocks.0.mlp.fc1.bias', 'blocks.0.mlp.fc2.weight', 'blocks.0.mlp.fc2.bias']:
        print(k, pt_state['model'][k].shape, pd_state['model'][k].shape, pt_state['model'][k].dtype ,pd_state['model'][k].dtype)

    # print(pt_state['model']['patch_embed.proj.weight'])

    # a = pt_state['model']['blocks.0.norm1.bias']
    # print(a.shape, a.type(), a)
    # tmp = a.numpy().T
    # b = torch.tensor(tmp, dtype=torch.float32)
    # print(b.shape, b.type(), b)

    times = 0 
    for k in pt_state['model'].keys():
        if ('fc' in k) or ('attn.qkv' in k) or ('attn.proj' in k) or ('head' in k):
            temp = pd_state['model'][k]
            temp = temp.numpy().T
            pt_state['model'][k] = torch.tensor(temp, dtype=torch.float32)
            if not torch.equal(mirror['model'][k],(pt_state['model'][k])):
                times = times +1
        else:
            temp = pd_state['model'][k]
            temp = temp.numpy()
            pt_state['model'][k] = torch.tensor(temp, dtype=torch.float32)
            if not torch.equal(mirror['model'][k],(pt_state['model'][k])):
                times = times +1
    
    print(f" number of weights loaded from checkpoint is {times}")
    
    for k in ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'blocks.0.norm1.weight', 'blocks.0.norm1.bias', 'blocks.0.attn.qkv.weight', 'blocks.0.attn.proj.weight', 'blocks.0.attn.proj.bias', 'blocks.0.norm2.weight', 'blocks.0.norm2.bias', 'blocks.0.mlp.fc1.weight', 'blocks.0.mlp.fc1.bias', 'blocks.0.mlp.fc2.weight', 'blocks.0.mlp.fc2.bias']:
        print(k, pt_state['model'][k].shape, pd_state['model'][k].shape, pt_state['model'][k].dtype ,pd_state['model'][k].dtype)

    torch.save(pt_state,'checkpoint-799_pd2torch.pth')
    print('Saved checkpoint!')
