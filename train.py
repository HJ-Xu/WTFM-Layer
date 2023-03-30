import os
import sys
from itertools import permutations
import os.path as osp
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio

sys.path.append(osp.join(os.getcwd(),'src'))
import diffusion_net

from remesh_dataset import MatchingDataset
from diffusion_net.utils import Meyer


# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--train_dataset", type=str, default = 'faust_5k', help="faust_5k, scape_5k")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz', 'hks', 'wks') default: wks", default = 'wks')
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--k_eig', type=int, default=128)
parser.add_argument('--n_fmap', type=int, default=50)
parser.add_argument('--Nf', type=int, default=5)
args = parser.parse_args()

# system things
# dtype = torch.float32

# training settings
device = torch.device('cuda:0')
train = not args.evaluate
lr = 1e-3
decay_every = 50
decay_rate = 0.5
C_in={'xyz':3, 'hks':16,'wks':128, 'shot':352}[args.input_features] # dimension of input features
C_out=128


# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join(base_path, 'data', args.train_dataset)
pretrain_path = osp.join(base_path, "saved_models/faust/ckpt_ep0.pth")

# === Create the model
model = diffusion_net.layers.WTFMNet(C_in=C_in,  C_out=C_out, n_fmap=args.n_fmap, is_mwp=True)
model = model.to(device)


# === Optimize
optimizer = torch.optim.Adam(model.parameters())

# def axio_MWP(massvec_x,evecs_x,gs_x,evecs_y,gs_y,T,num_iter=5):
#     # input: 
#     #   massvec_x: [M,]
#     #   evecs_x/y: [M/N,Kx/Ky]
#     #   gs_x/y: [Nf,Kx/Ky]
#     #   T: [M,]
#     gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
#     gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]
#     Nf=gs_x.size(0)
    
#     for it in range(num_iter):
#         C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(evecs_y[T,:]))
#         C_new=torch.zeros_like(C)
        
#         for s in range(Nf):
#             C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
        
#         T=nearest_neighbor(evecs_x,evecs_y@C.t())
    
#     return C_new, T
        
def train_epoch(epoch):
    total_loss = 0.0
    total_num = 0

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    for data in tqdm(train_loader):

        # Get data
        descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y=data
            
        
        #compute Meyer filters
        gs_x=Meyer(evals_x[args.n_fmap-1],Nf=args.Nf)(evals_x[:args.n_fmap])
        gs_y=Meyer(evals_y[args.n_fmap-1],Nf=args.Nf)(evals_y[:args.n_fmap])
        
        
        # Move to device
        descs_x, massvec_x, evals_x = descs_x.to(device), massvec_x.to(device), evals_x.to(device)
        evecs_x, gs_x, gradX_x, gradY_x = evecs_x.to(device), gs_x.to(device), gradX_x.to(device), gradY_x.to(device)

        descs_y, massvec_y, evals_y = descs_y.to(device), massvec_y.to(device), evals_y.to(device)
        evecs_y, gs_y, gradX_y, gradY_y = evecs_y.to(device), gs_y.to(device), gradX_y.to(device), gradY_y.to(device)
        
        # Apply the model
        loss, C = model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
                descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)

        # Evaluate loss
        loss.backward()
        
        # track accuracy
        total_loss+=loss.item()
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()
        
        if total_num%100==0:
            print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
            total_loss=0.0
            total_num=0

def evaluate(dataset):

    dataset_path = osp.join(base_path, 'data', dataset)
    test_dataset = MatchingDataset(dataset_path, train=False, k_eig=args.k_eig, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=None)
    results_dir=osp.join(pretrain_path.split('c')[0], dataset)
    diffusion_net.utils.ensure_dir_exists(results_dir)


    file=osp.join(dataset_path,'files_test.txt')
    with open(file, 'r') as f:
        names = [line.rstrip() for line in f]

    combinations = list(permutations(range(len(names)), 2))

    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

    model.eval()
    with torch.no_grad():
        count=0
        for data in tqdm(test_loader):

            # Get data
            descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x,\
                descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y=data
                
            
            #compute Meyer filters
            gs_x=Meyer(evals_x[args.n_fmap-1],Nf=args.Nf)(evals_x[:args.n_fmap])
            gs_y=Meyer(evals_y[args.n_fmap-1],Nf=args.Nf)(evals_y[:args.n_fmap])
            
            
            # Move to device
            descs_x, massvec_x, evals_x = descs_x.to(device), massvec_x.to(device), evals_x.to(device)
            evecs_x, gs_x, gradX_x, gradY_x = evecs_x.to(device), gs_x.to(device), gradX_x.to(device), gradY_x.to(device)

            descs_y, massvec_y, evals_y = descs_y.to(device), massvec_y.to(device), evals_y.to(device)
            evecs_y, gs_y, gradX_y, gradY_y = evecs_y.to(device), gs_y.to(device), gradX_y.to(device), gradY_y.to(device)
            
            
            # Apply the model
            loss, C = model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
                    descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)
            

            T=torch.argmax(model.p,dim=1)
            
            idx1,idx2=combinations[count]
            count+=1

            results_path=osp.join(results_dir,names[idx1]+'_'+names[idx2]+'.mat')
            Tnn=diffusion_net.utils.nn_search(evecs_y[:,:args.n_fmap]@C.t(),evecs_x[:,:args.n_fmap])
            sio.savemat(results_path, {'C':C.to('cpu').numpy().astype(np.float32),
                                       'T':T.to('cpu').numpy().astype(np.int64)+1,
                                       'Tnn':Tnn.to('cpu').numpy().astype(np.int64)+1}) # T: convert to matlab index


if __name__== "__main__" :
    # Load the train dataset
    if train:
        train_dataset = MatchingDataset(dataset_path, train=True, k_eig=args.k_eig, use_cache=True)
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
        model_save_dir=osp.join(dataset_path,'saved_models',folder_str)
        diffusion_net.utils.ensure_dir_exists(model_save_dir)

        print("Training...")

        for epoch in range(args.n_epoch):
            train_acc = train_epoch(epoch)
            
            model_save_path=osp.join(model_save_dir,'ckpt_ep{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_save_path)

        print(" ==> saving last model to " + model_save_path)
        torch.save(model.state_dict(), model_save_path)

    evaluate('faust_5k')
    evaluate('faust_a')
    evaluate('scape_5k')
    evaluate('scape_a')
