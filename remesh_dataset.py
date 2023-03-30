# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 13:21:22 2022

@author: Michael
"""

import os
import sys
from itertools import permutations

import potpourri3d as pp3d

import torch
from torch.utils.data import Dataset

import os.path as osp

sys.path.append(osp.join(os.getcwd(),'src'))
import diffusion_net



class MatchingDataset(Dataset):
    def __init__(self,root_dir,train,k_eig=128,use_cache=True):
        super().__init__()

        self.root_dir=root_dir
        self.train=train
        self.k_eig=k_eig
        self.cache_dir=osp.join(root_dir,'wks_cache')
        self.op_cache_dir=osp.join(root_dir,'wks_op_cache')

        # store in memory
        self.verts_list=[]
        self.faces_list=[]
        self.descs_list=[]
        

        if use_cache:
            train_cache=osp.join(self.cache_dir,'train.pt')
            test_cache=osp.join(self.cache_dir,'test.pt')
            load_cache=train_cache if self.train else test_cache
            print('using dataset cache path:' + str(load_cache))

            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                # to-do load data
                self.verts_list, self.faces_list, self.descs_list, \
                self.massvec_list, self.evals_list, self.evecs_list,\
                     self.gradX_list, self.gradY_list, self.L_list = torch.load(load_cache)
                
                self.combinations = list(permutations(range(len(self.evals_list)), 2))
                return
            print("  --> dataset not in cache, repopulating")
        
        self.files='files_train.txt' if self.train else 'files_test.txt'
        self.files=osp.join(root_dir,self.files)
        
        #read files names
        with open(self.files, 'r') as f:
            self.names = [line.rstrip() for line in f]
        
        # read file and process
        for name in self.names:
            mesh_path=osp.join(root_dir,'shapes',name+'.off')
            verts,faces=pp3d.read_mesh(mesh_path)
        
            # convert to torch
            verts=torch.from_numpy(verts).float()
            faces=torch.from_numpy(faces)
            
            #scale the total_area to 1
            verts=diffusion_net.geometry.normalize_positions(verts,faces,method='mean', scale_method='area')

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            # self.descs_list.append(descs)
            
        
        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, \
            self.evals_list, self.evecs_list, self.gradX_list,\
                 self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, \
                     k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)


        #hks descriptors
        for s in range(len(self.evals_list)):
            evals=self.evals_list[s]
            evecs=self.evecs_list[s]
            # self.descs_list.append(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16))
            self.descs_list.append(diffusion_net.utils.auto_WKS(evals, evecs, 128).float())
            
        
        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.descs_list, \
                self.massvec_list, self.evals_list, self.evecs_list,\
                     self.gradX_list, self.gradY_list, self.L_list), load_cache)
        
        self.combinations = list(permutations(range(len(self.evals_list)), 2))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        # get pair data 
        #   descs: [N,P]
        #   massvec: [N,]
        #   evals: [K,]
        #   evecs: [N,K]
        #   gs: [Nf,K,]
        idx1, idx2 = self.combinations[index]
        descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x=self.get_data(idx1)
        descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y=self.get_data(idx2)


        return descs_x,massvec_x,evals_x,evecs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gradX_y,gradY_y

    def get_data(self, idx):
        return self.descs_list[idx],self.massvec_list[idx],\
            self.evals_list[idx],self.evecs_list[idx],\
                self.gradX_list[idx],self.gradY_list[idx]
    



                








