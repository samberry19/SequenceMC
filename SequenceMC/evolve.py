import numpy as np
import pandas as pd

from bioviper import msa, phylo
from samplers import BaseSampler

def Evolver(f, tree, N, starting_seq, T=1):
    
    '''
    Evolve sequences along a phylogeny by sampling from a DCA model.
    '''
    
    S = []
    
    L = len(starting_seq)

    for clade in tree._biopython.clade:
        N_iterations = int(N*clade.branch_length)
        sampler = BaseSampler(f, L, 1, T=T, record_freq=1, initialization="defined", starting_seq=starting_seq)
        sampler.run(N_iterations, parallel_method="None", progress=False)

        for subclade in clade.clades:

            S = phylogenetic_mcmc(subclade, N, f=f, L=L,
                                      starting_seq=sampler.log[0][-1], seq_list=S, T=T)
            
    return msa.MultipleSequenceAlignment(smc.default_aa_alphabet[np.array(S)])



def phylogenetic_mcmc(clade, N, f, L, starting_seq, seq_list, T=1):
    
    '''
    Helper function to run recursively within DCAEvolver
    '''
    
    if clade.is_terminal():
        
        seq_list.append(starting_seq)
        
        return seq_list
    
    else:
        
        seqs = []
        
        for subclade in clade.clades:

            N_iterations = min(int(N*clade.branch_length), 1)
            
            sampler = BaseSampler(f, L, 1, T=T, record_freq=1, initialization="defined", 
                                  starting_seq=starting_seq)
            
            sampler.run(N_iterations, parallel_method="None", progress=False)
            
            seq_list = phylogenetic_mcmc(subclade, N, f, L, starting_seq=sampler.log[0][-1], seq_list=seq_list)
            
        return seq_list