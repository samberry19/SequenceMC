import numpy as np
import pandas as pd

from bioviper import msa, phylo
from .samplers import BaseSampler
from .utils import default_aa_alphabet

def Evolver(f, tree, N, starting_seq, T=1, pos_constraint=None):
    
    '''
    Evolve sequences along a phylogeny by sampling from a DCA model.

    Has a key parameter *N* which represents the "evolutionary timescale" - e.g. how many generations per one unit "branch length"
    on the tree.
    '''
    
    S = []
    
    L = len(starting_seq)

    for clade in tree._biopython.clade:
        N_iterations = int(N*clade.branch_length)
        sampler = BaseSampler(f, L, 1, T=T, record_freq=1, initialization="defined", starting_seq=starting_seq, pos_constraint=pos_constraint)
        sampler.run(N_iterations, parallel_method="None", progress=False)

        for subclade in clade.clades:

            S = phylogenetic_mcmc(subclade, N, f=f, L=L,
                                      starting_seq=sampler.log[0][-1], seq_list=S, T=T, pos_constraint=pos_constraint)
            
    return msa.MultipleSequenceAlignment(default_aa_alphabet[np.array(S)])



def phylogenetic_mcmc(clade, N, f, L, starting_seq, seq_list, T=1, pos_constraint=None):
    
    '''
    Helper function to run recursively within DCAEvolver
    '''
    
    if clade.is_terminal():
        
        seq_list.append(starting_seq)
        
        return seq_list
    
    else:
        
        seqs = []
        
        for subclade in clade.clades:

            N_iterations = max(int(N*clade.branch_length), 1)
            
            sampler = BaseSampler(f, L, 1, T=T, record_freq=1, initialization="defined", 
                                  starting_seq=starting_seq, pos_constraint=pos_constraint)
            
            sampler.run(N_iterations, parallel_method="None", progress=False)
            
            seq_list = phylogenetic_mcmc(subclade, N, f, L, starting_seq=sampler.log[0][-1], seq_list=seq_list, T=T, pos_constraint=pos_constraint)
            
        return seq_list