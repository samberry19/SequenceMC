'''
samplers.py
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from tqdm import tqdm
from bioviper import msa
from evcouplings.couplings import CouplingsModel

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
import os

from .utils import one_hot_encode, mutate, SequencePCA, to_numeric, one_hot_decode, default_aa_alphabet, gen_all_muts, get_random_mutation_order
from .bias import LinearDistanceRestraint, LatentVoyagerPotential
from .dca import DCAEnergy, potts_model_hamiltonian, calc_dca_conditionals


from SequenceMC import default_aa_alphabet
import os


class BaseSampler:

    '''A base class to sample sequences from energy-based models given a hamiltonian function. Comes with Metropolis sampling
        built-in, subclasses can custom define Gibbs sampling functions for particular kinds of hamiltonians.'''

    def __init__(self, hamiltonian, L, N_chains, T=1, record_freq=1000, independent=False,
                 extra_potential=None, pos_constraint=None, one_hot=False, alphabet=default_aa_alphabet,
                 initialization='random', starting_seq=None):

        ''' Initialize the DCA sampler.

            Parameters
            ----------
            hamiltonian : the hamiltonian (energy) function for the sequence of interest. make sure that *likelier* sequences have *lower* energies!
                needs to be passed as a function, e.g. define an outside function or use lambda
            L: the length of the sequence of interest
            N_chains : the number of MCMC chains to run

            optional:
                T : the computational temperature (defaults to 1)
                    (remember: probabilities are e^{U/T})

                record_freq : "record" a sequence in the log every N steps.
                                Defaults to 1000. Want fewer for Gibbs sampling.

                independent : if TRUE, ignore couplings terms and use only fields h_i.
                                Defaults to FALSE.

                extra_potential : add an extra potential to the hamiltonian (defaults to NONE)
                                    Needs to be a function; I'd recommend using lambda notation, e.g.
                                    extra_potential = lambda s: 9 * MutationalSteps(s, to_numeric(model.seq()))

                pos_constraint : a constraint on which positions can be mutated. If not NONE, only positions
                                  in pos_constraint can be mutated.

                one_hot : whether to one-hot encode sequences. defaults to FALSE. You may however want this for
                                    certain extra_potential forms.

                alphabet: what the amino acid alphabet is. defines which numbers map to which aas.
                            defaults to alphabetical with gaps as 0.

                initialization: method of initialization. either 'reference', in which all chains start with the
                            reference sequence; 'random', in which all chains start with a random sequence; or
                            "defined," where you tell it what sequence to start with

                starting_seq: for defined initialization, the sequence to start with. If you pass any argument other
                            than the default of None for this, we will assume that you want defined initialization

        '''

        # set class attributes
        self.N_chains = N_chains
        self.T = T
        self.log = []; self.energies = []
        self.N_iterations = 0
        self.record_freq = record_freq
        self.one_hot = one_hot
        self.L = L
        self.alphabet = default_aa_alphabet
        self.shape = (self.L, len(self.alphabet))
        self.pos_constraint = pos_constraint
        self.bias = []

        # without extra definitions, we can only do metropolis sampling
        self.default_method = 'metropolis'
        self.gibbs_implemented = True

        # If you pass a starting sequence, assume that you want a defined initialization
        if type(starting_seq) != type(None):
            initialization = "defined"

        # define the hamiltonian
        self._ham = hamiltonian

        # initialize all chains and start them off with the reference sequence
        for chain in range(self.N_chains):

            if initialization == 'random':
                initial_seq = np.random.choice(np.arange(1,21), self.L)
                self.log.append([initial_seq])
                self.energies.append([self.hamiltonian(initial_seq)])

            elif initialization == "defined":

                if type(starting_seq) == type(None):
                    raise InputError("Must pass an argument for starting_seq for defined initialization!")

                else:
                    if isinstance(starting_seq[0], (str, np.str_)):
                        initial_seq = to_numeric(starting_seq, self.alphabet)

                    else:
                        initial_seq = starting_seq

                    self.log.append([initial_seq])
                    self.energies.append([self.hamiltonian(initial_seq)])


    def hamiltonian(self, sequence, ignore_bias=False):

        if len(self.bias) == 0 or ignore_bias:
            return self._ham(sequence)

        else:
            
            U = self._ham(sequence)

            for bias in self.bias:
                U += bias.energy(sequence)

            return U


    def metropolis(self, N, nch, progress=True, suppress_log=False):

        ''' Run Metropolis sampling of sequences from the DCA model, plus any added potentials.

        Parameters
        ----------
        N : number of iterations
        nch: which chain to run
        suppress_log: if TRUE, no sequences are recorded to the log. defaults to FALSE.'''

        # Initialize with the last sequence from the log and take that energy
        s = self.log[nch][-1]
        current_energy = self.hamiltonian(s)

        # tqdm here creates a progress bar, which I think is nice
        if progress:
            with tqdm(total=N, position=0, leave=False) as pbar:

                for n in tqdm(range(N), position=0, leave=False):

                    # Propose a new sequence by changing one amino acid
                    s_prop = mutate(s, one_hot = self.one_hot, pos_constraint = self.pos_constraint)

                    # Calculate the energy of that new sequence
                    prop_energy = self.hamiltonian(s_prop)

                    # Acceptance probability is e^{-del E}
                    #  current - prop = -(prop - current)
                    acceptance_prob = np.exp((current_energy - prop_energy)/self.T)

                    # Accept based on this probability
                    if np.random.uniform(0,1) < acceptance_prob:
                        s = s_prop
                        current_energy = prop_energy

                    # Every self.record_freq times, record the sequence
                    if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                        self.log[nch].append(s)
                        self.energies[nch].append(current_energy)
                        
        else:
            for n in range(N):

                # Propose a new sequence by ping one amino acid
                s_prop = mutate(s, one_hot = self.one_hot, pos_constraint = self.pos_constraint)

                # Calculate the energy of that new sequence
                prop_energy = self.hamiltonian(s_prop)

                # Acceptance probability is e^{del E}
                acceptance_prob = np.exp((current_energy - prop_energy)/self.T)

                # Accept based on this probability
                if np.random.uniform(0,1) < acceptance_prob:
                    s = s_prop
                    current_energy = prop_energy

                # Every self.record_freq times, record the sequence
                if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                    self.log[nch].append(s)
                    self.energies[nch].append(current_energy)
                    
    def gibbs(self, N, nch=0, suppress_log=False):
        
        '''Standard Gibbs sampling based just on the Hamiltonian. If you have a faster way to calculate it,
            redefine this function in a new class!
            
            Arguments:
                N: the number of iterations to run. Note that one iteration loops through every position.
                nch: the number of chain being run. Defaults to zero (the first chain) if not specified'''
        
        s = self.log[nch][-1]
        current_energy = self.hamiltonian(s)

        # looping through iterations with a progress bar
        with tqdm(total=N, position=0, leave=False) as pbar:

            for n in tqdm(range(N), position=0, leave=False):

                s = s.copy()

                # Define a random order of mutations to sample (considering possible mutation constraints)
                random_mutation_order = get_random_mutation_order(len(s), self.pos_constraint)
                
                # Loop through each position
                for k in random_mutation_order:

                    # Generate all possible single mutants at that position
                    muts = gen_all_muts(s, k, pos_constraint=self.pos_constraint[k])
                    
                    H = np.array([np.exp(-self.hamiltonian(m)/self.T) for m in muts])

                    if len(self.bias) > 0:
                        for bias in self.bias:
                            H += bias.dU(s, k)

                    # Calculate the conditional probabilities
                    conditional_probs = np.exp(H)
                    
                    # And normalize them
                    conditional_probs = conditional_probs/np.sum(conditional_probs)

                    if type(self.pos_constraint)==type(None):
                        
                        # Without a position constrant, just choose from 20 amino acids
                        s[k] = np.random.choice(np.arange(1,21), p=conditional_probs)
                        
                    else:
                        # Or choose from the allowed amino acids at that position
                        s[k] = np.random.choice(np.arange(1,21)[self.pos_constraint[k]], p=conditional_probs)

                if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                    self.log[nch].append(s)
                    #self.energies[nch].append(current_energy)

    def run(self, N, method=None, n_jobs = os.cpu_count(), parallel_method="threads", progress=True, suppress_log=False):

        ''' Run Markov Chain Monte Carlo on all chains.

        Parameters
        ----------
        N : number of iterations to run
        n_jobs: how many threads to run in parallel. Defaults to the number of CPUs, but overriden
                    if the number of chains is smaller than that (will become the number of chains).
                    You likely want to override this if running on a personal computer and want to do anything else.
        method : override the self.default_method attribute to tell the sampler how to run MCMC
                    (options right now are metropolis and, if implemented, gibbs)'''

        if type(method) == type(None):
            method = self.default_method

        # Can't parallelize more than the number of chains - so if you asked for more jobs than there are
        #  chains, default to the number of chains
        if self.N_chains < n_jobs:
            n_jobs = self.N_chains

        if method.lower() == 'metropolis':

            if parallel_method.lower() == 'none':
                for i in range(self.N_chains):
                    if progress:
                        print("Sampling chain", i)
                    self.metropolis(N, i, progress=progress)

            else:

                # Run separate chains as parallel threads
                Parallel(n_jobs=n_jobs, prefer=parallel_method)(delayed(self.metropolis)(N, i, suppress_log=suppress_log) for i in range(self.N_chains))

        elif method.lower() == 'gibbs':

            if self.gibbs_implemented:
                if parallel_method.lower() == 'none':
                    for i in range(self.N_chains):
                        print("Sampling chain", i)
                        self.gibbs(N, i)
                else:
                    Parallel(n_jobs=n_jobs, prefer=parallel_method)(delayed(self.gibbs)(N, i) for i in range(self.N_chains))
            else:
                return NotImplementedError("Must define a Gibbs sampling method first!")

        # Record how much you ran (unless you're suppressing the log)
        if not suppress_log:
            self.N_iterations += N

    def generate_alignments(self, stride=1, burnin=0):

        '''Take raw output (integer numpy arrays) and convert to a set of multiple sequence alignments
            Optional arguments:
                stride (int): you may not want to take every recorded sequence, so do so every n
                burnin (int): number of iterations to throw out as "burnin" before recording - default 0
        '''

        if isinstance(burnin, float):
            burnin = burnin * self.N_iterations

        self.msas = []
        self.frequency_list = []

        indexer = np.arange(int(burnin/self.record_freq), int(self.N_iterations/self.record_freq), stride)

        for log in self.log:

            # Convert raw numeric sequence log to a MultipleSequenceAlignment object

            log = np.array(log)
            if self.one_hot:
                log_numeric = np.argmax(log.reshape(log.shape[0], len(self.model.seq()), 21), axis=2)
            else:
                log_numeric = log

            ali = msa.MultipleSequenceAlignment(np.array(self.alphabet[log_numeric])[indexer],
                                            ids=np.array([str(i) for i in np.arange(len(indexer))]))

            # Calculate frequencies
            ali.calc_frequencies()

            # Assign burnin and stride as attributes of the MSA
            ali.burnin = burnin; ali.stride = stride

            self.msas.append(ali)
            self.frequency_list.append(ali.frequencies)

        self.frequencies = np.mean(self.frequency_list)

        r = []
        for msa_ in self.msas:
            for record in msa_._records:
                r.append(record)

        # Re-initiate an MSA from that
        self.alignment = msa.MultipleSequenceAlignment(r)

    def chain_divergences(self, relative_burnin=0.25, stride=1):

        self.generate_alignments(burnin=relative_burnin, stride=stride)

        kl_divs = np.zeros((self.N_chains, self.N_chains))

        for i,msa1 in enumerate(self.msas):
            for j,msa2 in enumerate(self.msas):

                w = msa2.frequencies > 0
                kl_divs[(i,j)] = np.sum(np.sum(msa1.frequencies[w] * np.log(msa1.frequencies[w] / msa2.frequencies[w])))

        return kl_divs

    def save(self, filename):

        self.generate_alignments(burnin=0)

        TempSamplerDict = {}
        TempSamplerDict["msa"] = self.alignment
        TempSamplerDict["msas"] = self.msas
        TempSamplerDict["U"] = self.energies
        TempSamplerDict["index_list"] = self.model.index_list
        try:
            TempSamplerDict["encoder"] = self.encoder
        except:
            None

        try:
            TempSamplerDict["refseq"] = self.model.seq()
        except:
            None

        TempSamplerDict["T"] = self.T
        TempSamplerDict["N_iterations"] = self.N_iterations
        TempSamplerDict["N_chains"] = self.N_chains

        pickle.dump(TempSamplerDict, open(filename, "wb"))


class DCASampler(BaseSampler):
    
    def __init__(self, model, N_chains, T=1, record_freq=1000, independent=False,
                 extra_potential=None, pos_constraint=None, one_hot=False, alphabet=default_aa_alphabet,
                 initialization='reference', starting_seq=None):
        
        self.model = model
        self.L = model.L
        
        if type(extra_potential)==type(None):
            self.hamiltonian = lambda x: DCAEnergy(x, model, one_hot=one_hot, independent=independent)

        else:
            self.hamiltonian = lambda x: DCAEnergy(x, model, one_hot=one_hot, independent=independent) + \
                                            extra_potential(x)
            
        if initialization=="reference":
            starting_seq = model.seq()
        
        super().__init__(self.hamiltonian, model.L, N_chains, T=T, record_freq=record_freq, independent=independent,
                 pos_constraint=pos_constraint, one_hot=one_hot, alphabet=alphabet,
                 initialization=initialization, starting_seq=starting_seq)


class GibbsDCASampler(DCASampler):

    '''A Gibbs sampler for the "Restrained latent voyager potential." Subclasses from the OneHotDCASampler
            but has a different run_mcmc.'''

    def __init__(self, model, N_chains, T=1, record_freq=10, independent=False, pos_constraint=None):

        super().__init__(model, N_chains, T=T, record_freq=record_freq, independent=independent,
                         pos_constraint=pos_constraint, one_hot=False)

        self.default_method = 'gibbs'
        self.gibbs_implemented=True

        self._stored_state = []

    def gibbs(self, N, nch, progress=True, suppress_log=False):

        s = self.log[nch][-1]
        current_energy = self.hamiltonian(s)

        with tqdm(total=N, position=0, leave=False) as pbar:

            for n in tqdm(range(N), position=0, leave=False):

                s = s.copy()

                if type(self.pos_constraint)==type(None):
                    random_mutation_order = np.random.choice(np.arange(self.L), self.L, replace=False)
                else:
                    random_mutation_order = np.random.choice(self.pos_constraint, self.L, replace=False)

                for k in random_mutation_order:

                    # change in dca component
                    conditional_energies = -calc_dca_conditionals(s-1, self.model.h_i, self.model.J_ij, k)

                    if len(self.bias) > 0:

                        for bias_fn in self.bias:

                            conditional_energies += bias_fn.dU(s, k)

                    # All that matters is the difference between energies, magnitudes will be normalized out
                    #  subtract off the lowest to make numbers reasonable and prevent overflow
                    conditional_energies -= np.min(conditional_energies)

                    conditional_probs = np.exp(-conditional_energies / self.T)
                    conditional_probs = conditional_probs / np.sum(conditional_probs)

                    s[k] = np.random.choice(np.arange(20), p=conditional_probs) + 1

                if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                    self.log[nch].append(s)
                    self.energies[nch].append(current_energy)

def LatentVoyager(model, n_samples, n_chains, w1, w2, filename, method='gibbs', save=True, record_freq = 2, parallel_method='None',
                   pca_precalculated=(), ali=None, sparse_pca=False, alpha=1, refseq=None, burnin=10):

    if len(pca_precalculated) == 0:
        if type(ali) != type(None):
            pca_res = SequencePCA(ali, sparse=sparse_pca, alpha=alpha)
        else:
            raise TypeError("Must pass either precalculated PCA results or a multiple sequence alignment!")

    else:
        pca_res = pca_precalculated

    if type(refseq) == type(None):
        refseq = model.seq()

    if method=='gibbs':
        sampler = GibbsDCASampler(model, n_chains, record_freq=record_freq)
    elif method=='metropolis':
        sampler = DCASampler(model, n_chains, record_freq=record_freq)

    sampler.bias.append(LinearDistanceRestraint(refseq, w1))
    sampler.bias.append( LatentVoyagerPotential(refseq, w2, pca_precalculated=pca_res))
    sampler.run(n_samples, parallel_method='None')

    sampler.generate_alignments(burnin=burnin)

    sampler.save(filename)

    return sampler
