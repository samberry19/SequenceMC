'''
samplers.py

This script contains a set of functions and class for sampling sequences from a DCA model with optional
added potentials to bias the sequences. The core functionality is embedded in the OneHotDCASampler class.
So far it requires a model trained with EVcouplings, but my plan is to make it generalized to any DCA model
(for example I want to try it on a bmDCA inference to see how different the result is.)

Functions are also provided for three kinds of sampling:

1. "Vanilla" DCA sampling, which will draw a set of sample sequences from the DCA model at a given temperature "T"

2. "Restrained" DCA sampling, which will draw a set of sequences close to a given reference sequence, with the
        distribution of proximities determined by a weight parameter w (and the temperature)

3. "Restrained latent voyager" DCA sampling (apology for the whimsical name), which in addition to biasing sequences to
        be close in *sequence space* to a sequence of interest, biases them to be far in some sort of phylogenetically-
        informed *latent space*, such as to sample a set of sequences that are only a small set of mutations away from
        a reference but potentially change key features of the sequences, rather than (say) simply mutating the least
        conserved residues. This is inspired by the biological problem of designing new sequences that fold correctly
        but alter the biochemical functionality, for example change the substrate selectivity to be more like that of
        distant homologs.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from tqdm import tqdm
from MultipleSequenceAlignment import *
from evcouplings.couplings import CouplingsModel

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
import os

from numba import jit

default_aa_alphabet = np.array(['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

def one_hot_decode(one_hot_sequence, L, alphabet = default_aa_alphabet, remove_gaps=True, gap_char='-'):


    '''Decode a one-hot encoding of a sequence into a numeric representation of shape L x N_alphabet. '''

    alph_size = len(alphabet)

    if remove_gaps == True and gap_char in alphabet:

        ungapped_pos = np.where(alphabet!=gap_char)[0]
        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)[:, ungapped_pos]
        s = np.argmax(ohseq_reshaped, axis=1)

    else:

        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)
        s = np.argmax(ohseq_reshaped, axis=1)

    return s

def DCA_onehot(s, model, independent=False):

    ''' A DCA energy function using a one-hot encoded input sequence '''

    # Calculate the sitewise h_i sum (the per-residue "fields")
    seq = one_hot_decode(s, len(model.index_list))

    return dca_hamiltonian(seq, model.h_i, model.J_ij, independent=independent)


@jit(nopython=True)
def dca_hamiltonian(seq, hi, Jij, independent=False):

    E = 0

    for i,x in enumerate(seq):

        E += hi[(i,x)]

        if not independent:
            for j,y in enumerate(seq[:i]):

                E += Jij[(i, j, x, y)]

    return E

def RLVP(seq, pca_model, ref_seq, ref_embedding, seq_weight, pca_weight):

    """The 'restrained latent voyager' wants to get as far from "home" in latent space
        without making too many steps in sequence space.

        Begins with a one-hot encoding of the sequence."""

    # The number of mutational steps away from the reference sequence
    N = int(len(seq)/21)
    seq_num = np.argmax(seq.reshape(N,21), axis=1)  # Convert to numeric from 1-hot for ease
    seq_distance = np.sum(seq_num != ref_seq, axis=0)

    print(seq_num)

    # Transform into the PCA latent space
    pca_embedding = pca_model.transform(seq.reshape(1,-1))[0]

    # Distance in that space
    voyager_potential = np.sqrt(np.sum((ref_embedding - pca_embedding)**2))

    # Final potential is a weighted difference of these
    return (voyager_potential * pca_weight - seq_distance * seq_weight, \
                    voyager_potential, seq_distance)


def all_single_mutants(s, m=None):

    ''' Generate all single mutants of a sequence '''

    all_singles = [s]; names = ["WT"]
    for i in range(len(s)):
        for j in default_aa_alphabet:
            if s[i] != j:
                sc = s.copy()
                sc[i] = j
                all_singles.append(sc)
                if type(m)==type(None):
                    names.append(s[i]+str(i)+j)
                else:
                    names.append(s[i]+str(m.index_list[i])+j)

    return MultipleSequenceAlignment(np.array(all_singles), ids=names)

class OneHotDCASampler:

    '''A class to sample from a DCA model using MCMC, allowing for the additional of extra potentials.
        Uses a one-hot encoding of the sequences to make it easy to add certain kinds of extra potentials
        (e.g. one that accesses a latent space).'''

    def __init__(self, model, N_chains, T=1, record_freq=1000, independent=False,
                 extra_potential=None, pos_constraint=None):

        self.N_chains = N_chains
        self.T = T
        self.log = []; self.energies = []
        self.N_iterations = 0
        self.record_freq = record_freq

        self.model = model
        self.alphabet = default_aa_alphabet

        self.pos_constraint = pos_constraint

        self.single_mutants = all_single_mutants(self.model.seq(), self.model)
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(self.single_mutants.matrix)
        self.refseq = self.encoder.transform(self.model.seq().reshape(1, -1))[0]

        if independent:
            if type(extra_potential)==type(None):
                self.hamiltonian = lambda x: site_independent_model_hamiltonian(x, model)

            else:
                self.hamiltonian = lambda x: site_independent_model_hamiltonian(x, model) + extra_potential(x)

        else:
            if type(extra_potential)==type(None):
                self.hamiltonian = lambda x: DCA_onehot(x, model)

            else:
                self.hamiltonian = lambda x: DCA_onehot(x, model) + extra_potential(x)

        for chain in range(self.N_chains):
            self.log.append([self.refseq])
            self.energies.append([self.hamiltonian(self.refseq)])

    def run_mcmc(self, N, nch, progress=True, suppress_log=False):

        ''' Run N iterations of MCMC for chain nch. Called by .run() in parallel. '''

        # Initialize with the last sequence from the log and take that energy
        s = self.log[nch][-1]
        current_energy = self.hamiltonian(s)

        if progress:

            # tqdm here creates a progress bar, which I think is nice
            with tqdm(total=N, position=0, leave=False) as pbar:

                for n in tqdm(range(N), position=0, leave=False):

                    # Propose a new sequence by ping one amino acid
                    s_prop = mutate(s, pos_constraint = self.pos_constraint)

                    # Calculate the energy of that new sequence
                    prop_energy = self.hamiltonian(s_prop)

                    # Acceptance probability is e^{del E}
                    acceptance_prob = np.exp((prop_energy - current_energy)/self.T)

                    # Accept based on this probability
                    if np.random.uniform(0,1) < acceptance_prob:
                        s = s_prop
                        current_energy = prop_energy

                    # Every self.record_freq times, record the sequence

                    if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                        self.log[nch].append(s)
                        self.energies[nch].append(current_energy)

        else:
            for n in tqdm(range(N), position=0, leave=True):
                s_prop = mutate(s, pos_constraint = self.pos_constraint)
                prop_energy = self.hamiltonian(s_prop)
                acceptance_prob = np.exp(prop_energy - current_energy)

                if np.random.uniform(0,1) < acceptance_prob:
                    s = s_prop
                    current_energy = prop_energy

                if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                    self.log[nch].append(s)
                    self.energies[nch].append(current_energy)

    def run(self, N, n_jobs = os.cpu_count(), parallel_method="threads", progress=True, suppress_log=False):

        ''' Run all chains for N iterations in parallel. '''

        # Can't parallelize more than the number of chains - so if you asked for more jobs than there are
        #  chains, default to the number of chains
        if self.N_chains < n_jobs:
            n_jobs = self.N_chains

        # Run separate chains as parallel threads
        Parallel(n_jobs=n_jobs, prefer=parallel_method)(delayed(self.run_mcmc)(N, i, progress=progress, suppress_log=suppress_log) for i in range(self.N_chains))

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
            log_numeric = np.argmax(log.reshape(log.shape[0], len(self.model.seq()), 21), axis=2)

            msa = MultipleSequenceAlignment(np.array(self.alphabet[log_numeric])[indexer],
                                            ids=np.array([str(i) for i in np.arange(len(indexer))]))

            # Calculate frequencies
            msa.calc_frequencies()

            # Assign burnin and stride as attributes of the MSA
            msa.burnin = burnin; msa.stride = stride

            self.msas.append(msa)
            self.frequency_list.append(msa.frequencies)

        self.frequencies = np.mean(self.frequency_list)

        r = []
        for msa in self.msas:
            for record in msa._records:
                r.append(record)

        # Re-initiate an MSA from that
        self.alignment = MultipleSequenceAlignment(r)

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
        TempSamplerDict["encoder"] = self.encoder
        TempSamplerDict["refseq"] = self.model.seq()
        TempSamplerDict["T"] = self.T
        TempSamplerDict["N_iterations"] = self.N_iterations
        TempSamplerDict["N_chains"] = self.N_chains

        pickle.dump(TempSamplerDict, open(filename, "wb"))


def mutate(s, alphabet_size=21, nmax=20, one_hot=True, pos_constraint=None):

    """Randomly mutate the sequence of interest.
        Sequence must be numerically coded, either 1-20 (one_hot=False)
        or one-hot encoded in binary (one_hot=True)."""

    N = len(s)

    # Actual number of positions is 1/20th len(s) if we're one-hot encoded
    if one_hot:
        N = N/alphabet_size

    # Pick a position to mutate, with or without a constraint on which positions are available
    if type(pos_constraint) == type(None):
        pos = np.random.randint(N)
    else:
        pos = np.random.choice(pos_constraint)

    # Copy so we don't modify the original
    s_new = s.copy()

    # If one-hot encoded, we need to do something a bit weird
    if one_hot:

        # Initialize a new array of zeros and set a random position to 1
        new_aa = np.zeros(alphabet_size)
        new_aa[np.random.randint(1, 21)] = 1

        # Assign that to the correct part of the one-hot encoded sequence
        s_new[pos*alphabet_size:(pos+1)*alphabet_size] = new_aa

    # Otherwise swapping out the position is very simple
    else:
        snew[pos] = np.random.choice(np.arange(0, 20))

    return s_new.astype('int')


def RLVP(seq, pca_model, ref_seq, ref_embedding, seq_weight, pca_weight):

    """The 'restrained latent voyager' wants to get as far from "home" in latent space
        without making too many steps in sequence space.

        Begins with a one-hot encoding of the sequence."""

    # The number of mutational steps away from the reference sequence
    N = int(len(seq)/21)
    seq_num = np.argmax(seq.reshape(N,21), axis=1)  # Convert to numeric from 1-hot for ease
    seq_distance = np.sum(seq_num != ref_seq, axis=0)

    print(seq_num)

    # Transform into the PCA latent space
    pca_embedding = pca_model.transform(seq.reshape(1,-1))[0]

    # Distance in that space
    voyager_potential = np.sqrt(np.sum((ref_embedding - pca_embedding)**2))

    # Final potential is a weighted difference of these
    return (voyager_potential * pca_weight - seq_distance * seq_weight, \
                    voyager_potential, seq_distance)

def SequencePCA(alignment, n=2):

    # Generate all single mutants *with gaps* to ensure that our encoder maps the full space
    #  of possibilities
    single_mutants = all_single_mutants(alignment.matrix[0])

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(single_mutants)

    encoded_natural_sequences = encoder.transform(alignment.replace("X","-").matrix)

    sequence_pca = PCA(n_components=n)
    alignment_latent_pca = sequence_pca.fit_transform(encoded_natural_sequences)

    return sequence_pca, alignment_latent_pca, encoder

def LatentVoyager(model, w1, w2, N_iterations, alignment, n_components=2,
                  refseq=None, nref=0, n_chains=3, T=1, record_freq=1000,
                 burnin=0.25):

    pca_model, aln_embedding, _ = SequencePCA(alignment, n_components)
    refseq = alignment.as_numeric()[nref]
    ref_embedding = aln_embedding[nref]

    rlvp_sampler = OneHotDCASampler(model, n_chains, T = T, record_freq=record_freq,
        extra_potential=lambda seq: RLVP(seq, pca_model, refseq, ref_embedding, w1, w2)[0])

    rlvp_sampler.run(N_iterations)

    rlvp_sampler.generate_alignments(burnin=burnin)

    U = []

    for n,log in enumerate(rlvp_sampler.log):
        for seq in log:
            rlvp_potentials = RLVP(seq, pca_model, refseq, ref_embedding, w1, w2)
            dca_potential = DCA_onehot(seq, model)
            U.append([n, dca_potential, rlvp_potentials[1], rlvp_potentials[2], dca_potential + rlvp_potentials[0]])

    U_df = pd.DataFrame.from_records(U, columns=["nchain", "U_DCA", "U_voyager", "U_identity", "U_total"])

    return rlvp_sampler, U_df
