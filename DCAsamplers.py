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


    '''Decode a one-hot encoding of a sequence into a numeric representation of shape L x N_alphabet.

    Parameters
    ----------
    one_hot_sequence : a one-hot encoded protein sequence of length 21*L

    optional:
        alphabet : the alphabet (in particular, the order of letters used in the one-hot decoding)
                    defaults to alphabetical order, with a gap character as 0
        remove_gaps : if the one-hot encoding included gaps, return a numeric representation reindex to
                    not include the gap character
        gap_char : the identity of the gap character; defaults to '-'

    Returns
    -------
    s : the sequence as a numeric representation of length L
    """

    '''

    alph_size = len(alphabet)
    L = len(one_hot_sequence) / alph_size

    if remove_gaps == True and gap_char in alphabet:

        ungapped_pos = np.where(alphabet!=gap_char)[0]
        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)[:, ungapped_pos]
        s = np.argmax(ohseq_reshaped, axis=1)

    else:

        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)
        s = np.argmax(ohseq_reshaped, axis=1)

    return s

def DCAEnergy(s, model, independent=False, one_hot=False):

    '''DCA energy function that can take in numeric or one-hot encoded sequences.

    Parameters
    ----------
    s: a protein sequence, either a numpy array of length L or a
            one-hot encoded array of length 20*L
    model: a CouplingsModel object (the DCA model to score with)

    optional:
        independent : calculate without couplings terms (using only h_i); defaults to FALSE
        one_hot : whether the input sequence is one-hot encoded (defaults to FALSE)

    Returns
    -------
    E : the DCA energy of the sequence
    '''

    if one_hot:
        seq = one_hot_decode(s, len(model.index_list))
        return potts_model_hamiltonian(seq, model.h_i, model.J_ij, independent=independent)

    else:
        return potts_model_hamiltonian(s, model.h_i, model.J_ij, independent=independent)


@jit(nopython=True)
def potts_model_hamiltonian(seq, hi, Jij, independent=False):

    ''' This jit-compiled function does the actual math because it's faster '''

    E = 0

    for i,x in enumerate(seq):

        E += hi[(i,x)]

        if not independent:
            for j,y in enumerate(seq[:i]):

                E += Jij[(i, j, x, y)]

    return E

def to_numeric(seq, aa_alphabet=default_aa_alphabet):

    ''' Converts a sequence from strings to numbers '''

    return np.array([list(alphabet).index(k) for k in seq])


def RLVP(seq, pca_model, ref_seq, ref_embedding, seq_weight, pca_weight):

    """The 'restrained latent voyager' wants to get as far from "home" in latent space
        without making too many steps in sequence space.

        Begins with a one-hot encoding of the sequence."""

    # The number of mutational steps away from the reference sequence
    N = int(len(seq)/21)
    seq_num = np.argmax(seq.reshape(N,21), axis=1)  # Convert to numeric from 1-hot for ease
    seq_distance = np.sum(seq_num != ref_seq, axis=0)

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

class DCASampler:

    '''A class to sample from a DCA model using MCMC, allowing for the additional of extra potentials.
            Relatively flexible for a variety of applications.'''

    def __init__(self, model, N_chains, T=1, record_freq=1000, independent=False,
                 extra_potential=None, pos_constraint=None, one_hot=False, alphabet=default_aa_alphabet,
                 initialization='reference'):

        ''' Initialize the DCA sampler.

            Parameters
            ----------
            model :    a CouplingsModel object (the DCA model to score with)
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
                            reference sequence, or 'random', in which all chains start with a random sequence.

        '''

        # set class attributes
        self.N_chains = N_chains
        self.T = T
        self.log = []; self.energies = []
        self.N_iterations = 0
        self.record_freq = record_freq
        self.one_hot = one_hot
        self.model = model
        self.L = len(self.model.seq())
        self.alphabet = default_aa_alphabet
        self.shape = (self.L, len(self.alphabet))
        self.pos_constraint = pos_constraint

        # without extra definitions, we can only do metropolis sampling
        self.default_method = 'metropolis'
        self.gibbs_implemented = False

        # define the reference sequence; more complicated if one-hot
        if one_hot:
            # fit the one hot encoder based on all single mutants for consistency
            self.single_mutants = all_single_mutants(self.model.seq(), self.model)
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.fit(self.single_mutants.matrix)

            # then encode the actual reference sequence
            self.refseq = self.encoder.transform(self.model.seq().reshape(1, -1))[0]

        else:
            # otherwise just convert the sequence to numbers
            self.refseq = to_numeric(self.model.seq(), self.alphabet)

        # define the hamiltonian
        if type(extra_potential)==type(None):
            self.hamiltonian = lambda x: DCAEnergy(x, model, one_hot=one_hot, independent=independent)

        else:
            self.hamiltonian = lambda x: DCAEnergy(x, model, one_hot=one_hot, independent=independent) + \
                                            extra_potential(x)

        # initialize all chains and start them off with the reference sequence
        for chain in range(self.N_chains):

            if initialization == 'reference':
                self.log.append([self.refseq])
                self.energies.append([self.hamiltonian(self.refseq)])

            elif initialization == 'random':
                initial_seq = np.random.choice(np.arange(1,21), self.L)
                self.log.append([initial_seq])
                self.energies.append([self.hamiltonian(initial_seq)])


    def metropolis(self, N, nch, suppress_log=False):

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
        with tqdm(total=N, position=0, leave=False) as pbar:

            for n in tqdm(range(N), position=0, leave=False):

                # Propose a new sequence by ping one amino acid
                s_prop = mutate(s, one_hot = self.one_hot, pos_constraint = self.pos_constraint)

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

            # Run separate chains as parallel threads
            Parallel(n_jobs=n_jobs, prefer=parallel_method)(delayed(self.metropolis)(N, i, progress=progress, suppress_log=suppress_log) for i in range(self.N_chains))

        elif method.lower() == 'gibbs':

            if self.gibbs_implemented:
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

class GibbsRLVPSampler(DCASampler):

    '''A Gibbs sampler for the "Restrained latent voyager potential." Subclasses from the OneHotDCASampler
            but has a different run_mcmc.'''

    def __init__(self, model, N_chains, alignment, w1, w2, T=1, nref=0, n_components=2, record_freq=10, independent=False,
                 pos_constraint=None, pca_precalculated=None):

        if type(pca_precalculated)==type(None):
            self.pca_model, aln_embedding, self.encoder = SequencePCA(alignment, n_components)
        else:
            self.pca_model, aln_embedding, self.encoder = pca_precalculated

        self.input_alignment = alignment
        self.w1 = w1
        self.w2 = w2
        self.n_components = n_components

        #refseq = alignment.as_numeric()[nref]
        self._ref_embedding = aln_embedding[nref]

        super().__init__(model, N_chains, T=T, record_freq=record_freq, independent=independent,
                         #extra_potential=lambda seq: RLVP(seq, pca_model, refseq, self._ref_embedding, w1, w2)[0],
                         pos_constraint=pos_constraint, one_hot=False)

        self.default_method = 'gibbs'
        self.gibbs_implemented=True

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

                    # change in distance to reference sequence
                    conditional_energies = conditional_energies + 1 - self.w1 * np.identity(21)[self.refseq[k]][1:]

                    # change in pca potential
                    # first get the component terms for the position of interest, ignoring the gap term
                    components = self.pca_model.components_.reshape((self.n_components, self.L, 21))[:,k,1:].T

                    # calculate how far each one is from the identity of the reference
                    component_diffs = np.sqrt(np.sum((components - components[self.refseq[k]-1])**2, axis=1))

                    # add this term weighted by w2 into the overall weight
                    conditional_energies = conditional_energies - self.w2 * component_diffs

                    # All that matters is the difference between energies, magnitudes will be normalized out
                    #  subtract off the lowest to make numbers reasonable and prevent overflow
                    conditional_energies -= np.min(conditional_energies)

                    conditional_probs = np.exp(-conditional_energies / self.T)
                    conditional_probs = conditional_probs / np.sum(conditional_probs)

                    s[k] = np.random.choice(np.arange(20), p=conditional_probs) + 1

                if (self.N_iterations + n) % self.record_freq == 0 and not suppress_log:
                    self.log[nch].append(s)
                    self.energies[nch].append(current_energy)

@jit(nopython=True)
def calc_dca_conditionals(seq, hi, jij, k):

    '''The DCA energy term associated with the 20 conditional probabilities of observing any amino acid
        at site k given the rest of the sequence. For Gibbs sampling.'''

    U = []

    for aa in range(20):

        u = hi[k, aa]

        for i in range(len(hi)):
            u += jij[k, aa, i, seq[i]]

        U.append(u)

    return np.array(U)
