from numba import jit
from .utils import one_hot_decode

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
        return potts_model_hamiltonian(s-1, model.h_i, model.J_ij, independent=independent)


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


@jit(nopython=True)
def calc_dca_conditionals(seq, hi, jij, k):

    '''The DCA energy term associated with the 20 conditional probabilities of observing any amino acid
        at site k given the rest of the sequence. For Gibbs sampling.'''

    U = []

    for aa in range(20):

        u = hi[k, aa]

        for i in range(len(hi)):
            u += jij[k, i, aa, seq[i]]

        U.append(u)

    return np.array(U)
