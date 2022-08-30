import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA

from bioviper import msa

default_aa_alphabet = np.array(['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

def one_hot_encode(seq_num, dim=1):

    '''
    One-hot encode a numerically-encoded amino acid sequence. Faster than training a one-hot encoder
    and using .transform()
    '''

    L = len(seq_num)

    # Initialize an array of all zeros of the correct shape
    x = np.zeros((L,21))

    # Assign the appropriate positions to be ones
    x[np.arange(L), seq_num] = 1

    # Make one-dimensional if dim=1
    if dim==1:
        return x.reshape((L*21))
    elif dim==2:
        return x

def partial_pca(pca_model, ali_enc_mean, seq_enc, selection):

    '''
    Embeds a point in PCA space using only a subset of features. Used for comparing differences
    when the majority of features are unchanged for computational efficiency.
    '''

    if len(selection) > 0:
        component_slice = pca_model.components_.reshape((2, model.L, 21))[:,selection]
        sequence_slice = seq_enc.reshape((model.L, 21))[selection]
        mean_slice = ali_enc_mean.reshape((model.L, 21))[selection]

        return np.sum(component_slice * (sequence_slice - mean_slice), axis=(1,2))

    else:
        return 0

def position_mutants(seq, k):

    seqs = []

    for i in range(1,21):
        z = seq.copy()
        z[k] = default_aa_alphabet[i]
        seqs.append(z)

    return msa.MultipleSequenceAlignment(np.array(seqs), ids=default_aa_alphabet[1:])

def get_pos_index(model, index):
    return np.where(model.index_list==index)[0][0]

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
    L = int(len(one_hot_sequence) / alph_size)

    if remove_gaps == True and gap_char in alphabet:

        ungapped_pos = np.where(alphabet!=gap_char)[0]
        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)[:, ungapped_pos]
        s = np.argmax(ohseq_reshaped, axis=1)

    else:

        ohseq_reshaped = one_hot_sequence.reshape(L, alph_size)
        s = np.argmax(ohseq_reshaped, axis=1)

    return s


def to_numeric(seq, aa_alphabet=default_aa_alphabet):

    ''' Converts a sequence from strings to numbers '''

    return np.array([list(default_aa_alphabet).index(k) for k in seq])

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

    return msa.MultipleSequenceAlignment(np.array(all_singles), ids=names)

def mutate(s, alphabet_size=21, nmax=20, one_hot=False, pos_constraint=None):

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
        s_new[pos] = np.random.choice(np.arange(1, 21))

    return s_new.astype('int')

def SequencePCA(alignment, n=2, sparse=False, alpha=1, ridge_alpha=0.01):

    '''
    Run principal component analysis (PCA) on a one-hot encoded multiple sequence alignment.

    Arguments:
        MANDATORY
            alignment: the MSA as a bioviper MultipleSequenceAlignment object (loaded via msa.readAlignment(input.fa))
                (can also be coerced from a Biopython MultipleSeqAlignment, EVcouplings Alignment, or a 2D numpy array)
        OPTIONAL
            n: the number of components to use for the PCA model (defaults to 2)
            sparse: whether to perform sparse PCA
            alpha: the main parameter of sparse PCA, which controls the sparsity (see sklearn documentation)
            ridge_alpha: see sklearn documentation

    Returns:
        sequence_pca:         the pca model itself as an sklearn.decomposition.PCA object
        alignment_latent_pca: the embedded sequences
        encoded_natural_sequences: the one-hot encoded sequences used directly to perform the PCA
            (needed for if you want to access, say, the means used to normalize the data)

    '''

    if not isinstance(alignment, msa.MultipleSequenceAlignment):
        try:
            alignment = MultipleSequenceAlignment(alignment)
        except:
            raise TypeError("Must pass a valid multiple sequence alignment! (See docs for details)")

    encoded_natural_sequences = np.array([one_hot_encode(seq) for seq in alignment.as_numeric()])

    if sparse:
        sequence_pca = SparsePCA(n_components=n, alpha=alpha, ridge_alpha=ridge_alpha)
    else:
        sequence_pca = PCA(n_components=n)

    alignment_latent_pca = sequence_pca.fit_transform(encoded_natural_sequences)

    return sequence_pca, alignment_latent_pca, encoded_natural_sequences
