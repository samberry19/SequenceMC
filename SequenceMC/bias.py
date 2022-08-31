import numpy as np
from .utils import SequencePCA, partial_pca, to_numeric

class LinearDistanceRestraint:

    """
    A linear distance restraint for sampling sequences. Adds a constant potential term
    """

    def __init__(self, refseq, w):

        self.w = w
        self.L = len(refseq)
        self._requires_stored_state = False

        if isinstance(refseq[0], (str, np.str_)):
            self.refseq = refseq
            self.refseq_num = to_numeric(refseq)
        else:
            self.refseq_num = refseq
            self.refseq = default_aa_alphabet[refseq]

    def dist(self, seq):

        if isinstance(seq[0], (str, np.str_)):
            return np.sum(seq != self.refseq)

        else:
            return np.sum(seq != self.refseq_num)

    def energy(self, seq):

        return self.w * self.dist(seq)

    def dU(self, seq, k, gap=False):

        dU = 1 - np.identity(21)[self.refseq_num[k]]

        if gap:
            return self.w * dU
        else:
            return self.w * dU[1:]

class LatentVoyagerPotential:

    def __init__(self, refseq, w, ali=None, pca_precalculated=[], n_components=2):

        self.w = w
        self.L = len(refseq)

        if isinstance(refseq[0], (str, np.str_)):
            self.refseq = refseq
            self.refseq_num = to_numeric(refseq)
        else:
            self.refseq_num = refseq
            self.refseq = default_aa_alphabet[refseq]

        if len(pca_precalculated) < 3:
            if type(ali) != type(None):
                self.pca_model, self.pca_embedding, self.encoder, self._enc_ali = \
                        SequencePCA(ali, n=n_components)
            else:
                raise TypeError("Must pass either an alignment or a pretrained PCA model")

        else:
            self.pca_model, self.pca_embedding, self.encoder, self._enc_ali = pca_precalculated

        self._means = np.mean(self._enc_ali, axis=0)
        self._ref_one_hot = self.encoder.transform([self.refseq])[0]
        self._ref_embedding = self.pca_model.transform([self._ref_one_hot])[0]

        # Calculate the (constant) change in PCA difference for every mutational step
        dz = [[] for i in range(self.L)]

        for k in range(self.L):

            for seq in position_mutants(self.refseq, k).as_numeric():
                dz[k].append(partial_pca(self.pca_model, self._means, one_hot_encode(seq), [k]))

        self._dz_matrix = np.array(dz)


    def transform_sequence(self, seq):

        if len(seq) == len(self._ref_one_hot):
            return self.pca_model.transform([seq])[0]

        elif len(seq) == self.L:

            if isinstance(seq[0], (str, np.str_)):
                seq = to_numeric(seq)

            seq_enc = one_hot_encode(seq)
            return self.pca_model.transform([seq_enc])[0]

    def transform_sequences(self, sequences):

        return np.array([self.transform_sequence(seq) for seq in sequences])

    def energy(self, seq):

        diff = self.transform_sequence(seq) - self._ref_embedding
        return -self.w * np.sqrt(np.sum(diff**2))

    def dU(self, seq, k, gap=False):

        """
        Returns an array of the energies for this bias potential for mutating
        the site to all 20 amino acids at a position k. Called during Gibbs sampling.
        """

        # One hot encode the sequence for PCA transformation
        one_hot_seq = one_hot_encode(seq)

        # We will ignore all columns where the sequence is identical to the reference
        nonoverlap = np.where(seq != self.refseq_num)[0]

        # The components of the latent space embedding corresponding to the different positions for the input sequence
        #  (for comparison with the reference, this *can* change things alas)
        z_offset = partial_pca(self.pca_model, self._means, one_hot_seq, nonoverlap)

        # The components of the latent space embedding for the different positions, now for all 20 mutations
        #  the change for the twenty mutants is independent of input sequence and was precalculated as self._dz
        partial_difference_z = z_offset + self._dz_matrix[k]

        # That same partial for the reference sequence
        #  terms corresponding to the positions different between the input sequence and the reference AND
        #  position k that we are mutating
        partial_reference_z = partial_pca(self.pca_model, self._means, self._ref_one_hot, np.concatenate([nonoverlap, [k]]))

        # this is the change
        delta_z = np.sqrt(np.sum((partial_difference_z - partial_reference_z)**2, axis=1))

        return -self.w * delta_z
