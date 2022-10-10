import numpy as np
import argparse
from evcouplings.couplings import CouplingsModel
from bioviper import msa
import SequenceMC as smc


# Read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", help='Path to simulation file (.dcd)')
parser.add_argument("N", help='Path to topology file (.pdb or .psf)')
parser.add_argument("-a", "--alignment", dest="alignment", help='Path to alignment in fasta format')
parser.add_argument("-o", "--output", dest="output_file", help='Output filename', default="sampler.pkl")
parser.add_argument("--dist_restraint", dest="dist_restraint", help='Weight for distance restraint', default=0)
parser.add_argument("--pca_bias", dest="pca_bias", help='Weight for PCA (LatentVoyager) bias', default=0)
parser.add_argument("--n_chains", dest="n_chains", help='Number of chains to sample', default=1)
parser.add_argument("--method", dest="method", help='MCMC method (gibbs or metropolis)', default="gibbs")
parser.add_argument("--burnin", dest="burnin", help='Number of chains to sample', default=10)
parser.add_argument("--independent", dest="independent", help='Flag if model is using a site-independent model for DCA (ignore J_ij terms)', default=False)
parser.add_argument("--sparse", dest="sparse_pca", help='Whether to infer a sparse PCA model', default=False)
parser.add_argument("--alpha", dest="alpha", help='Alpha for sparse PCA', default=1)
parser.add_argument("--record_freq", dest="record_freq", help='How often to record samples', default=5)
parser.add_argument("--parallel_method", dest="parallel_method", help='Whether to run chains in parallel', default='None')

args = parser.parse_args()

# Load in the alignment
ali = msa.readAlignment(args.alignment)

# Load in the DCA model of interest
model = CouplingsModel(all_plmc_dir + "couplings/nramps.model")

# If we want a site-independent model, have to estimate the parameters of one from the original model
if args.independent:
    model = model.to_independent_model()

# Use only the columns of the alignment in the model
ali = ali[:,model.index_list-1]

# Run principal component analysis to infer the latent representation
pca_res = SequencePCA(ali, sparse=args.sparse_pca, alpha=args.alpha)

refseq = model.seq()

if args.method=='gibbs':
    sampler = GibbsDCASampler(model, args.n_chains, record_freq=args.record_freq)
elif args.method=='metropolis':
    sampler = DCASampler(model, args.n_chains, record_freq=args.record_freq)

sampler = smc.LatentVoyager(model, args.N, args.dist_restraint, args.pca_bias, 40, "model_1.pkl", ali=ali, pca_precalculated=pca_res)

if args.distance_restraint > 0:
    sampler.bias.append(LinearDistanceRestraint(refseq, args.distance_restraint))

if args.pca_bias > 0:
    sampler.bias.append( LatentVoyagerPotential(refseq, args.pca_bias, pca_precalculated=pca_res))

# Run the actual sampler
sampler.run(n_samples, parallel_method=args.parallel_method)

# Generate alignments from sampler
sampler.generate_alignments(burnin=args.burnin)

# Save results
sampler.save(args.output_file)
