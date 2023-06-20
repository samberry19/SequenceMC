# SequenceMC

Metropolis and Gibbs samplers for aligned biological sequences. Particular support for sampling from DCA models with additional added bias potentials, but can support any scoring function of sequences based on position.

I use my "bioviper" code for convenient manipulation of MSAs, [bioviper](https://github.com/samberry19/bioviper), which can be installed via:

```pip install git+https://github.com/samberry19/bioviper/.```

(although the core of the code works without it) 

Simple example usage would be:

installation:
```
pip install biopython
pip install git+https://github.com/samberry19/bioviper
pip install git+https://github.com/samberry19/SequenceMC
```

and then doing some sampling:
```
import SequenceMC as smc

logit_function = # define this however you'd like, taking in a numerically-encoded sequence
                 # by default the numerical encoding of sequences is given by default_aa_alphabet, although if you change the default_aa_alphabet you can change that
                 # default_aa_alphabet is the gap character '-' as zero followed by all amino acids alphabetically 1-20

sampler = smc.BaseSampler(logit_function,
                          L= , #aligned length of the sequences
                          N_chains=3, #number of MCMC chains to run
                          T=T, # computational "temperature" of the simulation
                          initialization="random", #could also pass defined with a "starting_seq" if you want to start in a particular place
                          record_freq=1) # how often to record - I like to do every time for gibbs or ever 10,000 for Metropolis

sampler.run(1000, method="gibbs", parallel_method="None") # run 1000 iterations of gibbs sampling
                                                          # method="metropolis" will run metropolis sampling (remember to simulate for waaay longer)

sampler.generate_alignments(burnin=0.2) # use 20% of the simulation as burnin and discard
                                        # will make a single alignment from all chains and attach it to the sampler as sampler.alignment

# save the alignment and do whatever you'd like with it
sampler.alignment.save("sampled_sequences.fa", "fasta")
```
