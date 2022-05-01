# DCAsamplers
A Metropolis sampler for generating sequences from DCA models, with optional added potentials for biasing sequences to be close to a reference sequence or to "voyage" in a latent space.

At the moment, only models trained with EVcouplings are supported, but I would like to change this eventually.

For example, to load in an EVcouplings model and run for 1 million iterations across 5 chains and a temperature of 1:

```
from DCAsamplers import *
from evcouplings.couplings import CouplingsModel

model = CouplingsModel(model_file)
sampler = OneHotDCASampler(model, N_chains=5, T=1)
sampler.run(1000000)
sampler.generate_alignments(burnin=0.2)
sampler.save("example_sequences.pkl")
```

Let's say you wanted to sample sequences relatively close to a reference sequence.

```
sampler = OneHotDCASampler(model, N_chains=5, T=1,
            extra_potential = lambda s: 9 * MutationalSteps(s, to_numeric(model.seq())))
```
