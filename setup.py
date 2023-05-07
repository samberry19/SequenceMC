from distutils.core import setup
setup(name='SequenceMC',
      version='0.0.2',
      author='Sam Berry',
      author_email="sberry@g.harvard.edu",
      description = "Monte Carlo samplers for aligned biological sequences",
      py_modules=["SequenceMC", "SequenceMC.samplers", "SequenceMC.dca", "SequenceMC.bias", "SequenceMC.utils", "SequenceMC.evolve"]
      )
