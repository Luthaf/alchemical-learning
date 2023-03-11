Writing down some of the things that are necessary to make this code run.

1st make a new env

2nd make sure to have a rust compiler installed.
    either through:
        conda install -c conda-forge rust
    or through (not tested):
        pip install setuptools_rust
    
    Also, check that cudNN and cuda are installed
         conda install -c anaconda cudnn 
         conda install cuda -c nvidia

3rd install the specific git commits (current requirements file does NOT work)
    pip install git+https://github.com/lab-cosmo/equistore.git@6cd0c9a518cc1b6ca99cfa30df2a070ed1f6fe0d
    pip install git+https://github.com/Luthaf/rascaline.git@fbec094e26d51333490f511c54fb970ed10df80c
    pip install git+https://github.com/Luthaf/rascaline-torch.git@28a06cb17287baab601cfaa695e8025f773bd7aa

4th clone the alchemical-learning repository and chekout the ./ensemble-nn branch 

5th in order to make the ipi-side of the code run, clone from the lab-cosmo/i-pi fork and checkout the alchemy-paper branch
    first pip install the code then make the f90 extensions, otherwise you get some weird pip setuptools error regarding UTF8
    
6th within the ./i-pi/drivers/py/pes/ directory make a softlink to the alchemical-learning directory

7th also, add the i-pi driver from ./i-pi/bin/ to your PATH

8th using these commits/versions the example.json parameterfile is broken and the "1"/"2" in the "radial_per_angular" have to be replaced with actual ints

9th When debugging things, make sure to set torch.set_default_dtype(torch.float64), otherwise you might get untracable error messages from rascaline_torch
