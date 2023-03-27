# How to run on Igrida
(using --array option to run several flavors of the same program)


## Step 0: installation on igrida
* Install required Python-libraries (in an interactive session `oarsub -I`)
    * module load python-3.6.8-gcc-4.9.2 py-numpy-1.17.0-gcc-9.1.0 py-pip-19.0.3-gcc-9.1.0 py-scikit-learn-0.21.2-gcc-4.9.2 py-scipy-1.3.0-gcc-4.9.2 py-virtualenv-16.4.1-gcc-9.1.0 py-setuptools-41.0.1-gcc-9.1.0 py-matplotlib-3.1.1-gcc-4.9.2 py-pyparsing-2.3.1-gcc-4.9.2 py-cycler-0.10.0-gcc-4.9.2 py-kiwisolver-1.0.1-gcc-4.9.2
      (new Igrida) module load spack/python/3.7.7 spack/python/3.7.6 spack/py-numpy/1.18.1 spack/py-setuptools/46.1.3 spack/py-virtualenv/16.7.6  spack/py-pip/19.3
    * python3 -m pip install --user docopt pandas statsmodels IPython seaborn
    * in `python3 -m site --user-site` directory, add a file named `bandits_to_rank.pth` which contains the path to *bandits_to_rank* library 
* Prepare your [ouput directory and oar script](http://igrida.gforge.inria.fr/tutorial.html#batch-job-using-one-single-core)

## Step 1.0: update the library on igrida

## Step 1.1: write parameters file for the Python script

## Step 1.2: write configuration script for oarsub
* Important parameters
    * name of the parameters file for the Python script
    * walltime
    * name of the Python script

## Step 1.3: run oarsub

> wait for the end of runs

## Step 2.1: download the results files on your own computer

## Step 2.2: merge results files

## Step 2.3: compute statistics


# Example for egreedy KDD
(cf. `egreedy_KDD` directory)
* Step 1.0:  run `$PATH_TO_BANDITS_TO_RANK/rg_playground/push_to_igrida.sh`
    * based on rsync
    * TO BE ADAPTED to your own configuration
    * **Warning**: the character `/` at the end of the directory names is required
* Step 1.1.0: Go to igrida, Go to  `$PATH_TO_BANDITS_TO_RANK/Automatisation/igrida/egreedy_KDD/`
* Step 1.1: run `python prepare_python_param.py`
    * script done to run both with Python2 and Python3
    * create file `python_param.txt`
* Step 1.2: set the right value for walltime in `oar_param.sh`
* Step 1.3: run `oarsub -S ./oar_param.sh` 

> wait for the end of runs

* Step 2.1: run `rsync -azP rgaudel@transit.irisa.fr:/temp_dd/igrida-fs1/$USER/SCRATCH/results/KDD* $SCRATCHDIR/results`
    * **Warning**: the character `/` at the end of the directory names is required
    * `$SCRATCHDIR` is a directory on your own computer to store raw files
* Step 2.2: run `python3 $PATH_TO_BANDITS_TO_RANK/Automatisation/igrida/egreedy_KDD/merge_records.py`
    * Equivalent to running `python3 $PATH_TO_BANDITS_TO_RANK/Automatisation/igrida/playgame_egreedy_KDD.py --merge ...` several times with proper parameters
    * require the parameters in the script to be set at the right values
* Step 2.3: use your preferred notebook ! 
