Short version of README
by RG for RG


# General

* python3 -m site --user-site
* ssh transit.irisa.fr
* ssh igrida


# Test at home

awk '{ system("../exp.py " $0) }' python_param.txt

# Run on igrida
* ~/Desktop/louis_vuitton/code/bandits-to-rank/rg_playground/push_to_igrida.sh
* rsync -azP rgaudel@transit.irisa.fr:/temp_dd/igrida-fs1/$USER/SCRATCH/results/*game_id.gz $SCRATCHDIR/results/
    * **Warning**: the character `/` at the end of the directory names is required


# todo

* checkpoints for EM inference of thetas/kappas
    * ? by writing our own library for inference ?
    * ? by enabling dill ? (require to install it on igrida nodes with `pip`)
* Automatic resubmission on IGRIDA
>>> # How can a checkpointable job be resubmitted automatically?
>>> You have to specify that your job is idempotent and exit from your script with the exit code 99. So, after a successful checkpoint, if the job is resubmitted then all will go right and there will have no problem (like file creation, deletion, ...).
>>> ## Example:
>>> `oarsub --checkpoint 600 --signal 2 -t idempotent /path/to/prog`
>>> So this job will send a signal SIGINT (see man kill to know signal numbers) 10 minutes before the walltime ends. Then if everything goes well and the exit code is 99 it will be resubmitted.

