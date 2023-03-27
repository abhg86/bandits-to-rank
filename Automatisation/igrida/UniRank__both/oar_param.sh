#!/bin/bash

#OAR -l core=1,walltime=10:0:00
EXECUTABLE="python3 $HOME/src/bandits-to-rank/Automatisation/igrida/exp.py"
EXERSYNC="rsync -azP /srv/tempdd/cgauthie/SCRATCH/results /udd/cgauthie/src/results --exclude='pickle.gz/' --exclude='.old/'"
#OAR --array-param-file ./python_param.txt
export SCRATCHDIR="/srv/tempdd/cgauthie/SCRATCH"
#OAR --name=CascadeKL_UCB

#OAR -O /temp_dd/igrida-fs1/cgauthie/SCRATCH/outputs/CascadeKL_UCB.%jobid%.output
#OAR -E /temp_dd/igrida-fs1/cgauthie/SCRATCH/outputs/CascadeKL_UCB.%jobid%.error
set -xv

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh

echo
echo OAR_WORKDIR : $OAR_WORKDIR
echo
echo "cat \$OAR_NODE_FILE :"
cat $OAR_NODE_FILE
echo

echo "
##########################################################################
# Where will your run take place ?
#
# * It is NOT recommanded to run in $HOME/... (especially to write),
#   but rather in /temp_dd/igrida-fs1/...
#   Writing directly somewhere in $HOME/... will necessarily cause NFS problems at some time.
#   Please respect this policy.
#
# * The program to run may be somewhere in your $HOME/... however
#
##########################################################################
"

#TMPDIR=$SCRATCHDIR/$OAR_JOB_ID
#mkdir -p $TMPDIR
#cd $TMPDIR


module load python-3.6.8-gcc-4.9.2
module load py-numpy-1.17.0-gcc-9.1.0
module load py-pip-19.0.3-gcc-9.1.0
module load py-scikit-learn-0.21.2-gcc-4.9.2
module load py-scipy-1.3.0-gcc-4.9.2
module load py-virtualenv-16.4.1-gcc-9.1.0
module load py-setuptools-41.0.1-gcc-9.1.0
module load py-matplotlib-3.1.1-gcc-4.9.2
module load py-pyparsing-2.3.1-gcc-4.9.2
module load py-cycler-0.10.0-gcc-4.9.2
module load py-kiwisolver-1.0.1-gcc-4.9.2
module load spack/python/3.7.7
module load spack/python/3.7.6
module load spack/py-numpy/1.18.1
module load spack/py-setuptools/46.1.3
module load spack/py-virtualenv/16.7.6
module load spack/py-pip/19.3

cat /proc/cpuinfo
cat /proc/meminfo
cat /etc/issue

echo
echo "=============== RUN ==============="
echo "Running ..."
time $EXECUTABLE $*
echo "Done"
echo "==================================="


