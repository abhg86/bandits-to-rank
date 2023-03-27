#!/bin/bash

#OAR -l core=1,walltime=00:05:00
#OAR -O /srv/tempdd/abehaege/SCRATCH/fake_job.%jobid%.output
#OAR -E /srv/tempdd/abehaege/SCRATCH/fake_job.%jobid%.output

set -xv


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
#   but rather in /srv/tempdd/...
#   Writing directly somewhere in $HOME/... will necessarily cause NFS problems at some time.
#   Please respect this policy.
#
# * The program to run may be somewhere in your $HOME/... however
#
##########################################################################
"

SCRATCHDIR=/srv/tempdd/abehaege/SCRATCH/
TMPDIR=$SCRATCHDIR/$OAR_JOB_ID
mkdir -p $TMPDIR
cd $TMPDIR

echo "pwd :"
pwd

echo
echo "=============== RUN ==============="
export PYTHONPATH="~/projet_bandit_to_rank/bandits-to-rank2"

python3 ~/projet_bandit_to_rank/bandits-to-rank2/Automatisation/igrida/exp.py --play 1 1000 -r 1000 --order_kappa --delta_variation_01 --GRAB --gap_type first --optimism TS ~/projet_bandit_to_rank/bandits-to-rank2/output/separated

