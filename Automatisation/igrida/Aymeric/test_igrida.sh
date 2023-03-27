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
#-- FAKE RUN EXECUTION
echo "Running ..."
sleep 60 # fake job 1 minute

#-- FAKE RUN OUTPUTS
cat > my_program_summary.out << EOF
For example, some short solver statistics are summarized here.
1.e-10
1.e-13
1.e-14
1.e-16
Converged
EOF

echo "Done"
echo "==================================="

#-- ECHO SOME SUMMARY OUTPUTS OF THE RUN IN THE ***.output FILE
echo
cat my_program_summary.out
echo "---------------------"
echo 
echo OK
