#!/usr/bin/env bash

IGRIDA="/Users/rgaudel/Desktop/louis_vuitton/code/bandits-to-rank/Automatisation/igrida"

rsync -azP rgaudel@transit.irisa.fr:/temp_dd/igrida-fs1/$USER/SCRATCH/results/*game_id.gz $SCRATCHDIR/results/

for case in PB_MHB__both PBM_TS__both BubbleRank__both TopRank__both
do
    cd "$IGRIDA/$case"
    pwd
    ./prepare_python_param.py
    ./merge_records.py
done

for case in  BC_MPTS__both egreedy__both
do
    cd "$IGRIDA/$case"
    pwd
    #./prepare_python_param.py
    #./merge_records.py
done

#$IGRIDA/rg_playground/push_to_igrida.sh

