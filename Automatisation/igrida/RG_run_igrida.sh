#!/usr/bin/env bash

set -xv

DIR="/udd/rgaudel/bandits-to-rank/Automatisation/igrida"

cd "$DIR/PB_MHB__both" ;        oarsub -S ./oar_param.sh
#cd "$DIR/PBM_TS__both" ;        oarsub -S ./oar_param.sh
#cd "$DIR/BubbleRank__both" ;    oarsub -S ./oar_param.sh
cd "$DIR/TopRank__both" ;       oarsub -S ./oar_param.sh
#cd "$DIR/BC_MPTS__both" ;       oarsub -S ./oar_param.sh
#cd "$DIR/egreedy__both" ;       oarsub -S ./oar_param.sh


oarstat --user -c --format 1

