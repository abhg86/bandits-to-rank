#!/usr/bin/env bash

libdir=`python3 -c "import bandits_to_rank; import os; print(os.path.dirname(bandits_to_rank.__path__[0]))"`
echo $libdir

rsync -azP $libdir/ rgaudel@transit.irisa.fr:/udd/rgaudel/bandits-to-rank --exclude='.git/' --exclude='.idea/' --exclude='venv/' --exclude='KDD_filtered.csv' --exclude='Test/exp_ECAI2020/result/real_KDD' --exclude='Test/exp_ECAI2020/result/simul' --exclude='Test/exp_ECAI2020/result/graph' --exclude='Test/exp_CIKM2020/result/real_KDD' --exclude='Test/exp_CIKM2020/result/real_Yandex' --exclude='Test/exp_CIKM2020/result/simul' --exclude='Test/exp_CIKM2020/result/graph' --exclude='Test/exp_AAAI2021/results/real_KDD' --exclude='Test/exp_AAAI2021/results/real_Yandex' --exclude='Test/exp_AAAI2021/results/simul' --exclude='Test/exp_AAAI2021/results/graph' --exclude='Automatisation/igrida/test/dev_null'  --exclude='Automatisation/igrida/test/dev_null_bis'

# todo: exclude __pycache__ and checkpoints and .idea and DSSTORE and ...