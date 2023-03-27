conda activate
#!/usr/bin/env bash

TESTDIR=`python3 -c 'import os; import bandits_to_rank; print(os.path.dirname(bandits_to_rank.__path__[0]))'`/Automatisation/igrida/test
mkdir $TESTDIR/dev_null
mkdir $TESTDIR/dev_null_bis

set -x
#todo : local scratchDIR
# basic usage
python3 ../exp.py --play 2 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null $TESTDIR/dev_null_bis || exit

# KDD subtleties
python3 ../exp.py --play 2 100 -r 10 --KDD 0 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 100 -r 10 --KDD 1 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --KDD 0 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --KDD_all --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# order_kappa
python3 ../exp.py --play 2 100 -r 10 --order_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --order_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --order_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --shuffle_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --shuffle_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --shuffle_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --shuffle_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --shuffle_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --shuffle_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --increasing_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --increasing_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --increasing_kappa --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --increasing_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --increasing_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --increasing_kappa_except_first --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --order_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --order_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --order_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --shuffle_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --shuffle_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --shuffle_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --shuffle_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --shuffle_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --shuffle_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --increasing_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --increasing_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --increasing_kappa --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --increasing_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 -s 2 100 -r 10 --increasing_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --increasing_kappa_except_first --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Yandex
python3 ../exp.py --play 2 100 -r 10 --Yandex 0 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 100 -r 10 --Yandex 1 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --Yandex 0 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_all --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Yandex CM
python3 ../exp.py --play 2 100 -r 10 --Yandex_CM 0 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 100 -r 10 --Yandex_CM 1 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_CM 0 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_CM_all --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Yandex CM L et K variable
python3 ../exp.py --play 2 100 -r 10 --Yandex_CM 0 5 10 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 100 -r 10 --Yandex_CM 1 5 10 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_CM 0 5 10 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_CM_all 5 10 --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Yandex K=L=5
python3 ../exp.py --play 2 100 -r 10 --Yandex_equi 0 5 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --play 2 100 -r 10 --Yandex_equi 1 5 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_equi 0 5 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --Yandex_equi_all 5 --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Other dataset
python3 ../exp.py --play 2 100 -r 10 --std --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --std --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --big --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --big --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --xsmall --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --xsmall --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --xxsmall --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --xxsmall --eGreedy 0.1 10 $TESTDIR/dev_null || exit


python3 ../exp.py --play 2 100 -r 10 --test --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test --eGreedy 0.1 10 $TESTDIR/dev_null || exit


# Other dataset CM
python3 ../exp.py --play 2 100 -r 10 --std_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --std_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --big_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --big_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --xsmall_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --xsmall_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --xxsmall_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --xxsmall_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit


python3 ../exp.py --play 2 100 -r 10 --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test_CM --eGreedy 0.1 10 $TESTDIR/dev_null || exit


# Other algorithms
python3 ../exp.py --play 2 100 -r 10 --small --eGreedy 0.1 10 --noSVD $TESTDIR/dev_null --force --nb_checkpoints 1  || exit # todo: does not support checkpoints for the time being
python3 ../exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 --noSVD $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-TS --greedySVD $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-TS --greedySVD $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-TS --greedyMLE $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-TS --greedyMLE $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-TS --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-TS --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-PIE 0.01 --greedySVD $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-PIE 0.01 --greedySVD $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-PIE 0.01 --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-PIE 0.01 --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-UCB 0.01 --greedySVD $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-UCB 0.01 --greedySVD $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBM-UCB 0.01 --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBM-UCB 0.01 --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --BC-MPTS --greedySVD $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --BC-MPTS --greedySVD $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --BC-MPTS --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --BC-MPTS --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --vari_sigma $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --vari_sigma $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --LGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --vari_sigma $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --vari_sigma $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --vari_sigma --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --RR 0.1 TGRW-LGRW-Max_Position-Pseudo_View --vari_sigma --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --MaxPos  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --MaxPos  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --MaxPos  --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --MaxPos --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --PseudoView $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --PseudoView $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PB-MHB 2 --PseudoView  --random_start $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PB-MHB 2 --PseudoView  --random_start $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --BubbleRank 0.5 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --BubbleRank 0.5 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --BubbleRank 0.5 --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --BubbleRank 0.5 --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --TopRank 100 --horizon_time_known $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --TopRank 100 --horizon_time_known $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --TopRank 100 --horizon_time_known --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --TopRank 100 --horizon_time_known --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --TopRank 50 --horizon_time_known --doubling_trick --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --TopRank 50 --horizon_time_known --doubling_trick --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --TopRank 0.1 --oracle $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --TopRank 0.1  --oracle $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --OSUB 100  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --OSUB 100  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --OSUB 500 --finit_memory  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --OSUB 500 --finit_memory  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --OSUB_PBM 100  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --OSUB_PBM 100  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --UniPBRank --known_horizon 100 --gamma 5 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --UniPBRank --known_horizon 100 --gamma 5 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --UniRank $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --UniRank $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --UniTopRank --version 2 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --UniTopRank --version 2  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --DCGUniRank $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --DCGUniRank $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PBubblePBRank 100 5  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PBubblePBRank 100 5  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --BubbleOSUB 100  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --BubbleOSUB 100  $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --small --PMED 1. 10 40 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --small --PMED 1. 10 40 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --test --MLMR $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test --MLMR $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --test --KL-MLMR 100 $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test --KL-MLMR 100 $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --test --CascadeKL-UCB  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test --CascadeKL-UCB $TESTDIR/dev_null || exit

python3 ../exp.py --play 2 100 -r 10 --test --PB-GB 1 0.001  $TESTDIR/dev_null --force || exit
python3 ../exp.py --merge 100 -r 10 --test --PB-GB 1 0.001 $TESTDIR/dev_null || exit

# --force
python3 ../exp.py --play 6 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 ../exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# checkpoints
python3 ../exp.py --play 1 100 -r 10 --small --eGreedy 1 10 $TESTDIR/dev_null --force  || exit
python3 ../exp.py --play 1 100 -r 10 --small --eGreedy 1 10 $TESTDIR/dev_null_bis --nb_checkpoints 5 --force  || exit
# todo: handle random number generator and ensure that checkpoints does not change the game
#gzcat  $TESTDIR/dev_null/purely_simulated__small__Bandit_EGreedy_SVD_1.0_c_10_maj__games_100_nb_trials_10_record_length_0_game_id.gz > $TESTDIR/dev_null/blip
#gzcat $TESTDIR/dev_null_bis/purely_simulated__small__Bandit_EGreedy_SVD_1.0_c_10_maj__games_100_nb_trials_10_record_length_0_game_id.gz > $TESTDIR/dev_null/blop
#diff $TESTDIR/dev_null/blip $TESTDIR/dev_null/blop

