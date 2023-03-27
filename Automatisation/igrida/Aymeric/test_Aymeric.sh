export PYTHONPATH="$PYTHONPATH:./bandits_to_rank";

python3 Automatisation/igrida/exp.py --play 1 1000 -r 1000 --order_kappa --delta_variation_01 --GRAB --gap_type first --optimism TS `pwd`/output/separated
# python3 Automatisation/igrida/exp.py --play 2 100 -r 10 --small --eGreedy 0.1 10 `pwd`/outputs/separated
