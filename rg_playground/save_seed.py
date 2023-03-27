#!/usr/bin/python3
# -*- coding: utf-8 -*-


import gzip
import json


def record_zip(filename,dico):
    print (type(dico))
    print ('file',filename)
    json_str = json.dumps(dico)
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json_bytes)
    return 'done'

def retrieve_data_from_zip(file_name):
    with gzip.GzipFile(file_name, 'r') as fin:
        json_bytes = fin.read()

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    return json.loads(json_str)


if __name__ == '__main2__':
    ### Marche très bien
    import tempfile
    import gzip
    import numpy as np
    import random
    import ast

    nprs = list(np.random.get_state())
    nprs[1] = list([int(val) for val in nprs[1]])
    record = {'python' : repr(random.getstate()), 'numpy' : nprs}

    print(np.random.rand(4))

    random.setstate(ast.literal_eval(record['python']))
    nprs = record['numpy']
    nprs[1] = np.array(nprs[1], dtype=np.uint32)
    np.random.set_state(tuple(nprs))

    print(np.random.rand(4))


if __name__ == '__main__':
    ### même idée ne fonctionne step qund appliquée à une partie. Il doit y avoir un générateur aléatoire que utilisé autre que celui de python et de numpy
    from bandits_to_rank.opponents import greedy
    from bandits_to_rank.referee import Referee
    from bandits_to_rank.environment import Environment_PBM

    import tempfile
    import gzip
    import numpy as np
    import random
    import ast

    #### Environment
    kappas = [1, 0.7, 0.5, 0.5, 0.3]
    thetas = [0.9, 0.3, 0.5, 0.4, 0.1, 0.2, 0.5, 0.1, 0.8, 0.4]
    nb_prop = len(thetas)
    nb_place = len(kappas)
    env = Environment_PBM(thetas, kappas)

    #### Referee
    nb_trials = 20
    len_record_short = 5
    referee = Referee(env, nb_trials, all_time_record=False, len_record_short=len_record_short)

    #### Player
    c = 1
    maj = 1
    player = greedy.greedy_EGreedy(c, nb_prop, nb_place, maj)

    #### Play first game
    nprs = list(np.random.get_state())
    nprs[1] = list([int(val) for val in nprs[1]])
    record = {'python' : repr(random.getstate()), 'numpy' : nprs}
    referee.play_game(player)

    """
    ### Save/load results
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpfile_name = tmpdir_name + '/my.gz'
        record_zip(tmpfile_name, referee.record_results)
        referee.record_results = retrieve_data_from_zip(tmpfile_name)
    """

    random.setstate(ast.literal_eval(record['python']))
    nprs = record['numpy']
    nprs[1] = np.array(nprs[1], dtype=np.uint32)
    np.random.set_state(tuple(nprs))
    #### Re-play first game
    referee.play_game(player, random_generator_state=referee.record_results['random_generator_state'][0])

    #### Compare results
    assert np.all(referee.record_results['reward'][0] == referee.record_results['reward'][1]) , str(referee.record_results)
    print(referee.record_results)


