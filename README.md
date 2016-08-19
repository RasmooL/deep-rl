# deep-rl
## NOTE: Not being updated or worked on anymore, some things are unfinished, contain badly documented additions or are perhaps buggy. Feel free to use the code for anything, but beware.
Deep Reinforcement Learning in [Tensorflow](https://github.com/tensorflow/tensorflow).
Aimed to reproduce results from some recent interesting papers.

## Contents
Currently implemented, somewhat tested:
- Vanilla DQN with Nature config, plus various experiments: dqn.NatureDQN.py & run_nature.py
- [Double Q-learning](http://arxiv.org/abs/1509.06461): dqn.DoubleDQN.py & run_double.py

Small list of other things that could be interesting (but are only partly implemented or not at all)
- [Dueling network](http://arxiv.org/pdf/1511.06581v2.pdf)
- [Bootstrapping](http://arxiv.org/abs/1602.04621)
- [Prioritized experience replay](http://arxiv.org/abs/1511.05952)
- [Asynchronous actor-critic](http://arxiv.org/abs/1602.01783)

## How to run
Install [Tensorflow](https://github.com/tensorflow/tensorflow) in Python 2.7.

Install [Sacred](https://sacred.readthedocs.org/en/latest/index.html) (only necessary for supplied run scripts, you can make your own and not use it).

Install [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment).

Configure the run script you want to use (e.g. run_nature.py).

Run it!


## License
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
