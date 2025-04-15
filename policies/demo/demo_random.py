from ..base_policy import BasePolicy

import random


class DemoRandom(BasePolicy):

    def act(self, env, task):
        return random.randint(0, len(env.scenario.get_nodes()) - 1)
    
    def priority_list(self, env, task):
        nodes = list(env.scenario.node_id2name.keys())
        random.shuffle(nodes)
        return nodes
