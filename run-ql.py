import argparse
import os
import sys
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""SUMO Q-Learning""")
    prs.add_argument("-n", dest="network", type=str,
                     default='nets/5x5-Raphael/synthetic.net.xml',
                     help="Network definition xml file.\n")
    prs.add_argument("-r", dest="route", type=str,
                     default='nets/5x5-Raphael/flow.rou.xml',
                     help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-o", dest="out_csv", type=str, default='', required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=60, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run without RL on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=80000, required=False,
                     help="Number of simulation seconds.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0].replace(' ', '_')
    #scenario = args.network.replace('nets/5x5-Raphael/', '').replace('.net.xml', '')
    scenario = args.network.replace('nets/charlottenburg/', '').replace('.net.xml', '')
    out_csv = f'outputs/charlottenburg/actuated'

    env = SumoEnvironment(net_file=args.network,
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          time_to_teleport=90,
                          max_depart_delay=0)


    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                             state_space=env.observation_spaces(ts),
                             action_space=env.action_spaces(ts),
                             alpha=args.alpha,
                             gamma=args.gamma,
                             exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}
    infos = []
    done = {'__all__': False}
    if args.fixed:
        while not done['__all__']:
            current_step = env.sim_step
            if current_step % 500 == 0:
                print(f'current step:  {current_step}')
            _, _, done, _ = env.step({})

    else:
        while not done['__all__']:
            current_step = env.sim_step
            if current_step % 500 == 0:
                print(f'current step:  {current_step}')
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

    env.save_csv(out_csv_name=out_csv, run=1)
