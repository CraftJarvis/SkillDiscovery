import gym
import pickle

from behavioural_cloning import load_model_parameters
from agent import MineRLAgent
from inverse_dynamics_model import IDMAgent

DEVICE = 'cuda'

def load_minecraft_model(model:str, weights:str):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(model)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(weights)
    env.close()
    return agent

def load_idm_model(model:str, weights:str):
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    return agent