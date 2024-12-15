import gym
from gym import spaces
from transformer_assembly.env.assembly_tree import AssemblyTree
import numpy as np
import collections
import random
import torch

import torch

def process_state(state, max_pipeline_length=None):
    """
    Convert the dictionary-based state into a single flattened tensor.
    This handles variable-length pipelines by padding them to a consistent length.
    """
    # Convert vector_of_inventory to tensor
    vector_of_inventory = torch.tensor(state['vector_of_inventory'], dtype=torch.float32)

    # Handle matrix_of_pipeline with padding
    pipelines = [list(pipe) for pipe in state['matrix_of_pipeline']]

    # Determine the max pipeline length if not provided
    if max_pipeline_length is None:
        max_pipeline_length = max(len(pipe) for pipe in pipelines)

    # Pad each pipeline to the max_pipeline_length with zeros
    padded_pipelines = [pipe + [0] * (max_pipeline_length - len(pipe)) for pipe in pipelines]
    
    # Convert to tensor and flatten
    matrix_of_pipeline = torch.tensor(padded_pipelines, dtype=torch.float32).flatten()

    # Convert current_node to tensor
    current_node = torch.tensor([state['current_node']], dtype=torch.float32)

    # Concatenate all components into a single tensor
    return torch.cat([vector_of_inventory, matrix_of_pipeline, current_node])



class AssemblyTreeEnv(gym.Env):
    def __init__(self, json_config, seed=None):
        super(AssemblyTreeEnv, self).__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.assembly_tree = AssemblyTree(json_config)
        self.num_of_nodes = json_config['num_of_nodes']
        self.max_order_size = json_config['MaxOrderSize']
        self.MinOrder = json_config['MinOrder']
        self.back_order_cost = json_config['back_order_cost']
        self.lostSalesPenalty = json_config['lostSalesPenalty']
        self.MaxInvNodes = json_config['MaxInvNodes']
        self.MaxBacklog= json_config['MaxBacklog']
        self.max_inventory = min(self.MaxInvNodes)
        self.I_offset = self.max_inventory  # offset for end node 0, allow to be negative [-max_inventory, max_inventory]
        self.demand_probs = json_config['DemandProbs']
        self.max_lead_time = max(self.assembly_tree.leadTimes.values()) # max lead time of all nodes
        self.demand_offset = json_config['Demand_offset']
        self.MinSafetyStock = json_config['MinSafetyStock']

        # action space [0, max_order_size]
        self.action_space = spaces.Discrete(self.max_order_size + 1)

        # Initialize state
        self.reset()

        # vector of observation space, pipeline of orders of all nodes, current inventory level of all nodes, current node
        # Notice that the inventory of end product can be negative, however, all other nodes can only be >=0. Index from 0 ... N-1
        states_dim = np.append(np.append(2 * self.max_inventory, np.full(self.num_of_nodes - 1, self.max_inventory)),
                               np.append(np.full(self.assembly_tree.get_total_lead_time(), self.max_order_size),
                               self.num_of_nodes - 1))
        self.state_space = gym.spaces.MultiDiscrete(states_dim)
        self.observation_space = self.state_space

    def generate_random_demand(self):
        demand_values = list(range(len(self.demand_probs)))
        demand = random.choices(demand_values, weights=self.demand_probs, k=1)[0] + self.demand_offset
        return demand

    def step(self, action, demand=None):
        if self.state['state_category'] == 'AwaitAction':
            
            if not self.isAllowedAction(action):
                raise ValueError(f"Action {action} is not allowed for node {self.state['current_node']}!")

            self.ModifyStateWithAction(action)

            if self.state['current_node'] > 0:
                self.state['current_node'] -= 1
            else:
                self.state['state_category'] = 'AwaitEvent'
            reward = 0 
            info = {} 

        elif self.state['state_category'] == 'AwaitEvent':
            reward, holding_cost, backorder_cost = self.ModifyStateWithEvent(demand) # maximize reward = -cost
            reward = -reward
            self.state['current_node'] = self.num_of_nodes - 1
            self.state['state_category'] = 'AwaitAction'
            info = { "holding_cost": holding_cost,
                     "backorder_cost": backorder_cost}

        done = self.check_done()

        self.state['vector_of_inventory'] = [max(inv, self.MinSafetyStock[i])
        for i, inv in enumerate(self.state['vector_of_inventory'])
        ]
        

        observation = self.state
        return observation, reward, done, info

    def reset(self):
        """Reset environment to initial state."""
        self.state = {
            'vector_of_inventory': [self.I_offset + self.max_order_size if i == 0 else self.max_order_size for i in range(self.num_of_nodes)], 
            'matrix_of_pipeline': [collections.deque([self.assembly_tree.capacities[i]] * (self.assembly_tree.leadTimes[i])) for i in range(self.num_of_nodes)],
            'current_node': self.num_of_nodes - 1,
            'state_category': 'AwaitAction'
        }
        return self.state

    def render(self, mode='human'):
        print(f"Inventory: {self.state['vector_of_inventory']}")
        print(f"Pipeline: {self.state['matrix_of_pipeline']}")
        print(f"Current Node: {self.state['current_node']}")

    def check_done(self):
        return False


    def ModifyStateWithAction(self, action):
        node_id = self.state['current_node']
        # print('ModifyStateWithAction, node id : ', node_id)

        for child_id in self.assembly_tree.childrenNodes[node_id]:
            self.state['vector_of_inventory'][child_id] -= (action + self.MinOrder)

        self.state['matrix_of_pipeline'][node_id][-1] = (action + self.MinOrder)
        return 0.0

    def ModifyStateWithEvent(self, demand):
        cost = 0.0
        holding_cost = 0.0
        backorder_cost = 0.0
        # print('ModifyStateWithEvent, demand: ', demand)
        def calculateCostAndUpdateInventory(i, temp_cost, holding_cost, backorder_cost):
            if i == 0:
                self.state['vector_of_inventory'][i] -= demand
                actual_inventory = self.state['vector_of_inventory'][i] - self.I_offset
                if actual_inventory >= 0:
                    temp_cost += actual_inventory * self.assembly_tree.holdingCosts[i]
                    holding_cost += actual_inventory * self.assembly_tree.holdingCosts[i]
                elif actual_inventory < 0 and actual_inventory >= -self.MaxBacklog:
                    temp_cost += abs(actual_inventory) * self.back_order_cost
                    backorder_cost += abs(actual_inventory) * self.back_order_cost
                else:
                    temp_cost += abs(actual_inventory) * self.lostSalesPenalty
                    backorder_cost += abs(actual_inventory) * self.lostSalesPenalty
            else:
                temp_cost += (self.state['vector_of_inventory'][i]) * self.assembly_tree.holdingCosts[i]
                holding_cost += self.state['vector_of_inventory'][i] * self.assembly_tree.holdingCosts[i]
            return temp_cost, holding_cost, backorder_cost

        def updatePipelineOrders(i):
            if self.state['matrix_of_pipeline'][i]:
                arriving_order = self.state['matrix_of_pipeline'][i].popleft()
                self.state['vector_of_inventory'][i] += arriving_order
                self.state['matrix_of_pipeline'][i].append(0)

        for i in range(self.num_of_nodes):
            updatePipelineOrders(i)

        for i in range(self.num_of_nodes):
            cost, holding_cost, backorder_cost = calculateCostAndUpdateInventory(i, cost, holding_cost, backorder_cost)

        self.state['current_node'] = self.num_of_nodes - 1
        self.state['state_category'] = 'AwaitAction'

        return cost, holding_cost, backorder_cost

    def isAllowedAction(self, action):
        node_id = self.state['current_node']
        node_inventory_in_pipeline = sum(self.state['matrix_of_pipeline'][node_id])

        # Condition 1: Check if the action is negative
        if action < 0:
            # print("Action is negative, not allowed.")
            return False

        # Condition 2: Check if action exceeds node capacity
        if action + self.MinOrder > self.assembly_tree.capacities[node_id]:
            # print(f"Action + MinOrder exceeds capacity: {action} + {self.MinOrder} > {self.assembly_tree.capacities[node_id]}")
            return False

        # Condition 3: Check if action causes inventory overflow
        if action + self.MinOrder + self.state['vector_of_inventory'][node_id] + node_inventory_in_pipeline > self.MaxInvNodes[node_id]:
            # print(f"Inventory overflow: {action} + {self.MinOrder} + {self.state['vector_of_inventory'][node_id]} + {node_inventory_in_pipeline} > {self.MaxInvNodes[node_id]}")
            return False

        # Condition 4: Check if any child node's inventory is lower than the action + MinOrder
        for child_id in self.assembly_tree.childrenNodes[node_id]:
            if self.state['vector_of_inventory'][child_id] < (action + self.MinOrder):
                # print(f"Child node {child_id} inventory is too low: {self.state['vector_of_inventory'][child_id]} < {action + self.MinOrder}")
                return False
                
        return True
    
    def checkSafetyStock(self):
        return


 