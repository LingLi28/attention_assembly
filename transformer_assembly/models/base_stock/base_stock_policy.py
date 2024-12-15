import numpy as np

class BaseStockPolicy:
    def __init__(self, env, config, if_short_fall = False):
        """
        Initialize BaseStockPolicy with environment (env) and configuration parameters.
        """
        self.env = env
        
        # Provide default values and load parameters from config
        if if_short_fall:
            self.base_stock_levels = config['base_stock_with_shortfall_levels']
        else:
            self.base_stock_levels = config['base_stock_levels']

        self.MinOrder =  self.env.demand_offset


    def get_action(self, state):
        """
        Get the action for the current state based on the base stock policy.
        """
        # Ensure we are not in the AwaitEvent category
        if state['state_category'] == 'AwaitEvent':
            return 0
        
        # Get the current node id
        node_id = state['current_node']
        action = 0
        echelon_inv = 0

        # Calculate echelon inventory for the current node
        if node_id == 0:
            echelon_inv += state['vector_of_inventory'][node_id]- self.env.I_offset
        else: 
            echelon_inv += state['vector_of_inventory'][node_id]
            
        echelon_inv += sum(state['matrix_of_pipeline'][node_id])

        # Calculate echelon inventory for all upstream nodes of the current node
        for upstream_node_id in self.env.assembly_tree.upstreamNodes[node_id]:
            echelon_inv += sum(state['matrix_of_pipeline'][upstream_node_id])
            if upstream_node_id == 0:
                echelon_inv += state['vector_of_inventory'][upstream_node_id] - self.env.I_offset
            else:
                echelon_inv += state['vector_of_inventory'][upstream_node_id]

        # Determine the desired action based on base stock levels
        action = self.base_stock_levels[node_id] - echelon_inv - self.MinOrder 

        # Ensure the action is within allowed limits
        if action <= 0:
            return 0

        # Action should not exceed node capacity minus MinOrder
        max_allowed_action = self.env.assembly_tree.capacities[node_id] - self.MinOrder
        if action > max_allowed_action:
            action = max_allowed_action

        # Decrement action if not allowed until a valid action is found
        while action > 0:
            if self.env.isAllowedAction(action):
                return action
            action -= 1

        return 0

