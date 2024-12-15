import numpy as np
import collections

class TreeNode:
    def __init__(self, id, capacity, holding_cost, lead_time):
        self.id = id
        self.capacity = capacity
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.parent = None
        self.children = []

class AssemblyTree:
    def __init__(self, json_config):
        self.root = None
        self.childrenNodes = {}
        self.upstreamNodes = {}
        self.capacities = {}
        self.holdingCosts = {}
        self.leadTimes = {}
        self.num_of_nodes = json_config['num_of_nodes']
        self.back_order_cost = json_config['back_order_cost']
        self.MinOrder = json_config['MinOrder']

        # Initialize tree
        self.loadTreeFromJson(json_config['assembly_tree'])
        self.initialize()

    def addChild(self, parent, child_id, capacity, holding_cost, lead_time):
        child = TreeNode(child_id, capacity, holding_cost, lead_time)
        child.parent = parent
        parent.children.append(child)
        return child

    def initialize(self):
        self.childrenNodes.clear()
        self.upstreamNodes.clear()
        self.capacities.clear()
        self.holdingCosts.clear()
        self.leadTimes.clear()
        self._populateDictionaries(self.root)

    def _populateDictionaries(self, node):
        if node is None:
            return 
        self.childrenNodes[node.id] = [] # only direct children nodes
        self.upstreamNodes[node.id] = [] # all upstream nodes, direct and non-direct
        self.capacities[node.id] = node.capacity
        self.holdingCosts[node.id] = node.holding_cost
        self.leadTimes[node.id] = node.lead_time

        for child in node.children:
            self.childrenNodes[node.id].append(child.id)
            self._populateDictionaries(child)

        current = node
        while current.parent:
            if node.id not in self.upstreamNodes:
                self.upstreamNodes[node.id] = []
            self.upstreamNodes[node.id].append(current.parent.id)
            current = current.parent

    def loadTreeFromJson(self, tree_json):
        # root node
        self.root = TreeNode(
            tree_json['id'], 
            tree_json['capacity'], 
            tree_json['holding_cost'], 
            tree_json['lead_time']
        )
        self._loadChildrenFromJson(tree_json, self.root)

    def _loadChildrenFromJson(self, tree_json, parent):
        if 'children' not in tree_json:
            return
        for child_json in tree_json['children']:
            child = self.addChild(
                parent,
                child_json['id'],
                child_json['capacity'],
                child_json['holding_cost'],
                child_json['lead_time']
            )
            self._loadChildrenFromJson(child_json, child)

    def __str__(self):
        return self._to_string(self.root)

    def _to_string(self, node, depth=0):
        if node is None:
            return ""

        indent = "  " * depth
        result = f"{indent}Node ID: {node.id}, Capacity: {node.capacity}, "
        result += f"Holding Cost: {node.holding_cost}, Lead Time: {node.lead_time}\n"

        for child in node.children:
            result += self._to_string(child, depth + 1)
        
        return result
    
    def get_total_lead_time(self, node=None):
        if node is None:
            node = self.root

        total_lead_time = node.lead_time
        for child in node.children:
            total_lead_time += self.get_total_lead_time(child)

        return total_lead_time
    
