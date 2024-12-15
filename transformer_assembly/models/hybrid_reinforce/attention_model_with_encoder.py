import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticDynamicEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim, max_lead_time):
        super(StaticDynamicEmbedding, self).__init__()
        self.num_nodes = num_nodes

        H_embedding_dim = 16

        # Static information embeddings (capacity, holding cost, lead time)
        self.capacity_embedding = nn.Linear(1, H_embedding_dim)
        self.holding_cost_embedding = nn.Linear(1, H_embedding_dim)
        self.lead_time_embedding = nn.Linear(1, H_embedding_dim)

        # Dynamic information embeddings (inventory and pipeline)
        self.fc_inventory = nn.Linear(1, H_embedding_dim)
        self.fc_pipeline = nn.Linear(max_lead_time, H_embedding_dim)

        # Used to map the concatenated embeddings to embedding_dim
        self.fc_combined = nn.Linear(5 * H_embedding_dim, embedding_dim)

        self.max_lead_time = max_lead_time

    def forward(self, capacities, holding_costs, lead_times, inventory, pipeline):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure all inputs are on the same device
        capacities = capacities.to(device)
        holding_costs = holding_costs.to(device)
        lead_times = lead_times.to(device)
        inventory = inventory.to(device)
        # Add batch dimension by reshaping
        capacities = capacities.view(1, self.num_nodes, 1)
        holding_costs = holding_costs.view(1, self.num_nodes, 1)
        lead_times = lead_times.view(1, self.num_nodes, 1)
        inventory = inventory.view(1, self.num_nodes, 1)
        # Handle the pipeline for non-batch case
        pipeline = [list(p) + [0] * (self.max_lead_time - len(p)) for p in pipeline]
        padded_pipeline = torch.tensor([pipeline], dtype=torch.float32, device=device)  # Add batch dimension

        # Static embeddings
        cap_emb = self.capacity_embedding(capacities)  # (1, num_nodes, embedding_dim)
        hc_emb = self.holding_cost_embedding(holding_costs)  # (1, num_nodes, embedding_dim)
        lt_emb = self.lead_time_embedding(lead_times)  # (1, num_nodes, embedding_dim)

        # Dynamic embedding (inventory)
        inv_emb = self.fc_inventory(inventory)  # (batch_size, num_nodes, embedding_dim)

        # Ensure pipeline is [batch_size, num_nodes, max_lead_time]
        pipe_emb = self.fc_pipeline(padded_pipeline)  # (batch_size, num_nodes, embedding_dim)

        # Concatenate embeddings: (batch_size, num_nodes, 5 * embedding_dim)
        combined_emb = torch.cat([cap_emb, hc_emb, lt_emb, inv_emb, pipe_emb], dim=-1)

        # Map to final embedding_dim
        final_emb = self.fc_combined(combined_emb)  # (batch_size, num_nodes, embedding_dim)

        # If no batch size was originally present, remove the batch dimension before returning
        final_emb = final_emb.squeeze(0)

        return final_emb

class GlobalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(GlobalEmbedding, self).__init__()
        self.fc_current_node = nn.Linear(1, embedding_dim)

    def forward(self, current_node):
        
        current_node = current_node.unsqueeze(0)  # Add batch dimension
        return self.fc_current_node(current_node.view(-1, 1))


class AttentionModel(nn.Module):
    def __init__(self, number_of_nodes, embedding_dim, num_heads, num_actions, max_lead_time):
        super(AttentionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Multihead Attention parameters
        self.number_of_nodes = number_of_nodes
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.embedding_dim = embedding_dim

        self.static_dynamic_embedding = StaticDynamicEmbedding(self.number_of_nodes, embedding_dim, max_lead_time).to(self.device)
        self.global_embedding = GlobalEmbedding(embedding_dim).to(self.device)

        # Define linear transformations for Query, Key, Value
        self.query_fc = nn.Linear(embedding_dim, embedding_dim)  # Global info as Query
        self.key_fc = nn.Linear(embedding_dim, embedding_dim)    # Static and dynamic info as Key
        self.value_fc = nn.Linear(embedding_dim, embedding_dim)  # Static and dynamic info as Value

        # Linear layer for multi-head attention, maps output back to embedding_dim
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer Normalization
        self.attention_norm = nn.LayerNorm(embedding_dim)

        # Encoder module (256 -> 128 -> 128 -> 64)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Encoder Layer Normalization
        self.encoder_norm = nn.LayerNorm(64)

        # Final Q-value prediction layer
        self.fc_q = nn.Linear(64, num_actions)

    def forward(self, capacities, holding_costs, lead_times, inventory, pipeline, current_node, attention_mask=None):
        
        static_dynamic_emb = self.static_dynamic_embedding(capacities, holding_costs, lead_times, inventory, pipeline)
        global_emb = self.global_embedding(current_node)

        num_of_nodes = static_dynamic_emb.shape[0]
        current_node_id = int(current_node[-1].item())
        assert static_dynamic_emb.shape[-1] == self.embedding_dim
       
        # Repeat global_emb to match batch_size of static_dynamic_emb
        global_emb = global_emb.repeat(num_of_nodes, 1)  # Make global_emb shape (batch_size, embedding_dim)
        
        Q = self.query_fc(global_emb)              # [batch_size, embedding_dim]
        # print('Q shape:', Q.shape)
        K = self.key_fc(static_dynamic_emb)        # [batch_size, embedding_dim]
        V = self.value_fc(static_dynamic_emb)      # [batch_size, embedding_dim]

        # Reshape Q, K, V for multi-head attention: [batch_size, num_nodes, num_heads, head_dim]
        Q = Q.view(self.number_of_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, head_dim]
        K = K.view(self.number_of_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, head_dim]
        V = V.view(self.number_of_nodes, self.num_heads, self.head_dim).transpose(1, 2)  
        
        attention_scores = torch.einsum("bnh,bmh->bnm", Q, K) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute attention output by weighing V with attention weights
        attention_output = torch.einsum("bnm,bmh->bnh", attention_weights, V)
        # print(f"attention_output: {attention_output.shape}")

        # Concatenate heads and reshape back to original embedding dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(num_of_nodes, self.embedding_dim)
        # print(f"attention_output reshaped: {attention_output.shape}")
        
        # print(f"Q reshaped: {Q.shape}, K reshaped: {K.shape}, V reshaped: {V.shape}")
        if attention_mask is not None:
            # Mask to block some computations, set mask=0 to -inf
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_output = self.fc_out(attention_output)  # Map back to embedding_dim

        # Apply Layer Normalization after attention
        attention_output = self.attention_norm(attention_output)

        # Encoder
        encoded_output = self.encoder(attention_output)  # (num_nodes, 64)

        # Apply Layer Normalization after encoder
        encoded_output = self.encoder_norm(encoded_output)

        # Q-value prediction
        q_values = self.fc_q(encoded_output)  # (num_nodes, num_actions)

        return q_values[current_node_id]  # Return Q-values for the current node



