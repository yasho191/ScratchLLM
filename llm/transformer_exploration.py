# add all  your Encoder and Decoder code here
import torch.nn as nn
import torch.nn.functional as F
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, block_size, masked, dropout):
        super().__init__()
        # input = (batch_size, time_step, embedding_size)
        # Each Head -> (batch_size, time_step, embedding_size) -> (batch_size, time_step, embedding_size//num_heads)
        # Example: 2 Heads | Input: (1, 8, 200)
        # Each Head returns: (1, 8, 100) -> Concat on last dim -> (1, 8, 200)
        
        hidden_dim = embedding_size // num_heads
        slope = 2**(-8/num_heads)
        self.attention_heads = nn.ModuleList([AlibiAttentionHead(embedding_size, hidden_dim, block_size, masked, slope**(i+1)) for i in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim*num_heads, embedding_size)
        
    def forward(self, x):
        attention_head_output = [head(x) for head in self.attention_heads]
        self_attention = torch.cat([head[0] for head in attention_head_output], dim=-1)
        x = self.linear(self_attention)
        x = self.dropout(x)
        return x, [head[1] for head in attention_head_output]
        

class AlibiAttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, block_size, masked, slope):
        super().__init__()  
        self.masked = masked
        self.slope = slope
        self.block_size = block_size
        self.q = nn.Linear(embedding_size, head_size, bias=False)
        self.k = nn.Linear(embedding_size, head_size, bias=False)
        self.v = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def get_bias_mask(self, batch_size, block_size):
        matrix = torch.zeros(batch_size, block_size, block_size)
        
        # Fill the lower triangular part with the Alibi pattern
        for i in range(1, block_size):
            matrix[:, i, :i] = torch.arange(-i, 0)
        
        matrix = matrix.to(device)
        return matrix
    
    def forward(self, x):
        batch, time_step, embedding = x.size()
        bias = self.get_bias_mask(batch, self.block_size)
        
        Q = self.q(x) # (batch, time_step, head_size)
        K = self.k(x) # (batch, time_step, head_size)
        V = self.v(x) # (batch, time_step, head_size)
        
        QK = Q@K.transpose(-2, -1) # (batch, time_step, time_step)
        QK *= K.size()[-1]**-0.5
        
        if self.masked:
            bias = bias.masked_fill(self.tril[:time_step, :time_step] == 0, float('-inf'))
            QK = QK + (bias*self.slope)
        else:
            # create a symmetric matrix as the lower triangular mask won't work for encoder based models
            bias = bias + bias.transpose(-2, -1)
            QK = QK + (bias*self.slope)
        
        QK = F.softmax(QK, dim=-1)
        QKV = QK@V # (batch, time_step, time_step) x (batch, time_step, head_size) 
        return QKV, QK
        
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_size, dropout, encoder=True):
        super().__init__()
        if encoder:
            self.feed_forward = nn.Sequential(
                nn.Linear(embedding_size, 4 * embedding_size),
                nn.ReLU(),
                nn.Linear(4 * embedding_size, embedding_size),
                nn.Dropout(dropout),
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(embedding_size, 100),
                nn.ReLU(),
                nn.Linear(100, embedding_size),
                nn.Dropout(dropout),
            )
    def forward(self, x):
        x = self.feed_forward(x)
        return x
    
class Block(nn.Module):
    def __init__(self, embedding_size, num_heads, block_size, masked, dropout, encoder=True):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(embedding_size, num_heads, block_size, masked, dropout)
        self.feed_forward = FeedForwardNetwork(embedding_size, dropout, encoder)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, inputs):
        x, _ = inputs
        mulihead_attention, attention_maps = self.multihead_attention(self.layer_norm1(x))
        x = x + mulihead_attention
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, attention_maps

        
class AlibiEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, block_size, num_heads, dropout, classifier_hidden_dim, classifier_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads, block_size, False, dropout) for i in range(num_layers)])
        
        self.linear = nn.Linear(embedding_size, classifier_hidden_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(classifier_hidden_dim, classifier_dim, bias=False)
    
    def forward(self, x):
        
        x = self.embedding(x)
        
        # Send Dummy Input for 1st Block -> Attention Maps
        block_out, attention_maps = self.blocks((x, torch.ones(1, 2, 2).to(device)))
        
        block_out = torch.mean(block_out, dim=1)
        out = self.relu(self.linear(block_out))
        out = self.classifier(out)
        
        return out, attention_maps
        
class AlibiDecoder(nn.Module):        
    def __init__(self, vocab_size, embedding_size, num_layers, block_size, num_heads, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads, block_size, True, dropout, False) for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    
    def forward(self, x, y=None):
        batch, time_steps = x.size()
        
        x = self.embedding(x)
        
        # Send Dummy Input for 1st Block -> Attention Maps
        block_out, attention_maps = self.blocks((x, torch.ones(1, 2, 2).to(device)))
        block_out = self.layer_norm(block_out)
        out = self.linear(block_out)
        
        if y is not None:
            batch, time_steps, embedding = out.shape
            out = out.view(batch*time_steps, embedding)
            y = y.view(batch*time_steps)

            loss = F.cross_entropy(out, y)
        else:
            loss = None
        
        return loss, attention_maps