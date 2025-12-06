import torch
from torch import nn

if __name__ == "__main__":
    inputs = torch.tensor([[1, 2, 3, 4, 5]])
    max_position = 10
    vocab_size = 20
    output_dim = 8

    # Token embeddings
    token_embeddings = nn.Embedding(vocab_size, output_dim)
    print(f"Token embedding weight: {token_embeddings.weight}")
    print(f"Token embedding: {token_embeddings(inputs)}")

    # Position embeddings
    position_embeddings = nn.Embedding(max_position, output_dim)
    print(f"Position embedding weight: {position_embeddings.weight}")
    print(f"Position embedding: {position_embeddings(inputs)}")

    # Combined embeddings
    combined_embeddings = token_embeddings(inputs) + position_embeddings(inputs)
    print(f"Combined embeddings: {combined_embeddings}")
    