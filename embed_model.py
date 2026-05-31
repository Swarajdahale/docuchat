# embed_model.py
# PyTorch custom sentence embedding model
# Interview talking point: "I built a lightweight embedding model using PyTorch
# that converts text chunks into vector representations for similarity search"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SentenceEmbedder(nn.Module):
    """
    A PyTorch model that wraps a HuggingFace transformer
    and adds a projection layer to control embedding dimensions.
    
    Why PyTorch here?
    - We add a trainable nn.Linear projection layer on top of the base model
    - This makes it a real PyTorch model (not just a wrapper)
    - In production, you could fine-tune this on domain-specific data
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dim=256):
        super(SentenceEmbedder, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # PyTorch layer: projects embeddings to desired dimension
        # This is what makes it a "custom" PyTorch model
        hidden_size = self.transformer.config.hidden_size  # usually 384
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.output_dim = output_dim
    
    def mean_pooling(self, model_output, attention_mask):
        """Average token embeddings weighted by attention mask"""
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Input:  list of text strings
        Output: tensor of shape (batch_size, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Forward pass through transformer
        with torch.no_grad():  # no gradient needed for inference
            transformer_output = self.transformer(**encoded)
        
        # Pool token embeddings -> sentence embedding
        sentence_embeddings = self.mean_pooling(
            transformer_output, encoded["attention_mask"]
        )
        
        # Project to output dimension (the custom PyTorch part!)
        projected = self.projection(sentence_embeddings)
        
        # L2 normalize for cosine similarity search
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        
        return normalized
    
    def embed_query(self, text: str) -> list[float]:
        """LangChain-compatible interface for single query embedding"""
        embedding = self.forward([text])
        return embedding[0].tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain-compatible interface for batch document embedding"""
        embeddings = self.forward(texts)
        return embeddings.tolist()


# --- Quick test ---
if __name__ == "__main__":
    print("Loading PyTorch embedding model...")
    model = SentenceEmbedder()
    model.eval()
    
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather is nice today."
    ]
    
    embeddings = model.forward(test_texts)
    print(f"Input:  {len(test_texts)} sentences")
    print(f"Output: tensor shape {embeddings.shape}")  # (3, 256)
    
    # Cosine similarity (should be high for sentences 1&2, low for 1&3)
    sim_12 = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )
    sim_13 = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0)
    )
    print(f"\nSimilarity (ML vs Deep Learning): {sim_12.item():.3f}  ← should be HIGH")
    print(f"Similarity (ML vs Weather):        {sim_13.item():.3f}  ← should be LOW")
    print("\n✅ PyTorch embedding model working!")