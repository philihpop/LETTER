import torch

# Load SASRec checkpoint
checkpoint_path = '/home/xshen/SASRec/python/vg_vg_32/SASRec.epoch=100.lr=0.001.layer=2.head=1.hidden=32.maxlen=200.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')

# Extract item embeddings INCLUDING padding at index 0
item_embeddings = state_dict['item_emb.weight']  # Shape: [25612, 32]

# DON'T remove index 0 - keep it as is
# Save with padding included
torch.save(item_embeddings, '../RQ-VAE/ckpt/videogames-32d-sasrec.pt')

print(f"Saved CF embeddings with shape: {item_embeddings.shape}")