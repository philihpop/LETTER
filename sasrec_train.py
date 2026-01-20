"""
Train SASRec model and extract item embeddings for LETTER
"""
import torch
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import os


def train_sasrec(config_file=None, dataset='Instruments', embedding_size=32):
    """
    Train SASRec model using RecBole
    """
    # Configuration
    config_dict = {
        'model': 'SASRec',
        'dataset': dataset,
        'data_path': './recbole_data',
        'embedding_size': embedding_size,
        'hidden_size': embedding_size,
        'inner_size': 256,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'MAX_ITEM_LIST_LENGTH': 50,
        'epochs': 200,
        'train_batch_size': 256,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,
        'stopping_step': 10,
        'gpu_id': '0',
        'reproducibility': True,
        'seed': 2020,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        'metrics': ['Recall', 'NDCG', 'Hit'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
    }
    
    # Initialize
    config = Config(model='SASRec', dataset=dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    # Create dataset
    dataset = create_dataset(config)
    
    # Split dataset
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Create model
    model = SASRec(config, train_data.dataset).to(config['device'])
    
    # Create trainer
    trainer = Trainer(config, model)
    
    # Train
    print("\n" + "="*60)
    print("Starting SASRec training...")
    print("="*60 + "\n")
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    
    # Test
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation result: {best_valid_result}")
    print(f"Test result: {test_result}")
    
    return model, dataset, config


def extract_item_embeddings(model, dataset, output_file, embedding_size=32):
    """
    Extract item embeddings from trained SASRec model
    
    Args:
        model: Trained SASRec model
        dataset: RecBole dataset object
        output_file: Path to save embeddings (.pt file)
        embedding_size: Dimension of embeddings
    """
    print("\n" + "="*60)
    print("Extracting item embeddings...")
    print("="*60)
    
    # Get item embeddings
    # SASRec stores item embeddings in item_embedding
    item_embeddings = model.item_embedding.weight.detach().cpu()
    
    # The first item (index 0) is usually padding, so we might need to handle it
    num_items = dataset.item_num
    print(f"Total items in dataset: {num_items}")
    print(f"Embedding shape: {item_embeddings.shape}")
    print(f"Embedding dimension: {embedding_size}")
    
    # Save embeddings
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    torch.save(item_embeddings, output_file)
    
    print(f"\n Item embeddings saved to: {output_file}")
    print(f"  Shape: {item_embeddings.shape}")
    print(f"  Can be used as --cf_emb in LETTER training")
    
    return item_embeddings


def verify_embeddings(embedding_file):
    """
    Verify the saved embeddings
    """
    print("\n" + "="*60)
    print("Verifying saved embeddings...")
    print("="*60)
    
    embeddings = torch.load(embedding_file)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Check for NaN or Inf
    if torch.isnan(embeddings).any():
        print(" Warning: Found NaN values in embeddings!")
    if torch.isinf(embeddings).any():
        print(" Warning: Found Inf values in embeddings!")
    
    print(" Embeddings verified successfully")


def main():
    DATASET = 'Instruments'
    EMBEDDING_SIZE = 32
    OUTPUT_FILE = f'./RQ-VAE/ckpt/{DATASET}-{EMBEDDING_SIZE}d-sasrec.pt'
    
    # Train SASRec
    model, dataset, config = train_sasrec(
        dataset=DATASET,
        embedding_size=EMBEDDING_SIZE
    )
    
    # Extract embeddings
    embeddings = extract_item_embeddings(
        model, 
        dataset, 
        OUTPUT_FILE,
        EMBEDDING_SIZE
    )
    
    # Verify
    verify_embeddings(OUTPUT_FILE)



if __name__ == '__main__':
    main()