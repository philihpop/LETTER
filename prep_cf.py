"""
Convert LETTER format to RecBole format for SASRec training
"""
import json
import os
from pathlib import Path

def convert_to_recbole_format(inter_file, output_dir, dataset_name):
    """
    Convert interaction data to RecBole's .inter format
    
    Your format:
    - User ID -> List of item IDs (chronological order)
    
    RecBole format (.inter file):
    - user_id:token  item_id:token  timestamp:float
    """
    # Load interactions
    with open(inter_file, 'r') as f:
        interactions = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}.inter')
    
    # Write in RecBole format
    with open(output_file, 'w') as f:
        # Write header
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        
        # Write interactions
        for user_id, item_list in interactions.items():
            for timestamp, item_id in enumerate(item_list):
                # Use position as timestamp (sequential order)
                f.write(f'{user_id}\t{item_id}\t{float(timestamp)}\n')
    
    print(f"✓ Created RecBole interaction file: {output_file}")
    print(f"  - Total users: {len(interactions)}")
    total_interactions = sum(len(items) for items in interactions.values())
    print(f"  - Total interactions: {total_interactions}")
    
    return output_file


def create_recbole_config(dataset_name, output_dir, embedding_size=32):
    """
    Create RecBole configuration file
    """
    config = {
        # Dataset config
        'data_path': output_dir,
        'dataset': dataset_name,
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        
        # Model config
        'MODEL_TYPE': 'SequentialRecommender',
        'model': 'SASRec',
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
        
        # Training config
        'epochs': 200,
        'train_batch_size': 256,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,
        'stopping_step': 10,
        
        # Other
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'MAX_ITEM_LIST_LENGTH': 50,
        'gpu_id': '0',
        'reproducibility': True,
        'seed': 2020,
        
        # Evaluation
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
    }
    
    config_file = os.path.join(output_dir, f'{dataset_name}_sasrec.yaml')
    
    # Write YAML config
    with open(config_file, 'w') as f:
        f.write('# RecBole Configuration for SASRec\n\n')
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'{key}: "{value}"\n')
            elif isinstance(value, dict):
                f.write(f'{key}:\n')
                for k, v in value.items():
                    f.write(f'  {k}: {v}\n')
            elif isinstance(value, list):
                f.write(f'{key}: {value}\n')
            else:
                f.write(f'{key}: {value}\n')
    
    print(f"Created RecBole config file: {config_file}")
    return config_file


def main():
    # Configuration
    DATASET_NAME = 'Instruments'
    INPUT_INTER_FILE = '/mnt/user-data/uploads/Instruments_inter.json'
    OUTPUT_DIR = './recbole_data'
    EMBEDDING_SIZE = 32  # Match the cf_emb dimension in LETTER
    
    print("="*60)
    print("Converting LETTER format to RecBole format")
    print("="*60)
    print()
    
    # Convert interaction file
    inter_file = convert_to_recbole_format(
        INPUT_INTER_FILE, 
        OUTPUT_DIR, 
        DATASET_NAME
    )
    
    print()
    
    # Create config file
    config_file = create_recbole_config(
        DATASET_NAME, 
        OUTPUT_DIR, 
        EMBEDDING_SIZE
    )
    
    print()
    print("="*60)
    print("Next steps:")
    print("="*60)
    print("1. Install RecBole:")
    print("   pip install recbole")
    print()
    print("2. Run training script (see train_sasrec.py)")
    print()


if __name__ == '__main__':
    main()