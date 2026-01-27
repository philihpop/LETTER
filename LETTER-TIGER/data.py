import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]

        return dict(input_ids=d["inters"], labels=d["item"])


class SeqRecDatasetGlobalSplit(BaseDataset):
    """
    Sequential Recommendation Dataset with Global Time Splitting.
    Splits data based on global timestamps rather than per-user leave-one-out.
    
    This version is PROMPT-FREE and designed for direct semantic ID generation
    (like TIGER/LETTER), not instruction-following.
    """
    
    def __init__(self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)
        
        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        
        # Load data
        self._load_data()
        self._remap_items()
        
        # Global time split
        self._create_global_time_splits()
        
        # Process data based on mode
        if self.mode == 'train':
            self.inter_data = self._process_train_data_global()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data_global()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data_global()
        else:
            raise NotImplementedError
    
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items
    
    def _create_global_time_splits(self):
        """
        Create global time-based splits (80% train, 10% valid, 10% test).
        For each user, split their chronological interactions proportionally.
        This simulates temporal progression while handling user-sorted data.
        """
        print("Creating global time splits...")

        # Apply 80/10/10 split per user
        self.train_inters = defaultdict(list)
        self.valid_inters = defaultdict(list)
        self.test_inters = defaultdict(list)

        total_train = 0
        total_valid = 0
        total_test = 0

        for uid, items in self.remapped_inters.items():
            n_items = len(items)

            # Calculate split points for this user
            train_end = max(1, int(n_items * 0.8))  # At least 1 for train
            valid_end = max(train_end + 1, int(n_items * 0.9))  # At least 1 for valid

            # Split this user's interactions
            self.train_inters[uid] = items[:train_end]
            self.valid_inters[uid] = items[train_end:valid_end]
            self.test_inters[uid] = items[valid_end:]

            total_train += len(self.train_inters[uid])
            total_valid += len(self.valid_inters[uid])
            total_test += len(self.test_inters[uid])

        print(f"Total interactions: {total_train + total_valid + total_test}")
        print(f"Train: {total_train} interactions ({total_train/(total_train+total_valid+total_test)*100:.1f}%)")
        print(f"Valid: {total_valid} interactions ({total_valid/(total_train+total_valid+total_test)*100:.1f}%)")
        print(f"Test: {total_test} interactions ({total_test/(total_train+total_valid+total_test)*100:.1f}%)")

        print(f"Train users: {len(self.train_inters)}")
        print(f"Valid users with targets: {len([u for u in self.valid_inters if self.valid_inters[u]])}")
        print(f"Test users with targets: {len([u for u in self.test_inters if self.test_inters[u]])}")

        # Check overlap
        test_users = set(self.test_inters.keys())
        train_users = set(self.train_inters.keys())
        users_with_history = test_users & train_users
        print(f"Test users who also appear in train: {len(users_with_history)} / {len(test_users)}")
    
    def _process_train_data_global(self):
        """Process training data from global time split."""
        inter_data = []
        
        for uid in self.train_inters:
            items = self.train_inters[uid]
            if len(items) < 2:  # Need at least one history item
                continue
            
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]
                history = items[:i]
                
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)
        
        return inter_data
    
    def _process_valid_data_global(self):
        """Process validation data from global time split."""
        inter_data = []
        
        for uid in self.valid_inters:
            target_items = self.valid_inters[uid]
            if not target_items:
                continue
            
            # Get history up to validation point
            history = self.train_inters.get(uid, [])
            
            for target_item in target_items:
                if len(history) == 0:  # Need at least some history
                    continue
                
                one_data = dict()
                one_data["item"] = target_item
                
                hist = history.copy()
                if self.max_his_len > 0:
                    hist = hist[-self.max_his_len:]
                if self.add_prefix:
                    hist = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(hist)]
                
                one_data["inters"] = self.his_sep.join(hist)
                inter_data.append(one_data)
                
                # Add current item to history for next validation item
                history.append(target_item)
        
        return inter_data
    
    def _process_test_data_global(self):
        """Process test data from global time split."""
        inter_data = []
        skipped_no_history = 0
        skipped_no_targets = 0
        
        # Get all unique users across all splits
        all_test_users = set(self.test_inters.keys())
        
        print(f"Processing test data for {len(all_test_users)} users...")
        
        for uid in all_test_users:
            target_items = self.test_inters.get(uid, [])
            if not target_items:
                skipped_no_targets += 1
                continue
            
            # Get history up to test point (train + valid)
            history = self.train_inters.get(uid, []) + self.valid_inters.get(uid, [])
            
            # DEBUG: Check if we have users with test items but no history
            if len(history) == 0:
                skipped_no_history += 1
                # Still process with empty history to avoid empty dataset
                # This allows cold-start evaluation
                pass
            
            for target_item in target_items:
                one_data = dict()
                one_data["item"] = target_item
                
                hist = history.copy()
                if self.max_his_len > 0 and len(hist) > 0:
                    hist = hist[-self.max_his_len:]
                if self.add_prefix and len(hist) > 0:
                    hist = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(hist)]
                
                # Handle empty history case
                if len(hist) == 0:
                    one_data["inters"] = ""  # Empty history for cold-start
                else:
                    one_data["inters"] = self.his_sep.join(hist)
                
                inter_data.append(one_data)
                
                # Add current item to history for next test item
                history.append(target_item)
        
        print(f"Test data stats:")
        print(f"  Total samples: {len(inter_data)}")
        print(f"  Skipped (no history): {skipped_no_history}")
        print(f"  Skipped (no targets): {skipped_no_targets}")
        
        if len(inter_data) == 0:
            raise ValueError(f"No test samples generated! Check your data splits. "
                           f"Test users: {len(all_test_users)}, "
                           f"Train users: {len(self.train_inters)}, "
                           f"Valid users: {len(self.valid_inters)}")
        
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, min(self.sample_num, len(inter_data)), replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
            print(f"  Sampled: {len(inter_data)}")
        
        return inter_data
    
    def set_prompt(self, prompt_id):
        """Dummy method for compatibility - prompts not used in this version."""
        self.prompt_id = prompt_id
    
    def __len__(self):
        return len(self.inter_data)
    
    def __getitem__(self, index):
        """
        Returns dict with semantic ID sequences for direct generation.
        Format: {"input_ids": history_sequence, "labels": target_item}
        
        This is the TIGER/LETTER paradigm - no prompts, just semantic IDs.
        """
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])