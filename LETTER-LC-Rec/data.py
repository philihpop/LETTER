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
from prompt import sft_prompt, all_prompt
import numpy as np


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

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
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

    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = tokenizer("Response:")["input_ids"][1:]

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
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
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
                one_data["inters"] = self.his_sep.join(history)
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
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        # print(index, idx)

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        # print({"input": input, "output": output})

        return dict(input_ids=input, labels=output)


class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = "".join(self.indices[str(items[i])])
                one_data["title"] = self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inters = ["".join(self.indices[str(j)]) for j in history]
                inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]


                if self.add_prefix:
                    inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                    inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-2])]["description"]


            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)


        return dict(input_ids=input, labels=output)


class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()
        self.feat_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)


    def _process_data(self):

        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data


    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class ItemSearchDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        # load data
        self._load_data()
        self.search_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)


    def _process_data(self):

        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == 'train':
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == 'test':
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["inters"] = self.his_sep.join(inters)

            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == 'test':
            return len(self.search_data)
        else:
            return len(self.search_data)


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)



class PreferenceObtainDataset(BaseDataset):

    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # load data
        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):

        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)





class SeqRecTestDataset(BaseDataset):

    def __init__(self, args, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompt = all_prompt["seqrec"][self.prompt_id]

        # load data
        self._load_data()
        self._remap_items()

        self.inter_data = self._process_test_data()

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)

            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

        self.prompt = all_prompt["seqrec"][self.prompt_id]

    def __len__(self):

        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")

        return input, response

    def __getitem__(self, index):

        d = self.inter_data[index]
        input, target = self._get_text_data(d, self.prompt)

        return dict(input_ids=input, labels=target)

class SeqRecDatasetGlobalSplit(BaseDataset):
    """
    Sequential Recommendation Dataset with Global Time Splitting.
    Splits data based on global timestamps rather than per-user leave-one-out.
    """
    
    def __init__(self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)
        
        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.prompts = all_prompt["seqrec"]
        
        # Load data
        self._load_data()
        self._remap_items()
        
        # Global time split
        self._create_global_time_splits()
        
        # Process data based on mode
        if self.mode == 'train':
            self.inter_data = self._process_train_data_global()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data_global()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data_global()
        else:
            raise NotImplementedError
    
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items
    
    def _create_global_time_splits(self):
        """
        Create global time-based splits (80% train, 10% valid, 10% test).
        For each user, determine which interactions belong to which split.
        """
        print("Creating global time splits...")
        
        # Collect all interactions with timestamps
        # Since we already have chronologically ordered sequences, we use positions as proxy
        all_interactions = []
        for uid, items in self.remapped_inters.items():
            for idx, item in enumerate(items):
                # Use (uid, position) as timestamp proxy since items are already ordered
                all_interactions.append((uid, idx, item))
        
        # Calculate split points
        total_interactions = len(all_interactions)
        train_end = int(total_interactions * 0.8)
        valid_end = int(total_interactions * 0.9)
        
        print(f"Total interactions: {total_interactions}")
        print(f"Train: 0-{train_end} ({train_end} interactions)")
        print(f"Valid: {train_end}-{valid_end} ({valid_end - train_end} interactions)")
        print(f"Test: {valid_end}-{total_interactions} ({total_interactions - valid_end} interactions)")
        
        # Assign interactions to splits
        self.train_inters = defaultdict(list)
        self.valid_inters = defaultdict(list)
        self.test_inters = defaultdict(list)
        
        for global_idx, (uid, local_idx, item) in enumerate(all_interactions):
            if global_idx < train_end:
                self.train_inters[uid].append(item)
            elif global_idx < valid_end:
                self.valid_inters[uid].append(item)
            else:
                self.test_inters[uid].append(item)
        
        print(f"Train users: {len(self.train_inters)}")
        print(f"Valid users with targets: {len([u for u in self.valid_inters if self.valid_inters[u]])}")
        print(f"Test users with targets: {len([u for u in self.test_inters if self.test_inters[u]])}")
    
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
        
        for uid in self.test_inters:
            target_items = self.test_inters[uid]
            if not target_items:
                continue
            
            # Get history up to test point (train + valid)
            history = self.train_inters.get(uid, []) + self.valid_inters.get(uid, [])
            
            for target_item in target_items:
                if len(history) == 0:
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
                
                # Add current item to history for next test item
                history.append(target_item)
        
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, min(self.sample_num, len(inter_data)), replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        
        return inter_data
    
    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})
    
    def _get_text_data(self, data, prompt):
        from prompt import sft_prompt
        
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)
        
        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)
        
        if self.mode == 'test':
            return input, response
        
        return input, output
    
    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]
        
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        
        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id
        
        prompt = self.prompts[prompt_id]
        input, output = self._get_text_data(d, prompt)
        
        return dict(input_ids=input, labels=output)