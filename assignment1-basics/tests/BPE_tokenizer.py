import os
import pickle
from typing import Iterable, Iterator
from collections import Counter,defaultdict,OrderedDict
import regex as re

class BPETokenizer:
    def __init__(self,vocab,merges,special_tokens = None):
        
        #初始化词表和合并规则
        self.vocab = vocab #id -> Bytes
        self.vocab_inv = {v:k for k,v in vocab.items()} #Bytes -> id
        self.merges = merges #目前是一个list[tuple(Bytes,Bytes)]
        self.merges = {pair: idx for idx,pair in enumerate(merges)} #我们后面要取rank最小的进行合并
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.cache = OrderedDict()
        self.cache_size = 600
    @classmethod
    def from_files(cls, vocab_filepath,merges_filepath,special_tokens = None):
        #从序列化文件中加载词表和合并规则
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
            # merges = {pair: idx for idx,pair in enumerate(merges)}
        return cls(vocab, merges, special_tokens)

    
    def encode(self,text):
        from .adapters import split_keep_special
        #对单个文本进行BPE编码，返回token id列表
        #首先对文本进行预分词(此时假设text为"Hello <PAD> world!")
        parts = split_keep_special(text, self.special_tokens)
        #此时parts = [""Hello ","<PAD>"," world!"]
        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.vocab_inv[part.encode("utf-8")])
                continue
            for m in re.finditer(self.PAT, part):
                token = m.group(0).encode("utf-8")
                if token in self.cache:
                    token_ids.extend(self.cache[token])
                    self.cache.move_to_end(token)
                    continue
                #此时Token为b"Hello"
                symbols = [bytes([b]) for b in token]
                #此时symbols为[b"H",b"e",b"l",b"l",b"o"]

                #接下来进行BPE合并
                while True:
                    best_pair = None
                    for i  in range(len(symbols)-1):
                        pair = (symbols[i], symbols[i+1])
                        if pair in self.merges:
                            if best_pair is None or self.merges[pair] < self.merges[best_pair]:
                                best_pair = pair
                    if best_pair is None:
                        break
                    merged_token = best_pair
                    new_symbols = []
                    i = 0
                    while i < len(symbols):
                        if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == merged_token:
                            new_symbols.append(merged_token[0] + merged_token[1])
                            i += 2
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                    symbols = new_symbols
                #将合并后的symbols转换为token ids
                current_ids = [self.vocab_inv[symbol] for symbol in symbols]
                self.cache[token] = current_ids
                self.cache.move_to_end(token)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                token_ids.extend(current_ids)
        return token_ids
                  
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #对输入的流式文本的指定范围进行BPE编码，返回token id生成器
        for text in iterable:
            token_ids = self.encode(text)
            for tid in token_ids:
                yield tid

    def decode(self,token_ids):
        #将token id列表解码为文本,很简单，遍历一遍就行
        bytes_list = [self.vocab[tid] for tid in token_ids]
        text = b"".join(bytes_list).decode("utf-8",errors="replace")
        return text