from .adapters import split_keep_special
from .test_tokenizer import get_tokenizer_from_vocab_merges_path
from .BPE_tokenizer import BPETokenizer

def main():
    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="/root/project/CS336/assignment1-basics/tests/fixtures/gpt2_vocab.json", 
        merges_path="/root/project/CS336/assignment1-basics/tests/fixtures/gpt2_merges.txt", 
        special_tokens=special_tokens
    )
    encoded_ids = tokenizer.encode(text)
    print(f"Encoded token IDs: {encoded_ids}")
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    print(f"Tokenized string: {tokenized_string}")
if __name__ == "__main__":
    main()