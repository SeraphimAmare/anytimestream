import glob, json, os, sys
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
held_out = ("carroll-alice.txt", "austen-persuasion.txt")
files = sorted(glob.glob(os.path.join(src, "*.txt")))
train = [f for f in files if os.path.basename(f) not in held_out]
read = lambda f: open(f, encoding="latin-1").read()
texts = [read(f) for f in train]
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=4096, special_tokens=["[UNK]", "<eos>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="<eos>")
fast.save_pretrained(os.path.join(out, "tokenizer"))
torch.save(torch.tensor(fast("\n".join(texts)).input_ids, dtype=torch.int32), os.path.join(out, "train_ids.pt"))
for name in held_out:
    ids = torch.tensor(fast(read(os.path.join(src, name))).input_ids, dtype=torch.int32)
    torch.save(ids, os.path.join(out, name.replace(".txt", "_ids.pt")))
    print(name, ids.numel(), "tokens")
print("train tokens:", torch.load(os.path.join(out, "train_ids.pt")).numel(), "| books:", len(train), "| vocab:", len(fast))
