import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import TiktokenTokenizer, SimpleTokenizer

class GPT2Dataset(Dataset):
    def __init__(self, text: str,
                 tokenizer: TiktokenTokenizer, 
                 max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize text
        tokenized_ids = tokenizer.encode(text)

        # Use sliding window to chunk text into sequences of max_length
        # Ensure at least one sequence even if text is shorter than max_length
        end_idx = max(0, len(tokenized_ids) - max_length) + 1
        for i in range(0, end_idx, stride):
            # Clamp slices to actual token length to avoid unnecessary padding
            length = min(i + max_length, len(tokenized_ids))
            
            input_chunk = tokenized_ids[i:length]
            target_chunk = tokenized_ids[i+1:length+1]
            
            # Pad to max_length if needed (use GPT2 EOS token as padding)
            if len(input_chunk) < max_length:
                # Unfortunately, GPT2 tokenizer does not have a padding token, so we use the EOS token
                # This is not ideal, but it is the simplest solution for now
                # However, this means we need to mask out the padding tokens when computing the loss
                pad_token = 50256  # GPT2 EOS token
                input_chunk = input_chunk + [pad_token] * (max_length - len(input_chunk))
                target_chunk = target_chunk + [pad_token] * (max_length - len(target_chunk))
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class GPT2DataLoader:
    def __init__(self, text: str, 
                 tokenizer: TiktokenTokenizer, 
                 batch_size: int, max_length: int, stride: int, shuffle: bool, 
                 drop_last: bool, 
                 num_workers: int) -> None:
        # Dataset
        self.dataset = GPT2Dataset(
            text=text,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride
        )
        
        # Dataloader with pinned memory for faster CPU-GPU transfers
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last, 
            num_workers=num_workers,
            pin_memory=True  # Faster CPU-to-GPU transfers
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def next_batch(self):
        return next(iter(self.dataloader))
    

if __name__ == "__main__":
    # Load raw text
    st = SimpleTokenizer()
    raw_text = st._load_training_text()
    # Tokenizer
    tokenizer = TiktokenTokenizer()
    # DataLoader
    dataloader = GPT2DataLoader(
        text=raw_text,
        tokenizer=tokenizer,
        batch_size=16,
        max_length=4,
        stride=4,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    inputs, targets = dataloader.next_batch()
    print(f"Inputs:\n {inputs}")
    print(f"Targets:\n {targets}")