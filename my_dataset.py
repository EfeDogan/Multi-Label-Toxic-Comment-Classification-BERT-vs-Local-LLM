import torch
from torch.utils.data import Dataset

class ToxicCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, target_columns=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_columns = target_columns if target_columns else ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        comment = str(self.df.loc[index, 'cleaned_text'])
        
        # HuggingFace Tokenizer işlemleri
        inputs = self.tokenizer(
            comment,
            None,
            add_special_tokens=True, # [CLS] ve [SEP] ekler
            max_length=self.max_len,
            padding='max_length',     # Sabit 256 boyuta tamamlar
            truncation=True,          # 256'dan uzunsa keser (biz zaten sildik ama güvenli kalmalı)
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'       # PyTorch Tensor formatında döner
        )
        
        # Etiketleri float tensor olarak alıyoruz (BCEWithLogitsLoss float bekler)
        labels_raw = self.df.loc[index, self.target_columns].values.astype(float)
        labels = torch.tensor(labels_raw, dtype=torch.float)
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels
        }