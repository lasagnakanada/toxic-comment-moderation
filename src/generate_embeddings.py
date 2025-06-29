import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm


CSV_PATH = 'data/your_file.csv'           
EMBEDDINGS_PATH = 'data/embeddings.pt'     
LABELS_PATH = 'data/labels.npy'           
BERT_MODEL = 'distilbert-base-uncased'
BATCH_SIZE = 64
MAX_LENGTH = 128


df = pd.read_csv(CSV_PATH)

texts = df['comment_text'].astype(str).tolist()
labels = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values


tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = DistilBertModel.from_pretrained(BERT_MODEL).to(device)
bert.eval()

all_embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    tokens = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
        return_token_type_ids=False
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = bert(**tokens)
        cls_emb = outputs.last_hidden_state[:,0,:].cpu()
        all_embeddings.append(cls_emb)

X = torch.cat(all_embeddings, dim=0)
torch.save(X, EMBEDDINGS_PATH)
np.save(LABELS_PATH, labels)

print(f"Saved embeddings to {EMBEDDINGS_PATH}")
print(f"Saved labels to {LABELS_PATH}")
