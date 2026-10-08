import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
from pandas.tests.indexes.multi.test_duplicates import idx_dup

class BigramLanguageModel(nn.Module):

   def __init__(self,vocab_size):

        super().__init__()
        self.voc_size= vocab_size
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)


   def forward(self,idx,target):
    # idx = [B,S]
    # target = [B,S]
       B, S = idx.shape
       # ============================================================
        # STEP 1: Embed tokens (this IS the bigram lookup)
        # ============================================================
        # idx:      [B, S]       (integers in [0, V))
        # embedding weight: [V, V]
        # logits:   [B, S, V]
        #
        # For each token i in the input, we look up row i of the
        # embedding table, which gives us the logits for predicting
        # the NEXT token. This is exactly the bigram probability table!

       logits= self.token_embedding_table(idx) # [B,S,V]
       
       # cross entropy little tricky 
       # it expects [N,C] and [N]

       # our tensors are 
         #   logits:   [B, S, V]
         #   targets:  [B, S]

                  # So we need to flatten B and S together:
            #   logits:   [B*S, V]
            #   targets:  [B*S]

       logits=logits.view(B*S,self.voc_size) 
       target=target.view(B*S)
       loss= F.cross_entropy(logits,target)


       return logits,loss 

  def generate(self, idx, max_new_tokens):

     # starting context idx [B,S]
     

     for _ in range(max_new_tokens):

         logits,loss=self(idx) # (B,S,V)

         logits = logits[:,-1,:]

         probs=F.softmax(logits,dim=-1)

         idx_next= torch.multinomial(probs,num_samples=1)

         idx= torch.cat([idx.idx_next],dim=1)

     return idx # [B,s+ max_new_tpkens]




         
     
     
    

