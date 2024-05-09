import torch
import torch.nn.functional as F
import math

def get_embeddings(model, patch_list, studentBool,batch_size=4):
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 내가 추가함
    
    embedding_list=[]
    
    
    for cls in range(len(patch_list)):
        num_batches=math.ceil(patch_list[cls].shape[0]/batch_size)
        if patch_list[cls] is not None:
            cls_embedding_list = []
            # print('yo',patch_list[cls].shape[0],math.ceil(patch_list[cls].shape[0]/batch_size))
            for i in range(num_batches):
                start_idx=i*batch_size
                end_idx=min((i+1)*batch_size,patch_list[cls].shape[0])
                
                if end_idx - start_idx < 2: # 마지막 배치수가 2 이하이면 버리도록 drop_last=True 와 같은 기능 삽입
                    continue
                
                batch = patch_list[cls][start_idx:end_idx,:,:,:]
                if studentBool is True:
                    batch=batch.to(dev)
                    
                emb = model(batch)
                emb = emb.to('cpu')
                
                # print('emb',emb.shape)
                emb = F.normalize(emb,p=2,dim=1) # Projecting onto hypersphere of radius 1
                cls_embedding_list.append(emb)
            embedding_list.append(torch.cat(cls_embedding_list,dim=0))
        else:
            embedding_list.append(None)
    return embedding_list