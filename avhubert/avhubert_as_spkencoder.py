from avhubert.avhubert import AVHubertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from avhubert.resnet import ResEncoder


class AVHubertSpeakerEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.temperature = 1.0
        self.cfg = cfg
    
    def load_pretrained(self, path, device):
        avhubert_weight = torch.load(path, map_location=device)['model']
        self.avhubert_model.load_state_dict(avhubert_weight)
        
    def avhubert_grad(self, enable:bool):
        for _, param in self.avhubert_model.named_parameters():
            param.requires_grad = enable
    
    def update_steps(self, current_step, total_steps):
        pass
    
    def compute_loss(self, emb_i, emb_j):
        raise NotImplementedError()
        

class SimCLRSpeakerEncoder(AVHubertSpeakerEncoder):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.avhubert_model = AVHubertModel(cfg=cfg)
        self.speaker_parameter = nn.Parameter(torch.empty(1, 1, self.avhubert_model.embed))
    
    def extract_feature(self, video):
        # source should only include video
        source = {
            "video": video,  #  [B, 1, T, H, W]
            "audio": None,
            }
        encoder_out, feature, mask = self.avhubert_model.extract_finetune_with_feature(
            source,
            speaker_params=self.speaker_parameter,
            )  # (bs, vidlen+1, 768)
        speaker_embedding = encoder_out[:, 0, :].squeeze(1)
        encoder_feature = encoder_out[:, 1:, :]
        return {
            "speaker_embedding": speaker_embedding,  # (bs, 768)
            "encoder_feature": encoder_feature,  # (bs, vidlen, 768)
        }
    
    def forward(self, video, detailed=False):
        # unsupervised InfoNCE loss
        B, THalf, _, H, W, _ = video.shape  # [B, T//2, 2, H, W, 1]
        video = video.permute(0, 2, 1, 3, 4, 5)  # [B, 2, T//2, H, W, 1]
        video = video.reshape(B*2, THalf, H, W, 1)  # [B*2, T//2, H, W, 1]
        video = video.permute(0, 4, 1, 2, 3)  # [B*2, 1, T//2, H, W]
        embeddings = self.extract_feature(video)["speaker_embedding"]  # [B*2, C]
        embeddings = embeddings.reshape(B, 2, -1)  # [B, 2, C]
        embeddings = embeddings.permute(1, 0, 2)  # [2, B, C]
        loss = self.compute_loss(embeddings[0], embeddings[1])
        if detailed:
            return {
                "embeddings":embeddings,
                "loss":loss,
            }
        else:
            return loss
    
    def compute_loss(self, emb_i, emb_j):
        """
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            
        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size, )).to(emb_i.device) \
                .scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
                
            loss_ij = -torch.log(numerator / denominator)
                
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss
        

class AVHuBERTVideoResEncoder(nn.Module):
    def __init__(self, dim, cfg) -> None:
        super().__init__()
        self.resnet = ResEncoder(
            relu_type=cfg.resnet_relu_type, 
            weights=cfg.resnet_weights
        )
        self.linear = nn.Linear(
            self.resnet.backend_out,
            dim,
        )
    
    def forward(self, x):
        res_out = self.resnet(x)  # [B, -1, Tnew]
        res_out = res_out.mean(dim=-1)  # [B, C]
        speaker_embedding = self.linear(res_out)
        return speaker_embedding  # [B, dim]


class MoCoSpeakerEncoder(AVHubertSpeakerEncoder):
    """
    https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """
    def __init__(self, cfg, dim=128, K=3000, m=0.999, T=0.07) -> None:
        """
        *Description copied from MoCo*
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(cfg)
        self.m = m
        self.K = K
        self.temperature = T
        
        self.encoder_q = AVHuBERTVideoResEncoder(dim, cfg)
        self.encoder_k = AVHuBERTVideoResEncoder(dim, cfg)
        self._overwrite_q2k()
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss_func = nn.CrossEntropyLoss()
        
    def _overwrite_q2k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"K={self.K}, {batch_size=}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def load_pretrained(self, path, device):
        avhubert_weight = torch.load(path, map_location=device)['model']
        resnet_weight = {'.'.join(k.split('.')[2:]):v for k,v in avhubert_weight.items() if 'feature_extractor_video.resnet' in k}
        self.encoder_q.resnet.load_state_dict(resnet_weight)
        self._overwrite_q2k()
    
    def forward(self, video, detailed=False):
        # unsupervised InfoNCE loss
        video = video.permute(2, 0, 5, 1, 3, 4)  # [2, B, 1, T//2, H, W]
        embeddings_q = self.encoder_q(video[0])  # [B, C]
        embeddings_k = self.encoder_k(video[1])  # [B, C]
        if detailed:
            return {
                "embeddings":
                    torch.stack((embeddings_q, embeddings_k), dim=0),
            }
        else:
            loss = self.compute_loss(embeddings_q, embeddings_k)
            return loss
    
    def compute_loss(self, q, k):
        """
        Input:
            q: speaker embedding for a batch of query audio clips
            k: speaker embedding for a batch of key audio clips
        Output:
            logits, targets
        """

        # compute query features
        q = nn.functional.normalize(q, dim=1)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            k, idx_unshuffle = self._batch_shuffle_ddp(k)  # keys: NxC

            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = self.loss_func(logits, labels)
        return loss

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    
    def avhubert_grad(self, enable: bool):
        # MoCo has two encoders and should handle gradients itself
        pass
        
