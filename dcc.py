# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import torch
from torch.nn.functional import normalize

device = torch.device('cuda')


class DCC(torch.nn.Module):
    def __init__(
            self,
            backbone_q,
            backbone_k,
            margin_loss: Callable,
            batch_size: int,
            embedding_size: int,
            fp16: bool = False,
            queue_size=65536,
            momentum=0.999,
            sample_num=2 + 1,
    ):
        super(DCC, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.fp16 = fp16

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

        self.queue_size = queue_size
        print(self.queue_size)
        self.momentum = momentum
        self.sample_num = sample_num
        self.backbone_q = backbone_q
        self.backbone_k = backbone_k

        for param_q, param_k in zip(
                self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("weight_queue", torch.randn([embedding_size, queue_size]))
        self.weight_queue = normalize(self.weight_queue, dim=0)

        self.register_buffer("label_queue", torch.randn([1, queue_size]))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(
            self,
            img_q,
            img_k,
            labels: torch.Tensor,
            cfg
    ):
        local_embeddings = self.backbone_q(img_q.to(device))
        norm_embeddings = normalize(local_embeddings, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            local_weight = self.get_weight(img_k, cfg, norm_embeddings)
            norm_weight = normalize(local_weight, dim=1)

        l_pos = torch.einsum("nc,nc->n", [norm_embeddings, norm_weight]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [norm_embeddings, self.weight_queue.clone().detach()])

        labels = labels.reshape(self.batch_size, 1)
        label_diff = labels - self.label_queue

        mask = (label_diff == 0).float()
        l_neg = l_neg * (1 - mask) + (-1e9 * mask)
        logits = torch.cat([l_pos, l_neg], dim=1)

        self._dequeue_and_enqueue(norm_weight, labels)

        zero_labels = torch.zeros([logits.shape[0]], dtype=torch.long).cuda()
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, zero_labels)
        loss = self.cross_entropy(logits, zero_labels)

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def get_weight(self, img_k, cfg, norm_embeddings):
        with ((torch.no_grad())):
            im_weight = torch.cat(img_k, dim=0)
            w = self.backbone_k(im_weight.to(device))

            weights = torch.split(w, split_size_or_sections=cfg.batch_size, dim=0)

            # compute attention weight
            cos_sims = torch.zeros([cfg.sample_num - 1, cfg.batch_size])
            i = 0
            for weight in weights:
                norm_weight = normalize(weight)
                cos_sim = torch.nn.functional.cosine_similarity(norm_embeddings, norm_weight, dim=1)
                cos_sims[i] = cos_sim
                i += 1
            attention = torch.nn.functional.softmax(cos_sims.transpose_(1, 0), dim=1)
            attention = attention.cuda()
            dcc_weight = 0
            for k in range(i):
                dcc_weight += attention[:, k].reshape([cfg.batch_size, 1]) * weights[k]

            return dcc_weight

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.weight_queue[:, ptr:ptr + batch_size] = keys.T
        self.label_queue[:, ptr:ptr + batch_size] = labels.T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
