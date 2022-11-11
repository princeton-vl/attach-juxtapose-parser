"""
Self-attention layers and other utilities from self-attentive-parser
https://github.com/nikitakit/self-attentive-parser
"""

"""
MIT License

Copyright (c) 2017-2018 Nikita Kitaev
Copyright (c) 2017 Victor Huang
Copyright (c) 2017 Mitchell Stern

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional, Tuple, Any, Union


class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """

    def __init__(self, batch_idxs_arr: Union[ArrayLike, torch.Tensor]) -> None:
        if torch.is_tensor(batch_idxs_arr):  # type: ignore
            self.batch_idxs_np = batch_idxs_arr.cpu().numpy()  # type: ignore
            self.batch_idxs_torch = batch_idxs_arr
        else:
            self.batch_idxs_np = batch_idxs_arr
            self.batch_idxs_torch = torch.from_numpy(batch_idxs_arr)

        self.batch_size = int(1 + np.max(self.batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], self.batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(
            batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1]
        )[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

    @staticmethod
    def from_lens(lens: List[int]) -> "BatchIndices":
        batch_idxs = np.zeros(np.sum(lens), dtype=np.int64)
        base = 0
        for i, l in enumerate(lens):
            batch_idxs[base : base + l] = i
            base += l
        return BatchIndices(batch_idxs)

    def inflate(self, flattened: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        padded = flattened.new_full(
            (len(self.seq_lens_np), self.max_len, flattened.size(-1)),
            fill_value=fill_value,
        )
        base = 0
        for i, l in enumerate(self.seq_lens_np):
            padded[i, :l] = flattened[base : base + l]
            base += l
        return padded


class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(  # type: ignore
        cls: Any,
        ctx: Any,
        input: torch.Tensor,
        batch_idxs: BatchIndices,
        p: float = 0.5,
        train: bool = False,
        inplace: bool = False,
    ) -> torch.Tensor:
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None


class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor, batch_idxs: BatchIndices) -> torch.Tensor:
        return FeatureDropoutFunction.apply(  # type: ignore
            input, batch_idxs, self.p, self.training, self.inplace
        )


class LayerNormalization(nn.Module):
    def __init__(self, d_hid: int, eps: float = 1e-3, affine: bool = True) -> None:
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        # NOTE(nikita): the t2t code does the following instead, with eps=1e-6
        # However, I currently have no reason to believe that this difference in
        # implementation matters.
        # mu = torch.mean(z, keepdim=True, dim=-1)
        # variance = torch.mean((z - mu.expand_as(z))**2, keepdim=True, dim=-1)
        # ln_out = (z - mu.expand_as(z)) * torch.rsqrt(variance + self.eps).expand_as(z)
        # ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, attention_dropout: float = 0.1) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model**0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), (
                "Attention mask shape {} mismatch "
                "with Attention logit tensor shape "
                "{}.".format(attn_mask.size(), attn.size())
            )

            attn.data.masked_fill_(attn_mask, -float("inf"))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        residual_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        d_positional: Optional[int] = None,
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        if d_positional is None:
            self.partitioned = False

            self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

            nn.init.xavier_normal_(self.w_qs)
            nn.init.xavier_normal_(self.w_ks)
            nn.init.xavier_normal_(self.w_vs)

        else:
            self.partitioned = True

            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_content, d_k // 2)
            )
            self.w_ks1 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_content, d_k // 2)
            )
            self.w_vs1 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_content, d_v // 2)
            )

            self.w_qs2 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_positional, d_k // 2)
            )
            self.w_ks2 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_positional, d_k // 2)
            )
            self.w_vs2 = nn.Parameter(
                torch.FloatTensor(n_head, self.d_positional, d_v // 2)
            )

            nn.init.xavier_normal_(self.w_qs1)
            nn.init.xavier_normal_(self.w_ks1)
            nn.init.xavier_normal_(self.w_vs1)

            nn.init.xavier_normal_(self.w_qs2)
            nn.init.xavier_normal_(self.w_ks2)
            nn.init.xavier_normal_(self.w_vs2)

        self.attention = ScaledDotProductAttention(
            d_model, attention_dropout=attention_dropout
        )
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head * (d_v // 2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head * (d_v // 2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(
        self, inp: torch.Tensor, qk_inp: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_inp_repeated = inp.repeat(self.n_head, 1).view(
            self.n_head, -1, inp.size(-1)
        )  # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(
                self.n_head, -1, qk_inp.size(-1)
            )

        if not self.partitioned:
            # n_head x len_inp x d_k
            q_s = torch.bmm(qk_inp_repeated, self.w_qs)
            # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks)
            # n_head x len_inp x d_v
            v_s = torch.bmm(v_inp_repeated, self.w_vs)
        else:
            q_s = torch.cat(
                [
                    torch.bmm(qk_inp_repeated[:, :, : self.d_content], self.w_qs1),
                    torch.bmm(qk_inp_repeated[:, :, self.d_content :], self.w_qs2),
                ],
                -1,
            )
            k_s = torch.cat(
                [
                    torch.bmm(qk_inp_repeated[:, :, : self.d_content], self.w_ks1),
                    torch.bmm(qk_inp_repeated[:, :, self.d_content :], self.w_ks2),
                ],
                -1,
            )
            v_s = torch.cat(
                [
                    torch.bmm(v_inp_repeated[:, :, : self.d_content], self.w_vs1),
                    torch.bmm(v_inp_repeated[:, :, self.d_content :], self.w_vs2),
                ],
                -1,
            )
        return q_s, k_s, v_s

    def pad_and_rearrange(
        self,
        q_s: torch.Tensor,
        k_s: torch.Tensor,
        v_s: torch.Tensor,
        batch_idxs: BatchIndices,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.bool)

        for i, (start, end) in enumerate(
            zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])
        ):
            q_padded[:, i, : end - start, :] = q_s[:, start:end, :]
            k_padded[:, i, : end - start, :] = k_s[:, start:end, :]
            v_padded[:, i, : end - start, :] = v_s[:, start:end, :]
            invalid_mask[i, : end - start].fill_(False)

        return (
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1)
            .expand(mb_size, len_padded, len_padded)
            .repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
        )

    def combine_v(self, outputs: torch.Tensor) -> torch.Tensor:
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = (
                torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)
            )

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            outputs1 = (
                torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            )
            outputs2 = (
                torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            )
            outputs = torch.cat(
                [
                    self.proj1(outputs1),
                    self.proj2(outputs2),
                ],
                -1,
            )

        return outputs

    def forward(
        self,
        inp: torch.Tensor,
        batch_idxs: BatchIndices,
        qk_inp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(
            q_s, k_s, v_s, batch_idxs
        )

        outputs_padded, attns_padded = self.attention(
            q_padded,
            k_padded,
            v_padded,
            attn_mask=attn_mask,
        )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded


class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_hid: int,
        d_ff: int,
        d_positional: int,
        relu_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff // 2)
        self.w_1p = nn.Linear(d_positional, d_ff // 2)
        self.w_2c = nn.Linear(d_ff // 2, self.d_content)
        self.w_2p = nn.Linear(d_ff // 2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, batch_idxs: BatchIndices) -> torch.Tensor:
        residual = x
        xc = x[:, : self.d_content]
        xp = x[:, self.d_content :]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)  # type: ignore
