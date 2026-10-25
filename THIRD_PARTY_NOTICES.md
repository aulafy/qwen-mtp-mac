# Third-party notices

## mlx-lm

Parts of `mtp.py` (the `GatedDeltaNet.__call__` replacement), `gdn_kernel.py` (a variant of
`mlx_lm/models/gated_delta.py`) and `gdn_fused.py` are adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm). The speculative decoding design (verifying
`[token, draft]` in one pass and rolling back the GatedDeltaNet state) follows the approach of
[ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990) by AirRunner; the code here is an
independent implementation.

    MIT License
    
    Copyright © 2023 Apple Inc.
    
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

## Model weights (not included)

This repository contains no model weights. At runtime it reads the `qwen3.5:9b-mlx` weights from your
local Ollama store and downloads only the multi-token-prediction head (`mtp.*` tensors) of
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) from Hugging Face. Qwen3.5 is released by the
Qwen team under the Apache License 2.0.

## Trademarks

Not affiliated with or endorsed by Alibaba/Qwen, Apple or Ollama. The server mimics the Ollama and OpenAI
HTTP APIs only for client compatibility.
