# AFT-MindSpore
Unofficial MindSpore implementation of **Attention Free Transformer**'s layers by [Zhai](https://twitter.com/zhaisf?lang=en), et al. [[abs](https://openreview.net/forum?id=pW--cu2FCHY), [pdf](https://arxiv.org/pdf/2105.14103.pdf)] from Apple Inc.

The implementation is referred to [aft-pytorch](https://github.com/rish-16/aft-pytorch) and [annotated deep learning paper implementations](https://nn.labml.ai/transformers/aft/index.html).

<img src="https://github.com/rish-16/aft-pytorch/raw/main/pic.png" width=650>

## Usage
You can import the **AFT-Full**, **AFT-Local** or **AFT-Simple** layer (as described in the paper) from the `src` like so:

### `AFTFull`

```python
import mindspore.numpy as mnp
from src import AFTFull

layer = AFTFull(
    d_model=512,
    seq_len=20
)

# a batch of sequences with 10 timesteps of length 512 each
x = mnp.rand(10, 32, 512)
y = layer(x, x, x) # [10, 32, 512]
```

### `AFTSimple`

```python
import mindspore.numpy as mnp
from src import AFTSimple

layer = AFTSimple(d_model=512)

# a batch of sequences with 10 timesteps of length 512 each
x = mnp.rand(10, 32, 512)
y = layer(x, x, x) # [10, 32, 512]
```

### `AFTLocal`
```python
import mindspore.numpy as mnp
from src import AFTLocal

layer = AFTLocal(
    d_model=512,
    seq_len=20,
    local_window_size=10
)

# a batch of sequences with 10 timesteps of length 512 each
x = mnp.rand(10, 32, 512)
y = layer(x, x, x) # [10, 32, 512]
```