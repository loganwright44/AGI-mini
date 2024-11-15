# AGI-mini
An attempt at a simple artificial general intelligence architecture (sub-super intelligence). See details on strategy and philosophy below.

# Model Architecture
## Conscious Sub-Model

**Size:** The bigger this model, the more sophisticated the short-term memory effects, hence increased likelihood of high-order abstract thought.

**Input:** `torch.cat(state_vector, conscious_out_vector): torch.Tensor`

**Output:** `conscious_out_vector: torch.Tensor`

|   Layer Component   |              Type                |
|---------------------|----------------------------------|
|      Embedding      |         torch.nn.Linear          |
| Transformer Encoder |    torch.nn.TransformerEncoder   |
|   Fully Connected   |         torch.nn.Linear          |

The role of this sub-model is (as shown in the input `torch.Tensor`) to encode information about both the input vector and it's own "train-of-thought" vector, if you will. Context continually updates and short-term memory is a natural consequence of the transformer encoder algorithm; context of old information loses relevance as time passes.

## Unconcious Sub-Model

**Size:** This sub-model is intentionally small.

**Input:** `state_vector: torch.Tensor`

**Output:** `unconcious_out_vector: torch.Tensor`

|   Layer Component   |              Type                |
|---------------------|----------------------------------|
|      Embedding      |         torch.nn.Linear          |
| Transformer Encoder |    torch.nn.TransformerEncoder   |
|   Fully Connected   |         torch.nn.Linear          |

The role of this model is one-fold: to handle extremes of the `state_vector` tensor by transforming them into something representing urgent actionables (i.e. pain, pleasure, etc.). This can be thought of as the reward algorithm of reinforcement learning, but is self-managed. This could potentially be replaced by a mathematical function, but I expect a reward system of this kind to naturally lead to curiosity within the model.

## Response Sub-Model

**Size:** This sub-model will behave best when given more parameters.

**Input:** `torch.cat(conscious_out_vector, unconscious_out_vector): torch.Tensor`

**Output:** `response_vector: torch.Tensor`

|   Layer Component   |              Type                |
|---------------------|----------------------------------|
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |
|   Fully Connected   |         torch.nn.Linear          |
| Activation Function |          torch.nn.ReLU           |

The role of these layers is to interpret the meaning of the transformer outputs and generate a response which minimizes punishment. Minimizing punishment is mathematically preferable since the value of the target converges.

# Resources
https://www.pytorch.com