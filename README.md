# MAGNUM
A Modality-Agnostic Multimodal Modular Architecture

## Installation

1. Clone the repository
2. Install the requirements file with `pip install -r requirements.txt`

## Basic usage

All the relevant code is stored into the **models** module.

```python
import torch.nn as nn
from models.low_level_module import RoBERTaPromptBottleneck, ViTPromptBottleneck, TabularMapper
from models.wrapper import BottomLevelModule, TopLevelModule, Magnum
```

##### Config
```python
d_model = 256
n_prompts = 8
knn_k = 3
d_hidden = d_model
gate_input_type = "same"
gate_output_type = "softmax-scalar"
```

##### Bottom Level (Low-Level Module)
```python
tabular_model = TabularMapper(d_model=d_model, n_num_vars=None, n_cat_vars=None, num_cat_vars_classes=None)
tabular_mapper = nn.Linear(d_model, d_model)

language_model = RoBERTaPromptBottleneck(n_prompts)
language_mapper = nn.Linear(language_model.d_model, d_model)

vision_model = ViTPromptBottleneck(n_prompts)
vision_mapper = nn.Linear(vision_model.d_model, d_model)

bottom_level_module = BottomLevelModule(
    d_model=d_model,
    tabular_model=tabular_model, tabular_mapper=tabular_mapper,
    language_model=language_model, language_mapper=language_mapper,
    vision_model=vision_model, vision_mapper=vision_mapper
)
```

##### Top Level (Mid-Level and High-Level Modules)
```python
top_level_module = TopLevelModule(
    d_model=d_model,
    hidden_size=d_hidden,
    gate_input_type=gate_input_type,
    gate_output_type=gate_output_type,
    k=knn_k,
    n_output_classes=None,
    modalities=["tabular", "language", "vision"]
)
```

##### Call Magnum Model
```python
magnum = Magnum(bottom_level_module, top_level_module)
```
