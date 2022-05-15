### model/resnets/resnet_files/downloader.py

This script is the same as the one in [PeekingDuck's weights utils](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/weights_utils/downloader.py). 

The only difference is I changed the global variable `BASE_URL`:

```python title="Original vs New"
BASE_URL = "https://storage.googleapis.com/peekingduck/models"          # Original
BASE_URL = "https://storage.googleapis.com/reighns/peekingduck/models"  # New
```

so that I can download the model weights from my own bucket.

### model/resnets/resnet_files/model.py

This script contains the **ResNet** model structure with backbone and head. Currently, the model is hardcoded to only take in models from the  [famous PyTorch `timm` library](https://github.com/rwightman/pytorch-image-models). As this was ported over from my personal project, the name of the class is called `CustomNeuralNet` instead of a more indicate name such as `ResNetModel`.


### model/resnets/resnet_files/detector.py

This script contains the `Detector` class which is used to predict melanoma. 