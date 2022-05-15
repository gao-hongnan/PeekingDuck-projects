## The Custom Nodes

This section shows how to use a custom trained model to perform inference with integration with the PeekingDuck framework. It assumes that you are already familiar with the process of creating custom nodes, covered in my other [tutorial](./exercise_counter.md).


### configs/model/melanoma_classifier.yml

We first see what content is inside the config file.

```yaml title="melanoma_classifier.yml" linenums="1"
input: ["img"]
output: ["pred_label", "pred_score", "gradcam_image"]

weights_parent_dir: pytorch_models
weights: {
    model_subdir: resnet50d,
    blob_file: resnet50d.zip,
    classes_file: melanoma_class_mapping.yml,
    model_file: {
        resnet34d: resnet34d.pt,
        resnet50d: resnet50d.pt,
    }
}

model_params: {
        model_name: resnet50d,
        out_features: 2,
        in_channels: 3,
        pretrained: false,
        use_meta: false
}

num_classes: 2

class_label_map: {
    0: "benign",
    1: "malignant",
}

model_type: resnet50d
input_size: 224
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
half: false
plot_gradcam: false
```

- `#!python [Lines 1-2]`: The input takes in the built-in PeekingDuck `img` input and outputs the `pred_label` and `pred_score`, the former being the predicted labels from `#!python [Lines 25-28]` and the latter, the corresponding probability of the predicted label. On top of that, we also output `gradcam_image` which is a heatmap overlayed on the input image.
- `#!python [Line 4]`: The weights of your trained model are stored in the `weights_parent_dir` directory.
- `#!python [Lines 5-13]`: The `weights` key corresponds to a dictionary:
    - `model_subdir`: The `model_subdir` is a sub-directory of `weights_parent_dir` directory. The name should be indicative of the model architecture.
    - `blob_file`: The `blob_file` is the name of the file that contains the trained model weights. For example, if you stored your trained weights `resnet50d.zip` on Google Cloud Storage, then the `blob_file` necessarily should be named `resnet50d.zip` in order for `downloader` to download the weights from the cloud.
    - `classes_file`: The `classes_file` is the name of the file that contains the mapping between the class labels and their corresponding class names. One can also define it in the current config file as well: `#!python [Lines 25-28]`.
    - `model_file`: The `model_file` is a dictionary that maps the model architecture to the trained model weights name. For example, `resnet50d.pt` is the weight extracted from `resnet50d.zip`. This mapping is needed for us to define the `model_path` in `detector.py`. See `model_path = (self.model_dir / self.config["weights"]["model_file"][model_type])` in `detector.py`.
- `#!python [Lines 15-21]`: The `model_params` key corresponds to a dictionary that will be unpacked in `resnet_files/model.py`.
- `#!python [Line 23]`: The `num_classes` indicates the number of unique classes in the dataset.
- `#!python [Lines 25-28]`: The `class_label_map` key corresponds to a dictionary that maps the class labels to their corresponding class names. This should be exactly the same as `classes_file` in `#!python [Line 8]`.
- `#!python [Line 30]`: The `model_type` indicates the model architecture. **This may be redundant since it is defined in the `model_params` key.**
- `#!python [Line 31-33]`: These 3 lines are the **transform** parameters, used to preprocess the input image. 
- `#!python [Line 34]`: The `half` parameter indicates whether the model is trained on a half precision or not.
- `#!python [Line 35]`: The `plot_gradcam` parameter indicates whether the gradcam heatmap should be plotted or not.


### model/melanoma_classifier.py

The highlighted lines below basically does the following:

- Convert input image from BGR to RGB;
- Make a prediction on the input image using the trained model;
- Make a heatmap of the input image using the gradcam algorithm;
- Return the predicted label and the corresponding probability alongside the heatmap.

    ```python title="model.melanoma_classifier.py" linenums="1" hl_lines="22 24-25 27-29 31-34"
    class Node(AbstractNode):
        """Initializes and uses a ResNet to predict if an image frame is a melanoma or not."""

        def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
            super().__init__(config, node_path=__name__, **kwargs)

            self.plot_gradcam: bool

            self.model = resnet_model.ResnetModel(self.config)
            self.input_shape = (self.input_size, self.input_size)

        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Reads the image input and returns the predicted class label and probability.

            Args:
                inputs (dict): Dictionary with key "img".

            Returns:
                outputs (dict): Dictionary with keys "pred_label", "pred_score" and "gradcam_image".
            """

            img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)

            reshaped_original_image = cv2.resize(img, self.input_shape)
            prediction_dict = self.model.predict(img)

            gradcam_image = self.model.show_gradcam(
                reshaped_original_image, self.plot_gradcam
            )

            return {
                **prediction_dict,
                "gradcam_image": gradcam_image,
            }
    ```

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

### model/resnets/resnet_model.py

This script contains the `ResnetModel` class which is used to validate configuration, loads the `ResNet` model through `Detector`, and performs inference with the method `predict`.

### model/melanoma_classifier.py

This is the custom node.



### pipeline_config.yml

```yaml title="pipeline_config.yml" linenums="1"
nodes:
- custom_nodes.input.visual:
   source: melanoma_data/test
- custom_nodes.model.melanoma_classifier
- output.csv_writer:
   stats_to_track: ["filename", "pred_label", "pred_score"]
   file_path: "./stores/artifacts/resnet_melanoma.csv"
   logging_interval: 0
- draw.legend:
   show: ["filename", "pred_label", "pred_score"]
- custom_nodes.output.screen
```