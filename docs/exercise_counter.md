<div align="center">
<h1>PeekingDuck Exercise Counter</a></h1>
by Hongnan Gao
1st May, 2022
<br>
</div>


## Push-Up Counter Logic

The aim of this project is to build an exercise counter that can be used to detect push-ups, and more generic exercises in future iterations. Some content below are referenced from [PeekingDuck's Tutorial](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#recipe-2-keypoints-count-hand-waves)


The main model is **MoveNet**, which outputs seventeen keypoints for the person corresponding to the different body parts as documented here[^movenet_keypoints_id]. Each keypoint is a pair of $(x, y)$ coordinates, where $x$ and $y$ are real numbers ranging from $0.0$ to $1.0$ (using relative coordinates[^coordinate_systems]).

!!! note
    This is a **Minimum Viable Product (MVP)** and uses simple logic to detect push-ups. Consequently, there are a few somewhat rigid assumptions that will be made in the code.

We will detail the logic in the following sections.

[^movenet_keypoints_id]: [Keypoint IDs](https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html#whole-body-keypoint-ids)
[^coordinate_systems]: [Coordinate Systems](https://peekingduck.readthedocs.io/en/stable/tutorials/01_hello_cv.html#tutorial-coordinate-systems)

### Assumptions

1. The user's **left body parts** are visible through the webcam/video and in particular, the **left elbow, wrist and shoulder** are crucial for our push-up counter. 
   
    We can improve on this rigid requirement in the future by checking both the **left** and **right** body parts, and take the **side** with **higher keypoints confidence**.

2. There should be only $1$ person in the video. As we are using **MoveNet's** `singlepose_thunder` model, we need to impose the restriction that there should be only $1$ person in the video to avoid performance issues.
   
    We can improve on this rigid requirement in the future by incorporating multi-person logic. The `multipose_lightning` can detect up to $6$ people in the video.

3. If user leaves the video and come back, we assume that the user is the same person and continue counting.


### The Logic

> Insert image for visual. Now brief walkthrough of the logic first. When done annotate all lines of the code.

- Assume the video is 10 seconds long with a frame rate of 30 fps. This means we have 300 frames (images) in the video.
- In each frame $i$, we check if the user is in the video by checking if the user's keypoints are visible, if not, we skip the frame.
    - $KP_{i}$ is the keypoints of the user in the frame $i$.
    - $KPC_{i}$ is the keypoints confidence of the user in the frame $i$.
    - $t$ is the threshold for the keypoints confidence. Anything with confidence lower than $t$ is discarded.
    - $LE_{i}$ is the left elbow keypoint of the user in the frame $i$.
    - $LS_{i}$ is the left shoulder keypoint of the user in the frame $i$.
    - $LW_{i}$ is the left wrist keypoint of the user in the frame $i$.
    - $LEA_{i}$ is the left elbow angle of the user in the frame $i$.
- In each frame $i$, we check the following logic:
    - Initalize $LE_{i}$, $LS_{i}$, $LW_{i}$ as `None`.
    - Check if $LE_{i}$, $LS_{i}$, $LW_{i}$ is above the threshold $t$. 
      - If yes, assign $LE_{i}$, $LS_{i}$, $LW_{i}$ the value of the keypoints.
      - If no, the variable will be `None`.
    - If all of $LE_{i}$, $LS_{i}$, $LW_{i}$ is not `None` (i.e. these keypoints passed the confidence threshold), then we can calculate the angle formed by the left elbow, wrist and shoulder.
      - $LEA_{i}$ is the angle formed by the left elbow, wrist and shoulder.



## Custom Node General Workflow

### Step 1. Initialize PeekingDuck Template 

After setting up from the [Workflows](./workflows.md) section, we can start using the PeekingDuck interface.

We initialize a new PeekingDuck project using the following commands:

!!! note "Terminal Session"
    ```bash title="Initializing PeekingDuck" linenums="1"
    mkdir custom_hn_exercise_counter
    cd custom_hn_exercise_counter
    peekingduck init
    ```

- Line 1: Create a new directory named `custom_hn_exercise_counter`.
- Line 2: Change to the newly created directory.
- Line 3: Initialize the PeekingDuck project in the current directory with default file/folder below.
    - `pipeline_config.yml`: This file[^pipeline_config] contains the pipeline configuration.
    - `src`: Folder for the custom nodes which we will create later.
  
The `custom_hn_exercise_counter` directory currently looks like this:

```tree title="Directory Tree of custom_hn_exercise_counter"
custom_hn_exercise_counter/
├── pipeline_config.yml
└── src/
```

[^pipeline_config]: You can read more [here](https://peekingduck.readthedocs.io/en/stable/tutorials/01_hello_cv.html).

---

### Step 2. Use Pipeline Recipe to Create Custom Nodes

After going through the tutorial on creating [custom nodes](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html), I settled for the [Pipeline Recipe method](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#pipeline-recipe). Here is how we can do it:

1. **`pipeline_config.yml`**: We first populate the `pipeline_config.yml` file with the configurations:

    ??? example "Show/Hide content for pipeline_config.yml"
        ```yaml title="pipeline_config.yml" linenums="1"
        nodes:
        - custom_nodes.input.visual:
            source: https://storage.googleapis.com/reighns/peekingduck/videos/push_ups.mp4
        - model.yolo:
            model_type: "v4tiny" # "v4tiny"
            iou_threshold: 0.1
            score_threshold: 0.1
            detect_ids: ["person"] # [0]
            num_classes: 1
        - model.movenet:
            model_type: "singlepose_thunder"
            resolution: {height: 256, width: 256}
            bbox_score_threshold: 0.05
            keypoint_score_threshold: 0.05
        - dabble.fps
        - custom_nodes.dabble.exercise_counter:
            exercise_name: "push_ups"
        - draw.poses
        - draw.legend:
            show: ["fps"]
        - output.csv_writer:
            stats_to_track: ["keypoints", "bboxes", "bbox_labels", "num_push_ups",
                            "frame_count", "body_direction", "elbow_angle",
                            "shoulder_keypoint", "elbow_keypoint", "wrist_keypoint", "filename"]
            file_path: "./store/artifacts/push_ups_output_movenet.csv"
            logging_interval: 0
        - output.screen
        ```

    In particular, we defined two **custom nodes** `custom_nodes.dabble.exercise_counter` and `custom_nodes.input.visual` which are not from the default PeekingDuck nodes. 

    !!! tip
        Notice that I added some configurations for the **default nodes**. For example, I specify that I want to use `v4tiny` model for the `model.yolo` node with a $0.1$ threshold for both iou and bounding box confidence score. These configurations will then be passed to the `config` parameter of the `model.yolo` node.

2. We then create the custom nodes using the following command:

    !!! note "Terminal Session"
        ```bash title="Creating Custom Nodes"
        peekingduck create-node --config_path pipeline_config.yml
        ```

    This will create all the nodes listed in the `pipeline_config.yml` file. If one decides to add more custom nodes, we can simply add them to the `pipeline_config.yml` file and run the command again.

    ```tree title="Directory Tree of custom_hn_exercise_counter"
    custom_hn_exercise_counter/
    ├── pipeline_config.yml
    └── src/
        └── custom_nodes/
            ├── configs/
            |    ├── dabble/
            |    |   └── exercise_counter.yml
            |    └── input/
            |        └── visual.yml
            ├── dabble/
            |    └── exercise_counter.py
            └── input/
                └── visual.py  
    ```

    `custom_hn_exercise_counter` now contains **five files** that we need to modify in order to implement our custom push-up counter.

    - `pipeline_config.yml`
    - `src/custom_nodes/dabble/exercise_counter.py`
    - `src/custom_nodes/configs/dabble/exercise_counter.yml`
    - `src/custom_nodes/input/visual.py`
    - `src/custom_nodes/configs/input/visual.py`

    We will go into details in the next [section](exercise_counter.md#step-3-deep-dive-into-the-custom-nodes).

3. We will also create an additional folder `stores` in the same level as `src`. For now, we will add a folder `artifacts` in the `stores` folder, this is where we will store the output files and model artifacts.

    ```tree title="Directory Tree of custom_hn_exercise_counter" 
    custom_hn_exercise_counter/
    ├── src/
    ├── stores/
    │   └── artifacts/
    └── pipeline_config.yml
    ```

---

### Step 3. Deep Dive into the Custom Nodes

After [Step 2](exercise_counter.md#step-2-use-pipeline-recipe-to-create-custom-nodes), three folders `config`, `dabble` and `input` will be populated in `src`. The `config` folder holds the **configurations** for the custom nodes while the `dabble` and `input` folders holds the **code** for the custom nodes. 
   
!!! info
    Something worth noting is that other **default nodes** that are in **PeekingDuck** will not be included in the `config` folder. For example, the `model.yolo` node is not included in the `config` folder because it is a **default node**. This is because the configurations for the default nodes are already included in the `pipeline_config.yml` file and will be **instantiated** when `peekingduck run` is called.

#### **input/visual.py**

!!! info
    This is with reference to **PeekingDuck** version `1.2.0`. I've noticed if you define `source` to be an **URL**, the `filename` is not overwritten by the `source` filename and maintains the default `video.mp4`. I am unsure if this is an expected behaviour for sources coming from **URLs** since the [documentation](https://peekingduck.readthedocs.io/en/stable/nodes/input.visual.html#module-input.visual) did not explicitly mention about the case whereby the `source` is an **URL**.
    
    I went to the latest developing version and noticed that there was a **[PR](https://github.com/aimakerspace/PeekingDuck/pull/646)** to fix **fix: 'input.visual' filename was not set if source is a single image**. However, it does not resolve the case whereby the `source` is an **URL**.

[Standing on the shoulder of giants](https://github.com/aimakerspace/PeekingDuck/pull/646), I quickly modified a few lines to suit my purpose as I want to save the `filename` parameter in my output csv file later. Therefore, the code inside this file is mostly the same except for the highlighted lines below.

```python title="input.visual.py: _determine_source_type()" linenums="1" hl_lines="12 17 26"
def _determine_source_type(self) -> None:
    """
    Determine which one of the following types is self.source:
        - directory of files
        - file
        - url : http / rtsp
        - webcam
    If input source is a directory of files,
    then node will have specific methods to handle it.
    Otherwise opencv can deal with all non-directory sources.
    """
    path = Path(self.source)
    if isinstance(self.source, int):
        self._source_type = SourceType.WEBCAM
    elif str(self.source).startswith(("http://", "https://", "rtsp://")):
        self._source_type = SourceType.URL
        self._file_name = path.name
    else:
        # either directory or file
        if not path.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist")
        if path.is_dir():
            self._source_type = SourceType.DIRECTORY
        else:
            self._source_type = SourceType.FILE
            self._file_name = path.name
```

**Explaination**: In the `run` method, we note that the `output` dict is called by `outputs = self._get_next_frame()` and `self._file_name` is therefore crucial to output the correct filename of the source. Since `source` is set as `self.source`, we use `pathlib.Path` to convert it to a `pathlib.Path` object and call the [`name`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.name) property to get the filename.

??? example "Show/Hide content for _get_next_frame()"
    ```python
    def _get_next_frame(self) -> Dict[str, Any]:
        """Read next frame from current input file/source"""
        self.file_end = True
        outputs = {
            "img": None,
            "filename": self._file_name if self._file_name else self.filename,
            "pipeline_end": True,
            "saved_video_fps": self._fps
            if self._fps > 0
            else self.saved_video_fps,
        }
        if self.videocap:
            success, img = self.videocap.read_frame()
            if success:
                self.file_end = False
                if self.do_resize:
                    img = resize_image(
                        img, self.resize["width"], self.resize["height"]
                    )
                outputs["img"] = img
                outputs["pipeline_end"] = False
            else:
                self.logger.debug("No video frames available for processing.")
        return outputs  
    ```

#### **input/visual.yml**

The `config` file for the `input.visual` node is therefore the same as the one in the default [source code](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/input/visual.yml).

??? example "Show/Hide content for input.visual.yml"
    ```yaml title="input.visual.yml"
    input: ["none"]
    output: ["img", "filename", "pipeline_end", "saved_video_fps"]

    filename: video.mp4
    frames_log_freq: 100
    mirror_image: False
    resize: {
                do_resizing: False,
                width: 1280,
                height: 720
            }
    saved_video_fps: 10
    source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
    threading: False
    buffering: False
    ```

#### **dabble/exercise_counter.yml**

> This section focuses on **`src/custom_nodes/configs/dabble/exercise_counter.yml`**. Recall that this `yml` file is created by the `peekingduck create-node --config_path pipeline_config.yml` command.

The **default configuration** is as follows:

```yaml title="Default config for dabble/exercise_counter.yml" linenums="1"
input: ["bboxes", "bbox_labels"]    # (1)
output: ["obj_attrs", "custom_key"] # (2)

threshold: 0.5                      # (3)
``` 

1.  Mandatory configs. The default configurations receive bounding boxes and their respective labels as input. Replace with other data types as required. List of built-in data types for PeekingDuck can be found in [here](https://peekingduck.readthedocs.io/en/stable/glossary.html).
2.  Output `obj_attrs` for visualization with `draw.tag` node and `custom_key` for use with other custom nodes. Replace as required.
3.  Optional configs depending on node. We will go through that later.

We need to edit the file according to our own needs. Recall that we are implementing a push-up (exercise) counter using the custom `dabble.exercise_counter` node. 

Since we are chaining **Yolo** and **MoveNet**, this node should therefore take in `img`, `bboxes`, `bbox_scores`, `keypoints`, and `keypoint_scores` as inputs from the pipeline and outputs the `output` dict with keys such as `frame_count`, `num_push_ups`, `body_direction`, `elbow_angle`, `shoulder_keypoint`, `elbow_keypoint`, and `wrist_keypoint`.

We also want to pass some optional configs to the `dabble.exercise_counter` node. For example, `exercise_name` is an optional config that user can specify. For now, the default and the only available option is `push_ups`.

Consequently, the newly updated `dabble.exercise_counter.yml` file should contain the following:

``` yaml title="dabble.exercise_counter.yml" linenums="1"
input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]     # (1)
output: ["frame_count", "num_push_ups", "body_direction", "elbow_angle",
        "shoulder_keypoint", "elbow_keypoint", "wrist_keypoint"]            # (2)

# Optional configs
keypoint_threshold: 0.3                                                     # (3)
exercise_name: "push_ups"                                                   # (4)                                   
```

1.  The `inputs` propagated from previous nodes.
    - `img`: Input image
    - `bboxes`: Bounding boxes
    - `bbox_scores`: Bounding box scores
    - `keypoints`: Keypoints
    - `keypoint_scores`: Keypoint scores
2.  The `outputs` of the current node.
    - `frame_count`: Incremented every time a new frame is processed.
    - `num_push_ups`: The number of push-ups detected in the current frame.
    - `body_direction`: The direction of the body in the current frame.
    - `elbow_angle`: The angle between the wrist, elbow and the shoulder in the current frame.
    - `shoulder_keypoint`: The keypoint of the shoulder in the current frame.
    - `elbow_keypoint`: The keypoint of the elbow in the current frame.
    - `wrist_keypoint`: The keypoint of the wrist in the current frame.
3.  This is an optional configuration parameter that will be initialized if defined. Here I defined a `keypoint_threshold` parameter.
4.  This is an optional configuration parameter that will be initialized if defined. Here I defined a `exercise_name` parameter.

Let us walk through the [mandatory inputs](exercise_counter.md#mandatory-inputs) and [optional configs](exercise_counter.md#optional-configs) of `dabble.exercise_counter.yml` in the next two sections. One should also refer back by clicking the ➕ beside each line of code in the above file for annotations.

##### Mandatory Inputs

!!! note
    The `input` in the `dabble.exercise_counter.yml` file means that previous nodes' `output` will be passed as input to the current node, and necessarily, the keys `img`, `bboxes` etc, must exist prior to the current node.

    You can click on the "add" button in the code block above to see more info on the individual `input` and `output`.


!!! warning
    Since both `MoveNet` and `Yolo` have the same `output` key `bboxes`, chaining `PoseNet/Movenet` after `Yolo` will cause the common output `bboxes` to be overwritten by the latter.


##### Optional Configs

Recall the optional config defined in `dabble.exercise_counter.yml`:

``` yaml title="optional config" linenums="1"
# Optional configs
keypoint_threshold: 0.3                                                
exercise_name: "push_ups"                                                                                    
```

These will be passed to the `Node(AbstractNode)` constructor as **attributes**.

```python title="abstract node class" linenums="1"
class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
```

!!! warning
    The config arguments are optional for users to define, but once defined, it should have values in the node's corresponding `yml` file so that it can be instantiated as an attributes in `Node`.

!!! info
    For example, when you pass an additional parameter `exercise_name` in `exercise_counter.yml`, this parameter will be instantiated in the `dabble.exercise_counter` `Node` as an attribute `self.exercise_name`. 

    A quick check in the source code `peekingduck.pipeline.nodes.node.py` shows that it will set all the keys defined in the `.yml` file.
    ```python
    # sets class attributes
    for key in self.config:
        setattr(self, key, self.config[key])
    ```

!!! tip
    Note you can also override the default `exercise_name` in `pipeline_config.yml`'s `custom_nodes.dabble.exercise_counter` portion but only after you defined it in `dabble.exercise_counter.yml`.

!!! note
    As an aside, the other optional argument passed is `keypoint_threshold` which is used to determine the threshold for the keypoints. Technically it behaves the same as the `keypoint_score_threshold` in `model.movenet`'s Configs.


#### **dabble/exercise_counter.py**

> `run` method:

- `#!python [Lines 1-6]:` We unpack the `inputs` dict into individual variables: `img`, `bboxes`, `bbox_scores`, `keypoints`, `keypoint_scores` and `filename`. Note that these are the outputs of the previous nodes.
- `#!python [Lines 7]:` This is the image width and height.
- `#!python [Lines 8]:` We add $1$ to the `frame_count` every time a new frame is processed.
- `#!python [Lines 9-11]:` We need to make sure that the `bboxes` and `keypoints` are not empty because we need to do some calculations on them. 
- `#!python [Lines 12-13]:` Based on earlier assumptions there is only $1$ person in the video, so `the_bbox` and `the_bbox_score` are the first element in the `bboxes` and `bbox_scores` respectively. **Note in particular that if `bbox_scores` is empty then we assign `bbox_score` to $0$. We did not check it via `is_bbox_and_keypoints_empty` because `bbox_scores` come from a different model.**
- `#!python [Lines 14-24]:` We first map `the_bbox` to [image coordinates](https://peekingduck.readthedocs.io/en/stable/tutorials/01_hello_cv.html#tutorial-coordinate-systems) and create a string `score_str` that contains the score of the bounding box from **MoveNet**. Finally, we draw the bounding box score on the image.
- `#!python [Lines 25-26]:` As there is only one person in the video, we get `the_keypoints` and `the_keypoint_scores` from `keypoints[0]` and `keypoint_scores[0]` respectively.
- `#!python [Lines 27-29]:` We call the `self.count_push_ups` method to count the number of push-ups, taking in `img`, `img_size`, `the_keypoints` and `the_keypoint_scores`. 
- `#!python [Lines 30-39]:` Return dictionary with keys to be passed in to the next node.

#### output/csv_writer.yml

This is a default node and we will make use of the [`output.csv_writer`](https://peekingduck.readthedocs.io/en/stable/nodes/output.csv_writer.html#module-output.csv_writer) node to write the results from `exercise_counter` to a CSV file. 

A quick check at the default settings from `peekingduck.configs.output.csv_writer.yml` yields:

```yaml title="default output.csv_writer.yml" linenums="1"
input: ["all"]
output: ["none"]

stats_to_track: ["keypoints", "bboxes", "bbox_labels"]
file_path: "PeekingDuck/data/stats.csv"
logging_interval: 1 # in terms of seconds between each log
```

We are fine with `input` and `output` as they are but need to modify the configurations from `#!python [lines 4-6]`.

Since the configurations' key names are not changed, we can directly overwrite them in `pipeline_config.yml` as follows:

```yaml title="pipeline_config.yml" linenums="1"
- output.csv_writer:
    stats_to_track: ["keypoints", "bboxes", "bbox_labels",
                     "num_push_ups", "frame_count", 
                     "body_direction", "elbow_angle",
                     "shoulder_keypoint", "elbow_keypoint",
                     "wrist_keypoint", "filename"]              # (1)
    file_path: "./stores/artifacts/push_ups_output_movenet.csv" # (2)
    logging_interval: 0                                         # (3)
```

1.  The main aim is to keep track of keypoints information per frame.
2.  The `file_path` is set to `./store/artifacts/push_ups_output_movenet.csv`. This is where the CSV file will be written to.
3.  The `logging_interval` is set to 0. This means that the CSV file will be written to every frame. The code block from `peekingduck.pipeline.nodes.output.utils.csvlogger` shows how the `logging_interval` logic is implemented.
    ```python
    if (curr_time - self.last_write).seconds >= self.logging_interval:
        self.writer.writerow(content)
        self.last_write = curr_time
    ```

!!! info
    A small recap, this node is used to write the results to a CSV file and is chained after the `dabble.exercise_counter` node. We pass the `output` dict of the `dabble.exercise_counter` as inputs to (`stats_to_track`).



##### Interpretation of CSV outputs

- Keypoints of elbow, shoulder, wrist as well as elbow angles are recorded every frame, if some of them are not detected by the model, it will be `None` and recorded as an **empty string** in the CSV file.
- If any of elbow, shoulder and wrist keypoints are `None`, then the corresponding elbow angle will be the same as the previous frame.
  

---

### Tests

Test the angle method. Throw in negative vectors and see what happens. Throw in empty vectors. Etc. maybe add a debug at that lvl if ppl throw in negative

