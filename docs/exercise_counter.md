<div align="center">
<h1>PeekingDuck Exercise Counter</a></h1>
by Hongnan Gao
1st May, 2022
<br>
</div>


## Push-Up Counter

The aim of this project is to build an exercise counter that can be used to detect push-ups, and more generic exercises in future iterations. Some content below are referenced from [PeekingDuck's Tutorial](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#recipe-2-keypoints-count-hand-waves).


The main model is **MoveNet**, which outputs seventeen keypoints for the person corresponding to the different body parts as documented here[^movenet_keypoints_id]. Each keypoint is a pair of $(x, y)$ coordinates, where $x$ and $y$ are real numbers ranging from $0.0$ to $1.0$ (using relative coordinates[^coordinate_systems]).

!!! note
    This is a **Minimum Viable Product (MVP)** and uses simple logic to detect push-ups. Consequently, there are a few somewhat rigid assumptions that will be made in the code.

We will detail the logic in the following sections.

[^movenet_keypoints_id]: [Keypoint IDs](https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html#whole-body-keypoint-ids)
[^coordinate_systems]: [Coordinate Systems](https://peekingduck.readthedocs.io/en/stable/tutorials/01_hello_cv.html#tutorial-coordinate-systems)

### Assumptions

This project makes a few assumptions about the input data for
simplicity of implementation:

1.  The user's **left body parts** are visible through the webcam/video
    and in particular, the **left elbow, wrist and shoulder** are
    crucial for our push-up counter.
   
    We can improve on this rigid requirement in the future by checking
    both the **left** and **right** body parts, and take the **side**
    with **higher keypoints confidence**.

2.  There should be only $1$ person in the video. As we are using
    **MoveNet's** `singlepose_thunder` model, we need to impose the
    restriction that there should be only $1$ person in the video to
    avoid performance issues.
   
    We can improve on this rigid requirement in the future by
    incorporating multi-person logic. The `multipose_lightning` can
    detect up to $6$ people in the video.

3. If user leaves the video and come back, we assume that the user is
   the same person and continue counting.
  

## Custom Node General Workflow

### Step 1. Initialize PeekingDuck Template 

After setting up from the [Workflows](./workflows.md) section, we can start using the PeekingDuck interface.

We initialize a new PeekingDuck project using the following commands:

!!! note "Terminal Session"
    ```bash title="Initializing PeekingDuck" linenums="1"
    $ mkdir custom_hn_exercise_counter
    $ cd custom_hn_exercise_counter
    $ peekingduck init
    ```

- `#!python [Line 1]`: Create a new directory named `custom_hn_exercise_counter` for the project.
- `#!python [Line 2]`: Change to the newly created directory.
- `#!python [Line 3]`: Initialize the PeekingDuck project in the current directory with default file/folder below.
  
Upon initialization of the project, PeekingDuck creates the following files in your
new project directory:

- `pipeline_config.yml` - Contains the pipeline[^pipeline_config] configuration and;
- `src/` - Folder for custom nodes. 
  
The `custom_hn_exercise_counter` directory currently looks like this:

```tree title="Directory Tree of custom_hn_exercise_counter"
custom_hn_exercise_counter/
├── pipeline_config.yml
└── src/
    └── custom_nodes/
        └──configs/
```

[^pipeline_config]: You can read more about pipeline config [in PeekingDuck's Documentation](https://peekingduck.readthedocs.io/en/stable/tutorials/01_hello_cv.html).

---

### Step 2. Use Pipeline Recipe to Create Custom Nodes

PeekingDuck provides several node types out of the box, for example a
MoveNet node to detect human poses within an image. To implement
additional functionality not provided by built-in nodes, we can create
[custom nodes](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html)
with our own logic written in Python.

For our project, we need to create two custom nodes. First, we will
create a `custom_nodes.input.visual` node. Although PeekingDuck
provides a visual input node, a small issue prevents filenames from URL
visual sources from registering properly. Registering the filename
properly is important, because we use it in later stages of our
pipeline, including writing output data to CSV. To fix this issue, we
implement this custom node.

Additionally, we implement a `custom_nodes.dabble.exercise_counter`
node. This node will take the keypoints from the MoveNet model and
count the number of push-ups performed by the person in the input
video.


1. To create these custom nodes, we first edit our `pipeline_config.yml`:

    ???+ example "Show/Hide content for pipeline_config.yml" 
        ```yaml title="pipeline_config.yml" linenums="1" hl_lines="2 17"
        nodes:
        - custom_nodes.input.visual:
            source: https://storage.googleapis.com/reighns/peekingduck/videos/push_ups.mp4
        - model.yolo:
            model_type: "v4tiny" 
            iou_threshold: 0.1
            score_threshold: 0.1
            detect_ids: ["person"] # [0]
            num_classes: 1
        #- custom_nodes.dabble.debug_yolo
        - model.movenet:
            model_type: "singlepose_thunder"
            resolution: {height: 256, width: 256} 
            bbox_score_threshold: 0.05
            keypoint_score_threshold: 0.05
        - dabble.fps
        - custom_nodes.dabble.exercise_counter:
            keypoint_threshold: 0.3
            exercise_name: "push_ups"
            push_up_pose_params: {
                starting_elbow_angle: 155,
                ending_elbow_angle: 90,
            }   
        - draw.poses
        - draw.legend:
            show: ["fps"]
        - output.csv_writer:
            stats_to_track: ["keypoints", "bboxes", "bbox_labels", "num_push_ups", "frame_count", "expected_pose", "elbow_angle","shoulder_keypoint", "elbow_keypoint", "wrist_keypoint", "filename"]
            file_path: "./stores/artifacts/push_ups_output_movenet.csv"
            logging_interval: 0
        - output.screen
        ```

    Adding our custom nodes (highlighted) is as simple as adding
    entries for them to the `pipeline_config.yml` file.
    
    Additionally, we add configuration for some nodes. For example, I
    specify that I want to use `v4tiny` model for the `model.yolo`
    node with a $0.1$ threshold for both iou and bounding box
    confidence score. These configurations will then be passed to the
    `config` parameter of the `model.yolo` node.

2. We then create the custom nodes using the [Pipeline Recipe method](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#pipeline-recipe)[^creating_nodes] with the following command: 
   
    !!! note "Terminal Session"
        ```bash title="Creating Custom Nodes"
        $ peekingduck create-node --config_path pipeline_config.yml
        ```

    This will create all the nodes listed in the `pipeline_config.yml` file. If one decides to add more custom nodes, we can simply add them to the `pipeline_config.yml` file and run the command again.

    The updated `custom_hn_exercise_counter` directory currently looks like this:

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

[^creating_nodes]: There are various ways to create custom nodes. See more from the [PeekingDuck documentation](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#).

---

### Step 3. Deep Dive into the Custom Nodes

After [Step 2](exercise_counter.md#step-2-use-pipeline-recipe-to-create-custom-nodes), three folders `config`, `dabble` and `input` will be populated in `src`. The `config` folder holds the *configurations* for the custom nodes while the `dabble` and `input` folders holds the *code* for the custom nodes. 
   
!!! info
    Something worth noting is that other **default nodes** that are in **PeekingDuck** will not be included in the `config` folder. For example, the `model.yolo` node is not included in the `config` folder because it is a **default node**. This is because the configurations for the default nodes are already included in the `pipeline_config.yml` file and will be **instantiated** when `peekingduck run` is called.

#### **custom_nodes.input.visual**

##### **input/visual.yml**

When defining a custom node, we must provide a default
configuration. We can use the [default
configuration](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/input/visual.yml)
from the built-in `input.visual` node:

???+ example "Show/Hide content for input.visual.yml"
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

##### **input/visual.py**

In **PeekingDuck** version `1.2.0`, if you define `source` to be a
**URL**, the `filename` is not overwritten by the `source` filename
and maintains the default `video.mp4`. There is a [pull
request](https://github.com/aimakerspace/PeekingDuck/pull/646) to fix
a similar issue where the filename was not set if the source is a single
image. In our `custom_nodes.input.visual` node, we apply a similar fix
for the URL case. This change will help us save the `filename` parameter
in the output CSV file later.

Standing on the shoulder of giants, I quickly modified a few lines to suit my purpose as I want to save the `filename` parameter in my output csv file later. 
Therefore, the code inside this file is mostly the same except for the highlighted lines below.

To implement this custom node, copy the
[`peekingduck/pipeline/nodes/input/visual.py`](https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/pipeline/nodes/input/visual.py)
file from the PeekingDuck `dev` branch to
`custom_hn_exercise_counter/src/custom_nodes/input/visual.py` and
update the highlighted lines:

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

!!! info "Explanation"
    In the `run` method of `input.visual.py`, we note that the `output` dict is called by
    `outputs = self._get_next_frame()` and `self._file_name` is therefore
    crucial to output the correct filename of the source. Since `source`
    is set as `self.source`, we use `pathlib.Path` to convert it to a
    `pathlib.Path` object and call the
    [`name`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.name)
    property to get the filename.

    ???+ example "Show/Hide content for _get_next_frame()"
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

#### **custom_nodes.dabble.exercise_counter**

##### **dabble/exercise_counter.yml**

Running `peekingduck create-node` command creates a default configuration for the `custom_nodes.dabble.exercise_counter` node:

```yaml title="Default config for dabble/exercise_counter.yml" linenums="1"
input: ["bboxes", "bbox_labels"]    # (1)
output: ["obj_attrs", "custom_key"] # (2)

threshold: 0.5                      # (3)
``` 

1.  Mandatory configs. The default configurations receive bounding
    boxes and their respective labels as input. Replace with other
    data types as required. List of built-in data types for
    PeekingDuck can be found in
    [here](https://peekingduck.readthedocs.io/en/stable/glossary.html).
2.  Output `obj_attrs` for visualization with `draw.tag` node and
    `custom_key` for use with other custom nodes. Replace as required.
3.  Optional configs depending on node. We will go through that later.

We need to edit the file according to our own needs. Since we are
chaining **Yolo** and **MoveNet**, this node should therefore take in
`img`, `bboxes`, `bbox_scores`, `keypoints`, and `keypoint_scores` as
inputs from the pipeline and outputs a dictionary with keys such as
`frame_count`, `num_push_ups`, `body_direction`, `elbow_angle`,
`shoulder_keypoint`, `elbow_keypoint`, and `wrist_keypoint`.

We also want to define some optional configuration items for the
`dabble.exercise_counter` node. For example, users may specify a
custom exercise name in the `exercise_name` configuration item. For now,
the only supported value is `push_ups`.

Consequently, the newly updated `dabble.exercise_counter.yml` file
should contain the following:

``` yaml title="dabble.exercise_counter.yml" linenums="1"
input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]         # (1)
output: ["frame_count", "num_push_ups", "expected_pose", "elbow_angle",
        "shoulder_keypoint", "elbow_keypoint", "wrist_keypoint", "filename"]    # (2)

# Optional configs
keypoint_threshold: 0.3                                                         # (3)
exercise_name: "push_ups"                                                       # (4)     
push_up_pose_params: {
    starting_elbow_angle: 155,
    ending_elbow_angle: 90,
}                                                                               # (5)                              
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
    - `elbow_angle`: The angle between the wrist, elbow and the
      shoulder in the current frame.
    - `shoulder_keypoint`: The keypoint of the shoulder in the current frame.
    - `elbow_keypoint`: The keypoint of the elbow in the current frame.
    - `wrist_keypoint`: The keypoint of the wrist in the current frame.
3.  This is an optional configuration parameter that will be
    initialized if defined. Here I defined a `keypoint_threshold`
    parameter.
4.  This is an optional configuration parameter that will be
    initialized if defined. Here I defined a `exercise_name`
    parameter.
5.  This is an optional configuration parameter that will be
    initialized if defined. Here I defined a `push_up_pose_params`
    parameter.

Let us walk through the [mandatory
inputs](exercise_counter.md#mandatory-inputs) and [optional
configs](exercise_counter.md#optional-configs) of
`dabble.exercise_counter.yml` in the next two sections. One should
also refer back by clicking the ➕ beside each line of code in the
above file for annotations.


###### Mandatory Default Configuration Items

All nodes must specify `input` and `output` items in their default
configuration file. All items specified in the `input` array must be
computed and returned by previous nodes in the pipeline. For the
`dabble.exercise_counter` node, we rely on previous nodes to provide
inputs for our exercise detection algorithm, such as the detected
keypoints.

!!! warning
    Since both `PoseNet/MoveNet` and `Yolo` have the same `output` key
    `bboxes`, chaining `PoseNet/Movenet` after `Yolo` will cause the
    common output `bboxes` to be overwritten by the latter.


###### Optional Configs

Recall the optional configs defined in `dabble.exercise_counter.yml`:

``` yaml title="optional config" linenums="1"
# Optional configs
keypoint_threshold: 0.3                                                
exercise_name: "push_ups" 
push_up_pose_params: {
    starting_elbow_angle: 155,
    ending_elbow_angle: 90,
}                                                                                    
```

These configuration items will set as class instance **attributes** of the custom node `Node(AbstractNode)` upon initialization. 

```python title="abstract node class" linenums="1"
class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
```

!!! info
    For example, when you pass an additional parameter `exercise_name` in `exercise_counter.yml`, this parameter will be instantiated in the `dabble.exercise_counter` `Node` as an attribute `self.exercise_name`. 

    A quick check in the source code `peekingduck.pipeline.nodes.node.py` shows that it will set all the keys defined in the `.yml` file.
    ```python
    # sets class attributes
    for key in self.config:
        setattr(self, key, self.config[key])
    ```

!!! note
    As an aside, the other optional argument passed is
    `keypoint_threshold` which is used to determine the threshold for the
    keypoints. Technically it behaves the same as the
    `keypoint_score_threshold` in `model.movenet`'s Configs.

##### **dabble/exercise_counter.py**

The `dabble.exercise_counter` implements our push-up counter. 

<figure markdown>
  ![Image title](https://storage.googleapis.com/reighns/peekingduck/images/pushup_fortune-vieyra-jD4MtXnsJ6w-unsplash_LI.jpg){ width="600" }
  <figcaption>Fig 1: Push-up Image by Fortune Vieyra via Unsplash - Copyright-free</figcaption>
</figure>

With the
[assumptions](exercise_counter.md#assumptions) in the earlier section,
our heuristic is:

1.  As shown in Figure 1, let the person's left shoulder, elbow and
    wrist be point (keypoints of $(x,y)$ coordinates) $A$, $B$ and $C$ respectively. Then the angle
    $\angle{ABC}$ formed by the line vector $\vec{BA}$ and $\vec{BC}$ is
    the elbow angle.
   
    Define the following:

    - $\angle{S}=155^{\circ}$ to be the threshold for starting elbow angle;
    - $\angle{E}=90^{\circ}$ to be the threshold fpr ending elbow angle[^s_configurable];
    - $N=0$ as the number of push-ups;
    - $H=\text{False}$ as whether the person has started performing push-ups, and;
    - $P\in\{\text{up}, \text{down}\}=\text{down}$ as the expected pose.
    
    ${\color{red} \text{We say that the person is in an up pose if} \angle{ABC} >
    S, \text{and in a down pose if} \angle{ABC} \le E.}$
 
2.  Once the person assumes an $\text{up}$ pose for the first time (i.e. getting ready for push up),
    set $H=\text{True}$.

3.  If $H=\text{True}$, and the person assumes a $\text{down}$ pose,
    and $P=\text{down}$, set $N=N+0.5$ and $P=\text{up}$.

    Otherwise, if $H=\text{True}$ and the person assumes an $\text{up}$
    pose, and $P=\text{up}$, set $N=N+0.5$ and $P=\text{down}$.
    
We detect the person performing a push-up as a cycle. A push-up starts
when the person has an $\text{up}$ pose. When the person moves to a
$\text{down}$ pose, they have completed half the cycle, so we
increment $N$ by $0.5$. When they return to an $\text{up}$ pose, they
have completed the cycle, so $N$ is incremented by $0.5$ again, giving
a total incremnt of $1$ per cycle.

This code below implements the logic detailed previously.
!!! note
    Note that there are three helper functions `map_bbox_to_image_coords`, `map_keypoint_to_image_coords` and `draw_text` to convert relative to absolute coordinates and to draw text on-screen. These are taken from [PeekingDuck's Tutorial on counting hand waves](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#recipe-2-keypoints-count-hand-waves). 

??? example "Show/Hide code for dabble/exercise_counter.py"
    ```python title="dabble/exercise_counter.py" linenums="1"
    @dataclass(frozen=True)
    class GlobalParams:
        """Global parameters for the node.

        Attributes:
            FONT (int): Font for the text.
            WHITE (int): White color in BGR.
            YELLOW (int): Yellow color in BGR.
            KP_NAME_TO_INDEX (Dict[str, int]): PoseNet/MoveNet's skeletal keypoints name to index mapping.
        """

        FONT: int = cv2.FONT_HERSHEY_SIMPLEX
        WHITE: Tuple[int] = (255, 255, 255)
        YELLOW: Tuple[int] = (0, 255, 255)

        KP_NAME_TO_INDEX: Dict[str, int] = field(
            default_factory=lambda: {
                "nose": 0,
                "left_eye": 1,
                "right_eye": 2,
                "left_ear": 3,
                "right_ear": 4,
                "left_shoulder": 5,
                "right_shoulder": 6,
                "left_elbow": 7,
                "right_elbow": 8,
                "left_wrist": 9,
                "right_wrist": 10,
                "left_hip": 11,
                "right_hip": 12,
                "left_knee": 13,
                "right_knee": 14,
                "left_ankle": 15,
                "right_ankle": 16,
            }
        )


    @dataclass(frozen=False)
    class PushupPoseParams:
        """Push up pose parameters.

        Attributes:
            starting_elbow_angle (float): The threshold angle formed between the wrist, elbow and shoulder for starting (up) pose.
            ending_elbow_angle (float): The threshold angle formed between the wrist, elbow and shoulder for ending (down) pose.
        """

        starting_elbow_angle: float
        ending_elbow_angle: float

        @classmethod
        def from_dict(
            cls: Type["PushupPoseParams"], params_dict: Dict[str, Any]
        ) -> Type["PushupPoseParams"]:
            """Takes in a dictionary of parameters and returns a PushupPoseParams Dataclass.

            Args:
                params_dict (Dict[str, Any]): Dictionary of parameters.

            Returns:
                (PushupPoseParams): Dataclass with the parameters initalized.
            """
            return cls(
                starting_elbow_angle=params_dict["starting_elbow_angle"],
                ending_elbow_angle=params_dict["ending_elbow_angle"],
            )


    class Node(AbstractNode):
        """Custom node to display keypoints and count number of hand waves

        Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
        config.exercise_name (str): Name of the exercise. Default: "pushups".
        config.keypoint_threshold (float): Ignore keypoints below this threshold. Default: 0.3.
        config.push_up_pose_params (:obj:`Dict[str, Any]`): Parameters for the push up pose. Default: {starting_elbow_angle: 155, ending_elbow_angle: 90}.


        Attributes:
            self.frame_count (int): Track the number of frames processed.
            self.expected_pose (str): The expected pose. Default: "down".
            self.num_push_ups (float): Cumulative number of push ups. Default: 0.
            self.have_started_push_ups (bool): Whether or not the push ups have started. Default: False.
            self.elbow_angle (float): Angle of the elbow. Default: None.
            self.global_params_dataclass (GlobalParams): Global parameters for the node.
            self.push_up_pose_params_dataclass (PushupPoseParams): Push up pose parameters.
            self.interested_keypoints (List[str]): List of keypoints to track. Default: ["left_elbow", "left_shoulder", "left_wrist"].
            self.left_elbow (float): Keypoints of the left elbow. Default: None.
            self.left_shoulder (float): Keypoints of the left shoulder. Default: None.
            self.left_wrist (float): Keypoints of the left wrist. Default: None.
        """

        def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
            super().__init__(config, node_path=__name__, **kwargs)

            self.logger = logging.getLogger(__name__)

            self.exercise_name: str
            self.keypoint_threshold: float  # ignore keypoints below this threshold
            self.push_up_pose_params: Dict[str, Any]

            self.logger.info(f"Initialize Exercise Type: {self.exercise_name}!")

            self.frame_count = 0

            self.expected_pose = "down"
            self.num_push_ups = 0
            self.have_started_push_ups = False
            self.elbow_angle = None

            self.global_params_dataclass = GlobalParams()
            self.push_up_pose_params_dataclass = PushupPoseParams.from_dict(
                self.push_up_pose_params
            )

            self.interested_keypoints = [
                "left_elbow",
                "left_shoulder",
                "left_wrist",
            ]

            # each element in the self.interested_keypoints list will now become an attribute initialized to None
            self.reset_keypoints_to_none()

        def reset_keypoints_to_none(self) -> None:
            """Reset all keypoints attributes to None after each frame."""

            for interested_keypoint in self.interested_keypoints:
                setattr(self, interested_keypoint, None)

        def inc_num_push_ups(self) -> float:
            """Increments the number of push ups by 0.5 for every directional change.

            Returns:
                self.num_push_ups (float): Cumulative number of push ups.
            """
            self.num_push_ups += 0.5
            return self.num_push_ups

        def is_up_pose(self, elbow_angle: float) -> bool:
            """Checks if the pose is an "up" pose by checking if the elbow angle is
            more than the starting elbow angle threshold.

            Elbow angle is defined by connecting the wrist to the elbow to the shoulder.

            Args:
                elbow_angle (float): The angle of the elbow.

            Returns:
                bool: True if the pose is a up pose, False otherwise.
            """
            return (
                elbow_angle
                > self.push_up_pose_params_dataclass.starting_elbow_angle
            )

        def is_down_pose(self, elbow_angle: float) -> None:
            """Checks if the pose is an "down" pose by checking if the elbow angle is
            more than the ending elbow angle threshold.

            Elbow angle is defined by connecting the wrist to the elbow to the shoulder.

            Args:
                elbow_angle (float): The angle of the elbow.

            Returns:
                bool: True if the pose is a down pose, False otherwise.
            """
            return (
                elbow_angle
                <= self.push_up_pose_params_dataclass.ending_elbow_angle
            )

        @staticmethod
        def is_bbox_or_keypoints_empty(
            bboxes: np.ndarray,
            keypoints: np.ndarray,
            keypoint_scores: np.ndarray,
        ) -> bool:
            """Checks if the bounding box or keypoints are empty.

            If any of them are empty, then we won't perform any further processing and go to next frame.

            Args:
                bboxes (np.ndarray): The bounding boxes.
                keypoints (np.ndarray): The keypoints.
                keypoint_scores (np.ndarray): The keypoint scores.

            Returns:
                bool: True if the bounding box or keypoints are empty, False otherwise.
            """

            return (
                len(bboxes) == 0
                or len(keypoints) == 0
                or len(keypoint_scores) == 0
            )

        # pylint: disable=too-many-locals
        def count_push_ups(
            self,
            img: np.ndarray,
            img_size: Tuple[int, int],
            the_keypoints: np.ndarray,
            the_keypoint_scores: np.ndarray,
        ) -> None:
            """Counts the number of push ups.

            Args:
                img (np.ndarray): The image in each frame.
                img_size (Tuple[int, int]): The width and height of the image in each frame.
                the_keypoints (np.ndarray): The keypoints predicted in each frame.
                the_keypoint_scores (np.ndarray): The keypoint scores predicted in each frame.
            """

            interested_keypoints_names_to_index = {
                self.global_params_dataclass.KP_NAME_TO_INDEX[
                    interested_keypoint
                ]: interested_keypoint
                for interested_keypoint in self.interested_keypoints
            }

            self.reset_keypoints_to_none()

            for keypoint_idx, (keypoints, keypoint_score) in enumerate(
                zip(the_keypoints, the_keypoint_scores)
            ):
                if keypoint_score >= self.keypoint_threshold:

                    x, y = map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size
                    )
                    x_y_str = f"({x}, {y})"

                    if keypoint_idx in interested_keypoints_names_to_index:
                        keypoint_name = interested_keypoints_names_to_index[
                            keypoint_idx
                        ]
                        setattr(self, keypoint_name, (x, y))
                        the_color = self.global_params_dataclass.YELLOW
                    else:
                        the_color = self.global_params_dataclass.WHITE

                    draw_text(
                        img,
                        x_y_str,
                        (x, y),
                        color=the_color,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        thickness=2,
                    )

            # all keypoints must be non-none
            if self.left_elbow and self.left_shoulder and self.left_wrist:

                left_elbow_angle = self.calculate_angle_using_dot_prod(
                    self.left_shoulder, self.left_elbow, self.left_wrist
                )

                self.elbow_angle = left_elbow_angle

                # Check to ensure right form before starting the program
                if self.is_up_pose(left_elbow_angle):
                    self.have_started_push_ups = True

                # Check for full range of motion for the pushup
                if self.have_started_push_ups:
                    # the two if-statements are mutually exclusive: won't happen at the same time.
                    if (
                        self.is_down_pose(left_elbow_angle)
                        and self.expected_pose == "down"
                    ):
                        self.inc_num_push_ups()
                        self.expected_pose = "up"

                    if (
                        self.is_up_pose(left_elbow_angle)
                        and self.expected_pose == "up"
                    ):
                        self.inc_num_push_ups()
                        self.expected_pose = "down"

                pushup_str = f"#push_ups = {self.num_push_ups}"
                draw_text(
                    img,
                    pushup_str,
                    (20, 30),
                    color=self.global_params_dataclass.YELLOW,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=3,
                )

        # pylint: disable=trailing-whitespace
        @staticmethod
        def calculate_angle_using_dot_prod(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            return_as_degrees: bool = True,
        ) -> float:
            r"""Takes in three points and calculates the angle between them using dot product.
            Let B be the common point of two vectors BA and BC, then angle ABC is the angle between vectors BA and BC.

            This function calculates the angle ABC using the dot product formula:
        
            $$
            \begin{align*}
            BA &= a - b \\
            BC &= c - b \\
            \cos(angle(ABC)) &= \dfrac{(BA \cdot BC)}{(|BA||BC|)}
            \end{align*}
            $$

            Args:
                a (np.ndarray): Point a corresponding to A.
                b (np.ndarray): Point b corresponding to B.
                c (np.ndarray): Point c corresponding to C.
                return_as_degrees (bool): Returns angle in degrees if True else radians. Default: True.

            Returns:
                angle (float): Angle between vectors BA and BC in radians or degrees.

            Shape:
                - Input:
                    - a (np.ndarray): (2, )
                    - b (np.ndarray): (2, )
                    - c (np.ndarray): (2, )

            Examples:
                >>> import numpy as np
                >>> a = (6, 0)
                >>> b = (0, 0)
                >>> c = (6, 6)
                >>> calculate_angle_using_dot_prod(a,b,c)
                45.0
            """
            # turn the points into numpy arrays
            a = np.asarray(a)
            b = np.asarray(b)
            c = np.asarray(c)
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (
                np.linalg.norm(ba) * np.linalg.norm(bc)
            )
            # arccos range is [0, pi]
            angle = np.arccos(cosine_angle)

            if return_as_degrees:
                return np.degrees(angle)
            return angle

        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            """This node draws keypoints and counts the number of push ups.

            Args:
                inputs (dict): Dictionary with keys
                    - img (np.ndarray): The image in each frame.
                    - bboxes (np.ndarray): The bounding boxes predicted in each frame. Note this belongs to MoveNet.
                    - bbox_scores (np.ndarray): The bounding box scores predicted in each frame. Note this belongs to Yolo.
                    - keypoints (np.ndarray): The keypoints predicted in each frame.
                    - keypoint_scores (np.ndarray): The keypoint scores predicted in each frame.
                    - filename (str): The filename of the image/video.

            Note:
                To check the shapes of bboxes, bbox_scores, keypoints, and keypoint_scores, please refer to [PeekingDuck API Documentation](https://peekingduck.readthedocs.io/en/stable/nodes/model.movenet.html#module-model.movenet).

            Returns:
                outputs (dict): Dictionary with keys
                    - filename: The filename of the image/video.
                    - expected_pose: The expected pose of the image/video.
                    - num_push_ups: The number of cumulative push ups.
                    - frame_count: The number of frames processed.
                    - elbow_angle: The angle between the wrist, elbow and shoulder.
                    - elbow_keypoint: The keypoint coordinates of the elbow.
                    - shoulder_keypoint: The keypoint coordinates of the shoulder.
                    - wrist_keypoint: The keypoint coordinates of the wrist.
            """

            # get required inputs from pipeline
            img = inputs["img"]
            bboxes = inputs["bboxes"]
            bbox_scores = inputs["bbox_scores"]
            keypoints = inputs["keypoints"]
            keypoint_scores = inputs["keypoint_scores"]
            filename = inputs["filename"]

            # image width, height
            img_size = (img.shape[1], img.shape[0])

            # frame count should not be in the if-clause
            self.frame_count += 1

            if not self.is_bbox_or_keypoints_empty(
                bboxes, keypoints, keypoint_scores
            ):
                # assume each frame has only one person;
                # note this bbox is from the posenet/movenet and not from yolo.
                the_bbox = bboxes[0]
                # bbox_scores are from yolo and not posenet/movenet.
                the_bbox_score = bbox_scores[0] if len(bbox_scores) > 0 else 0

                x1, _y1, _x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
                score_str = f"BBox {the_bbox_score:0.2f}"

                # get bounding box confidence score and draw it at the left-bottom
                # (x1, y2) corner of the bounding box (offset by 30 pixels)
                draw_text(
                    img,
                    score_str,
                    (x1, y2 - 30),
                    color=self.global_params_dataclass.WHITE,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    thickness=3,
                )

                # assume each frame has only one person;
                the_keypoints = keypoints[0]
                the_keypoint_scores = keypoint_scores[0]

                # count the number of push ups
                self.count_push_ups(
                    img, img_size, the_keypoints, the_keypoint_scores
                )

            # careful not to indent this return statement;
            # if the if-clause is false, then no dict will be returned and will crash the pipeline
            return {
                "filename": filename,
                "expected_pose": self.expected_pose,
                "num_push_ups": self.num_push_ups,
                "frame_count": self.frame_count,
                "elbow_angle": self.elbow_angle,
                "elbow_keypoint": self.left_elbow,
                "shoulder_keypoint": self.left_shoulder,
                "wrist_keypoint": self.left_wrist,
            }
    ```

[^s_configurable]: The values for $S$ and $E$ are configurable, but $155^{\circ}$ and $90^{\circ}$ are good defaults.

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

The snippet below shows the CSV file contents.

<figure markdown>
  ![Image title](https://storage.googleapis.com/reighns/peekingduck/images/exercise_counter_csv_snippet.PNG){ width="600" }
  <figcaption>Fig 2: Push-up Counter CSV</figcaption>
</figure>

- Keypoints of elbow, shoulder, wrist as well as elbow angles are recorded every frame, if some of them are not detected by the model, it will be `None` and recorded as an **empty string** in the CSV file.
- If any of elbow, shoulder and wrist keypoints are `None`, then the corresponding elbow angle will be the same as the previous frame.
  

## References

This is my first encounter with Pose Estimation. Here are some references that I draw inspiration from.

- **[Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)**
- **[Multi-Person Pose Estimation with Mediapipe](https://shawntng.medium.com/multi-person-pose-estimation-with-mediapipe-52e6a60839dd)**
- **[How I created the Workout Movement Counting App using Deep Learning and Optical Flow Algorithm](https://towardsdatascience.com/how-i-created-the-workout-movement-counting-app-using-deep-learning-and-optical-flow-89f9d2e087ac)**
- **[Human Pose Classification with MoveNet and TensorFlow Lite](https://www.tensorflow.org/lite/tutorials/pose_classification)**
- **[MoveNet: Ultra fast and accurate pose detection model](https://www.tensorflow.org/hub/tutorials/movenet)**
- **[Deep learning approaches for workout repetition counting and validation](https://www.sciencedirect.com/science/article/pii/S016786552100324X#!)**
- **[Push-up counter using Mediapipe python](https://www.youtube.com/watch?v=ZI2-Xl0J8S4)**
- **[Pose Classification From MediaPipe](https://google.github.io/mediapipe/solutions/pose_classification.html)**
- **[RepCounter using PoseNet](https://github.com/abishekvashok/Rep-Counter)**
- **[Exercise Reps Counter || Pose Estimation](https://github.com/akshatkaush/exercise-count)**
- **[Deep Learning Exercise Repetitions Counter](https://github.com/NetoPedro/Deep-Learning-Push-Up-Counter)**
- **[Real-time Human Pose Estimation in the Browser with TensorFlow.js](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)**
- **[Fitness Camera – Turn Your Phone's Camera Into a Fitness Tracker](https://miguelrochefort.com/blog/fitness-camera/)**
