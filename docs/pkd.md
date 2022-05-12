

## Source Code Tracing

- `peekingduck run`: If you check `entry_points.txt`, the entry point is indeed `peekingduck` pointing to the package `peekingduck.cli` script. The `cli` after the colon `:` means that we will look at the `decorator` commands `@cli.command()`.
    
    ```txt title="setup.cfg | entry_points.txt" linenums="1"
    # https://github.com/aimakerspace/PeekingDuck/blob/dev/setup.cfg
    [options.entry_points]
    console_scripts =
        peekingduck = peekingduck.cli:cli
    ```
    Consequently, when `peekingduck run` is called, it goes through the following files:

- `#!python [peekingduck.cli.run()-Line 1]:` The decorator `@cli.command()` is called. No arguments were passed on the command line and therefore the default arguments be taken from `@click.option()`.
- `#!python [peekingduck.cli.run()-Lines 2-26]:` The decorator `@click.option()` takes in various things such as default arguments for the function. For example, the default argument for `node_config` is defaulted to `None`.
- `#!python [peekingduck.cli.run()-Lines 27-33]:` This is the function that is called when `peekingduck run` is called. 
    - Let us check the input arguments here:
        - `config_path`: If not provided, it defaults to `pipeline_config.yml`. The new variable is called `pipeline_config_path`.
        - `log_level`: `INFO` level.
        - `node_config`: `None` as no custom arguments provided.
        - `num_iter`: `None`
        - `nodes_parent_dir`: Points to `src`, the folder that was created by default! Override if you want to rename your folder.
- `#!python [peekingduck.cli.run()-Lines 38-46]:` We set `pipeline_config_path` to `Path(config_path)`. This variable holds the path to our `pipeline_config.yml` file.
- `#!python [peekingduck.cli.run()-Lines 49-55]:` The `Runner` class is called, we can trace the code in the `peekingduck.runner.py`. We will now go to the `peekingduck.runner.py` file. Note that I highlighted line 54 to set `nodes=None` for my own clarity purposes.
- `#!python [peekingduck.runner.Runner()-Lines 27-34]:` We see that `Runner()` class takes in the following arguments:
    - `pipeline_path: Path = None`
    - `config_updates_cli: str = None`
    - `custom_nodes_parent_subdir: str = None`
    - `num_iter: int = None`
    - `nodes: List[AbstractNode] = None`
  
    All of which were passed in to the `Runner` class in `#!python [peekingduck.cli.run()-Lines 49-55]`.

- `#!python [peekingduck.runner.Runner()-Lines 41-45]:` The `Runner()` class took in `pipeline_path`, `config_updates_cli` and `custom_nodes_parent_subdir` and thus will go into the `elif` of the `if-elif-else` statement.
- `#!python [peekingduck.runner.Runner()-Lines 47-51]:` `self.node_loader` will call `DeclarativeLoader` which is a helper class to create the final `Pipeline <peekingduck.pipeline.pipeline.Pipeline>` object. Let us see what it does at a high level. Take note that at this stage we are just **initializing** the the `DeclarativeLoader` class; consequently, we should look what happens in its `__init__` method.
- `#!python [peekingduck.declarative_loader.DeclarativeLoader()-Lines 53-57]:` We take a look at some key attributes and methods of the `DeclarativeLoader` class.
    - `#!python [peekingduck.declarative_loader.DeclarativeLoader()-Line 33]:` The attribute `self.node_list = self._load_node_list(pipeline_path)` is instantiated. We take a look at one of the important attributes of the `DeclarativeLoader` class, `node_list`. This is a `NodeList` object (defined in the same file) that has the attribute `nodes` which gives us the following. This means they loaded all the config in pipeline_config.yml into a list of dict.
        - `self.node_list`: It returns `NodeList(upgraded_nodes)` as a `NodeList` object, an abstract object defined in `#!python [peekingduck.declarative_loader.DeclarativeLoader()-Line 214-240]`.
        - `self.node_list.nodes`: List of Dict - basically loads all config from `pipeline_config.yml` into a list of dict.
        ```python
        [
            {
                "input.visual": {
                    "source": "https://storage.googleapis.com/reighns/peekingduck/videos/wave.mp4"
                }
            },
            {
                "model.yolo": {
                    "model_type": "v4tiny",
                    "iou_threshold": 0.1,
                    "score_threshold": 0.1,
                    "detect_ids": ["person"],
                    "num_classes": 1,
                }
            },
            {
                "model.posenet": {
                    "model_type": "resnet",
                    "resolution": {"height": 224, "width": 224},
                    "score_threshold": 0.05,
                }
            },
            "dabble.fps",
            "custom_nodes.dabble.exercise_counter",
            "draw.poses",
            "model.mtcnn",
            "draw.mosaic_bbox",
            {"draw.legend": {"show": ["fps"]}},
            "output.screen",
        ]
        ```
- `#!python [peekingduck.runner.Runner()-Line 52]:` Creates `self.pipeline` which calls the `get_pipeline()` method in `#!python [peekingduck.declarative_loader.DeclarativeLoader()-Line 199]`. 
- `#!python [peekingduck.declarative_loader.DeclarativeLoader()-Line 204]:` `instantiated_nodes = self._instantiate_nodes()` is where we create all the nodes. I'd imagine the nodes are created by looping over the list of dict just now, and for each node dict in the list, we create the `AbstractNode` class from it. It may be useful to check what `Pipeline` contains! Let us hop over to from `peekingduck.pipeline.pipeline.Pipeline`.
- Going into `from peekingduck.pipeline.pipeline import Pipeline` object, we see that the `Pipeline` object holds an attribute `nodes`, which unsurprisingly, refers to a list of `AbstractNode` object (i.e. `List[AbstractNode]`).
- `#!python [peekingduck.cli.run()-Line 58]:` We are finally back to the last line of the `peekingduck.cli.run()` function. We call the `Runner.run()` method.
- `#!python [peekingduck.runner.Runner()-Line 78]:` Start of `while` loop with `self.pipeline.terminate` as `False` at first.
- `#!python [peekingduck.runner.Runner()-Line 79]:` Loop over `self.pipeline.nodes`, recall that `self.pipeline.nodes` is a list of `AbstractNode` objects.
- `#!python [peekingduck.runner.Runner()-Line 83-84]:` Terminating conditions.
- `#!python [peekingduck.runner.Runner()-Line 91-95]:` Update `inputs` dict if we can find a `key` in `self.pipeline.data`. This part was confusing at first since the first **iteration** the `self.pipeline.data` is an empty dict. Upon checking, it turns out that the first `Node` is `input.visual` which takes in `input` as `None` and `output` as `output: ["img", "filename", "pipeline_end", "saved_video_fps"]`. Therefore in the first iteration, there won't be any `key` and thus `inputs` will be `{}` but `outputs` from `#!python [peekingduck.runner.Runner()-Line 103]` will be `{"img": ..., "filename": ..., "pipeline_end": ..., "saved_video_fps": ...}`. Consequently, `self.pipeline.data` will update the dictionary `inputs` with the `outputs` from the first `Node`. It will keep chaining on till the last node. 

    ```python title="peekingduck.cli.run()" linenums="1" hl_lines="54"
    @cli.command()
    @click.option(
        "--config_path",
        default=None,
        type=click.Path(),
        help=(
            "List of nodes to run. None assumes pipeline_config.yml at current working directory"
        ),
    )
    @click.option(
        "--log_level",
        default="info",
        help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
    )
    @click.option(
        "--node_config",
        default="None",
        help="""Modify node configs by wrapping desired configs in a JSON string.\n
            Example: --node_config '{"node_name": {"param_1": var_1}}'""",
    )
    @click.option(
        "--num_iter",
        default=None,
        type=int,
        help="Stop pipeline after running this number of iterations",
    )
    def run(
        config_path: str,
        log_level: str,
        node_config: str,
        num_iter: int,
        nodes_parent_dir: str = "src",
    ) -> None:
        """Runs PeekingDuck"""

        LoggerSetup.set_log_level(log_level)

        if config_path is None:
            curr_dir = _get_cwd()
            if (curr_dir / "pipeline_config.yml").is_file():
                config_path = curr_dir / "pipeline_config.yml"
            elif (curr_dir / "run_config.yml").is_file():
                config_path = curr_dir / "run_config.yml"
            else:
                config_path = curr_dir / "pipeline_config.yml"
        pipeline_config_path = Path(config_path)

        start_time = perf_counter()
        runner = Runner(
            pipeline_path=pipeline_config_path,
            config_updates_cli=node_config,
            custom_nodes_parent_subdir=nodes_parent_dir,
            num_iter=num_iter,
            nodes=None
        )
        end_time = perf_counter()
        logger.debug(f"Startup time = {end_time - start_time:.2f} sec")
        runner.run()
    ```

    ```python title="peekingduck.runner.Runner()" linenums="1"
    class Runner:
        """The runner class for creation of pipeline using declared/given nodes.

        The runner class uses the provided configurations to setup a node pipeline
        which is used to run inference.

        Args:
            pipeline_path (:obj:`pathlib.Path` | :obj:`None`): If a path to
                *pipeline_config.yml* is provided, uses
                :py:class:`DeclarativeLoader <peekingduck.declarative_loader.DeclarativeLoader>`
                to load the YAML file according to PeekingDuck's specified schema
                to obtain the declared nodes that would be sequentially initialized
                and used to create the pipeline for running inference.
            config_updates_cli (:obj:`str` | :obj:`None`): Configuration changes
                passed as part of the CLI command used to modify the node
                configurations directly from CLI.
            custom_nodes_parent_subdir (:obj:`str` | :obj:`None`): Relative path to
                a folder which contains custom nodes that users have created to be
                used with PeekingDuck. For more information on using custom nodes,
                please refer to
                `Getting Started <getting_started/03_custom_nodes.html>`_.
            num_iter (int): Stop pipeline after running this number of iterations
            nodes (:obj:`List[AbstractNode]` | :obj:`None`): If a list of nodes is
                provided, initialize by the node stack directly.
        """

        def __init__(  # pylint: disable=too-many-arguments
            self,
            pipeline_path: Path = None,
            config_updates_cli: str = None,
            custom_nodes_parent_subdir: str = None,
            num_iter: int = None,
            nodes: List[AbstractNode] = None,
        ) -> None:
            self.logger = logging.getLogger(__name__)
            try:

                if nodes:
                    # instantiated_nodes is created differently when given nodes
                    self.pipeline = Pipeline(nodes)
                elif (
                    pipeline_path
                    and config_updates_cli
                    and custom_nodes_parent_subdir
                ):
                    # create Graph to run
                    self.node_loader = DeclarativeLoader(
                        pipeline_path,
                        config_updates_cli,
                        custom_nodes_parent_subdir,
                    )
                    self.pipeline = self.node_loader.get_pipeline()
                else:
                    raise ValueError(
                        "Arguments error! Pass in either nodes to load directly via "
                        "Pipeline or pipeline_path, config_updates_cli, and "
                        "custom_nodes_parent_subdir to load via DeclarativeLoader."
                    )
            except ValueError as error:
                self.logger.error(str(error))
                sys.exit(1)
            if RequirementChecker.n_update > 0:
                self.logger.warning(
                    f"{RequirementChecker.n_update} package"
                    f"{'s' * int(RequirementChecker.n_update > 1)} updated. "
                    "Please rerun for the updates to take effect."
                )
                sys.exit(3)
            if num_iter is None or num_iter <= 0:
                self.num_iter = 0
            else:
                self.num_iter = num_iter
                self.logger.info(f"Run pipeline for {num_iter} iterations")

        def run(self) -> None:  # pylint: disable=too-many-branches
            """execute single or continuous inference"""
            num_iter = 0
            while not self.pipeline.terminate:
                for node in self.pipeline.nodes:
                    if num_iter == 0:  # report node setup times at first iteration
                        self.logger.debug(f"First iteration: setup {node.name}...")
                        node_start_time = perf_counter()
                    if self.pipeline.data.get("pipeline_end", False):
                        self.pipeline.terminate = True
                        if "pipeline_end" not in node.inputs:
                            continue

                    if "all" in node.inputs:
                        inputs = copy.deepcopy(self.pipeline.data)
                    else:
                        inputs = {
                            key: self.pipeline.data[key]
                            for key in node.inputs
                            if key in self.pipeline.data
                        }
                    if hasattr(node, "optional_inputs"):
                        for key in node.optional_inputs:
                            # The nodes will not receive inputs with the optional
                            # key if it's not found upstream
                            if key in self.pipeline.data:
                                inputs[key] = self.pipeline.data[key]

                    outputs = node.run(inputs)
                    self.pipeline.data.update(outputs)
                    if num_iter == 0:
                        node_end_time = perf_counter()
                        self.logger.debug(
                            f"{node.name} setup time = {node_end_time - node_start_time:.2f} sec"
                        )
                num_iter += 1
                if self.num_iter > 0 and num_iter >= self.num_iter:
                    self.logger.info(
                        f"Stopping pipeline after {num_iter} iterations"
                    )
                    break

            # clean up nodes with threads
            for node in self.pipeline.nodes:
                if node.name.endswith(".visual"):
                    node.release_resources()

        def get_pipeline(self) -> NodeList:
            """Retrieves run configuration.

            Returns:
                (:obj:`Dict`): Run configurations being used by runner.
            """
            return self.node_loader.node_list
    ```

    ```python title="peekingduck.declarative_loader.DeclarativeLoader()" linenums="1"
    class DeclarativeLoader:  # pylint: disable=too-few-public-methods
        """A helper class to create
        :py:class:`Pipeline <peekingduck.pipeline.pipeline.Pipeline>`.

        The declarative loader class creates the specified nodes according to any
        modifications provided in the configs and returns the pipeline needed for
        inference.

        Args:
            pipeline_path (:obj:`pathlib.Path`): Path to a YAML file that
                declares the node sequence to be used in the pipeline.
            config_updates_cli (:obj:`str`): Stringified nested dictionaries of
                configuration changes passed as part of CLI command. Used to modify
                the node configurations directly from the CLI.
            custom_nodes_parent_subdir (:obj:`str`): Relative path to parent
                folder which contains custom nodes that users have created to be
                used with PeekingDuck. For more information on using custom nodes,
                please refer to
                `Getting Started <getting_started/03_custom_nodes.html>`_.
        """

        def __init__(
            self,
            pipeline_path: Path,
            config_updates_cli: str,
            custom_nodes_parent_subdir: str,
        ) -> None:
            self.logger = logging.getLogger(__name__)

            self.pkd_base_dir = Path(__file__).resolve().parent
            self.config_loader = ConfigLoader(self.pkd_base_dir)

            self.node_list = self._load_node_list(pipeline_path)
            self.config_updates_cli = ast.literal_eval(config_updates_cli)

            custom_nodes_name = self._get_custom_name_from_node_list()
            if custom_nodes_name is not None:
                custom_nodes_dir = (
                    Path.cwd() / custom_nodes_parent_subdir / custom_nodes_name
                )
                self.custom_config_loader = ConfigLoader(custom_nodes_dir)
                sys.path.append(custom_nodes_parent_subdir)

                self.custom_nodes_dir = custom_nodes_dir

        def _load_node_list(self, pipeline_path: Path) -> "NodeList":
            """Loads a list of nodes from pipeline_path.yml"""

            # dotw 2022-03-17: Temporary helper methods
            def deprecation_warning(
                name: str, config: Union[str, Dict[str, Any]]
            ) -> None:
                self.logger.warning(
                    f"`{name}` deprecated, replaced by `input.visual`"
                )
                self.logger.warning(f"convert `{name}` to `input.visual`:{config}")

            with open(pipeline_path) as node_yml:
                data = yaml.safe_load(node_yml)
            if not isinstance(data, dict) or "nodes" not in data:
                raise ValueError(
                    f"{pipeline_path} has an invalid structure. "
                    "Missing top-level 'nodes' key."
                )

            nodes = data["nodes"]
            if nodes is None:
                raise ValueError(f"{pipeline_path} does not contain any nodes!")

            upgraded_nodes = []
            for node in nodes:
                if isinstance(node, str):
                    if node in ["input.live", "input.recorded"]:
                        deprecation_warning(node, "input.visual")
                        if node == "input.live":
                            node = {"input.visual": {"source": 0}}
                        else:
                            self.logger.error(
                                "input.recorded with no parameters error!"
                            )
                            node = "input.visual"
                else:
                    if "input.live" in node:
                        node_config = node.pop("input.live")
                        if "input_source" in node_config:
                            node_config["source"] = node_config.pop("input_source")
                        node["input.visual"] = node_config
                        deprecation_warning("input.live", node_config)
                    if "input.recorded" in node:
                        node_config = node.pop("input.recorded")
                        if "input_dir" in node_config:
                            node_config["source"] = node_config.pop("input_dir")
                        node["input.visual"] = node_config
                        deprecation_warning("input.recorded", node_config)
                upgraded_nodes.append(node)

            self.logger.info("Successfully loaded pipeline file.")
            return NodeList(upgraded_nodes)

        def _get_custom_name_from_node_list(self) -> Any:
            custom_name = None

            for node_str, _ in self.node_list:
                node_type = node_str.split(".")[0]

                if node_type not in PEEKINGDUCK_NODE_TYPES:
                    custom_name = node_type
                    break

            return custom_name

        def _instantiate_nodes(self) -> List[AbstractNode]:
            """Given a list of imported nodes, instantiate nodes"""
            instantiated_nodes = []

            for node_str, config_updates_yml in self.node_list:
                node_str_split = node_str.split(".")

                self.logger.info(f"Initializing {node_str} node...")

                if len(node_str_split) == 3:
                    # convert windows/linux filepath to a module path
                    path_to_node = f"{self.custom_nodes_dir.name}."
                    node_name = ".".join(node_str_split[-2:])

                    instantiated_node = self._init_node(
                        path_to_node,
                        node_name,
                        self.custom_config_loader,
                        config_updates_yml,
                    )
                else:
                    path_to_node = "peekingduck.pipeline.nodes."

                    instantiated_node = self._init_node(
                        path_to_node,
                        node_str,
                        self.config_loader,
                        config_updates_yml,
                    )

                instantiated_nodes.append(instantiated_node)

            return instantiated_nodes

        def _init_node(
            self,
            path_to_node: str,
            node_name: str,
            config_loader: ConfigLoader,
            config_updates_yml: Optional[Dict[str, Any]],
        ) -> AbstractNode:
            """Imports node to filepath and initializes node with config."""
            node = importlib.import_module(path_to_node + node_name)
            config = config_loader.get(node_name)

            # First, override default configs with values from pipeline_config.yml
            if config_updates_yml is not None:
                config = self._edit_config(config, config_updates_yml, node_name)

            # Second, override configs again with values from cli
            if self.config_updates_cli is not None:
                if node_name in self.config_updates_cli.keys():
                    config = self._edit_config(
                        config, self.config_updates_cli[node_name], node_name
                    )

            return node.Node(config)

        def _edit_config(
            self,
            dict_orig: Dict[str, Any],
            dict_update: Dict[str, Any],
            node_name: str,
        ) -> Dict[str, Any]:
            """Update value of a nested dictionary of varying depth using recursion"""
            for key, value in dict_update.items():
                if isinstance(value, collections.abc.Mapping):
                    dict_orig[key] = self._edit_config(
                        dict_orig.get(key, {}), value, node_name  # type: ignore
                    )
                else:
                    if key not in dict_orig:
                        self.logger.warning(
                            f"Config for node {node_name} does not have the key: {key}"
                        )
                    else:
                        if key == "detect_ids":
                            key, value = obj_det_change_class_name_to_id(
                                node_name, key, value
                            )

                        dict_orig[key] = value
                        self.logger.info(
                            f"Config for node {node_name} is updated to: '{key}': {value}"
                        )
            return dict_orig

        def get_pipeline(self) -> Pipeline:
            """Returns a compiled
            :py:class:`Pipeline <peekingduck.pipeline.pipeline.Pipeline>` for
            PeekingDuck :py:class:`Runner <peekingduck.runner.Runner>` to execute.
            """
            instantiated_nodes = self._instantiate_nodes()

            try:
                return Pipeline(instantiated_nodes)

            except ValueError as error:
                self.logger.error(str(error))
                sys.exit(1)


    class NodeList:
        """Iterator class to return node string and node configs (if any) from the
        nodes declared in the run config file.
        """

        def __init__(self, nodes: List[Union[Dict[str, Any], str]]) -> None:
            self.nodes = nodes
            self.length = len(nodes)

        def __iter__(self) -> Iterator[Tuple[str, Optional[Dict[str, Any]]]]:
            self.current = -1
            return self

        def __next__(self) -> Tuple[str, Optional[Dict[str, Any]]]:
            self.current += 1
            if self.current >= self.length:
                raise StopIteration
            node_item = self.nodes[self.current]

            if isinstance(node_item, dict):
                node_str = next(iter(node_item))
                config_updates = node_item[node_str]
            else:
                node_str = node_item
                config_updates = None

            return node_str, config_updates
    ```


## The Abstract Node Class

```python title="abstract node class" linenums="1"
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        # return outputs
```

### ABC Class

Notice that all **custom nodes** are derived from the `AbstractNode` class from `peekingduck.pipeline.nodes.node import AbstractNode`. This class is the base class for all custom nodes, as we will soon see.

- `@abstractmethod`: This is a decorator which indicates that the method is abstract. More concretely, once a class inherits from `AbstractNode`, it must implement all the abstract methods. In this case, we need to implement the `run` method in each custom node.

    ```python title="abstract method" linenums="1"
    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """abstract method needed for running node"""
        raise NotImplementedError("This method needs to be implemented")
    ```


### The Run Method

As we previously see, the custom nodes are derived from the `AbstractNode` class. Each custom node must implement the `run` method. **A brief check tells me that the output of the `run` method is a dictionary of strings and values.** This is reasonable since we can see from `line 133` in `runner.py` that `outputs = node.run(inputs)`, and `outputs` is indeed a dictionary of strings and values.

We can also find out what keys the `output` hold by going to `visual.yml` and looking at the `output` key. The logic holds for other default nodes.


## Problems 

### Chaining Yolo and Posenet

- I set all thresholds from Yolo and Posenet to be $0$ to avoid any quality check on the `bboxes` so that I can attempt to debug the issue. 
- The issue turns out to be pretty confusing, when `inputs` were printed in the `dabble.exercise_counter`, it turns out that some `bboxes` were populated but the `bbox_scores` were not. This should not be the case since a bijective relationship should be formed.
- Upon further digging, here is what I found:
    - Using `yolov4tiny`, the model did not manage to predict the `person` class. Thus `outputs` dict consisting of `bboxes, bbox_labels, bbox_scores` were empty in the `model.yolo` portion.
    - We turn our attention to `model.posenet`, it turns out that `bboxes` were again predicted at the Posenet level. This tells us half the story, because `bboxes` variable is now indeed populated (if Posenet says yes) but `bbox_scores` variable is still empty.
    ```python
    bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(
        inputs["img"]
    )
    bbox_labels = np.array(["person"] * len(bboxes))
    bboxes = np.clip(bboxes, 0, 1)

    outputs = {
        "bboxes": bboxes,
        "keypoints": keypoints,
        "keypoint_scores": keypoint_scores,
        "keypoint_conns": keypoint_conns,
        "bbox_labels": bbox_labels,
    }
    ```
    Consequently, when unpacking in the `dabble.exercise_counter.py`, 
    ```python
    img = inputs["img"]
    bboxes = inputs["bboxes"]

    bbox_scores = inputs["bbox_scores"]
    keypoints = inputs["keypoints"]
    keypoint_scores = inputs["keypoint_scores"]
    ```
    The `bboxes` are not empty but `bbox_scores` is, resulting in error.
    - A bit more digging into why the `bboxes` are returned from Posenet. We take a look into `self.model = posenet_model.PoseNetModel(self.config)` which points to `model.posenetv1.posenet_model`, and the `predict` method points to `Predictor` class in `model.posenetv1.postnet_files.predictor`. It turns out the `bbox` is derived from `bbox = self._get_bbox_of_one_pose(pose_coords, pose_masks)` and these are actually the bounding boxes of the `keypoints`. So if there are 17 pairs of keypoints, we simply find the max corners of them. (i.e. imagine the keypoints depicts a human skeleton and we enclose them with bboxes). 
    - Something worth noting is in `model.posenet`, `bbox_labels = np.array(["person"] * len(bboxes))` was coded to depict `person`. 
    - Need to add a clause to catch when yolo does not predict anything handling empty `bbox_scores`. 
    - I believe the purpose of chaining yolo then posenet is to let yolo get the person's bounding box coordinates first, then chain the cropped person to posenet for keypoint predictions... KIV first.


- Created `debug` file for `yolo`, `posenet` and `exercise_counter` as `v4tiny` seem to return errors on index out of range.
- Turns out `yolov4tiny` is not capable of detecting human sometimes, creating empty bboxes and scores, but get overwritten by `bboxes` found in `posenet`, which is a bit different from the `bboxes` in yolo.
- In the pipeline where we chain `posenet` after `yolo`, I created another debug file before `exercise_counter` and confirmed that if both `yolo` and `posenet` have `bbox` values, the latter `posenet` will overwrite the one from `yolo`. This setting is 

```python
nodes:
- input.visual:
    source: https://storage.googleapis.com/reighns/peekingduck/videos/push_ups.mp4
- model.yolo:
    model_type: "v4" # "v4tiny"
    iou_threshold: 0.1
    score_threshold: 0.1
    detect_ids: ["person"] # [0]
    num_classes: 1
- custom_nodes.dabble.debug_yolo
- model.posenet:
    model_type: "resnet"
    resolution: {height: 224, width: 224}
    score_threshold: 0.05
- custom_nodes.dabble.debug_posenet
- dabble.fps
- custom_nodes.dabble.debug_exercise_counter
- custom_nodes.dabble.exercise_counter
- draw.poses
# - model.mtcnn
# - draw.mosaic_bbox
- draw.legend:
    show: ["fps"]
- output.screen
```