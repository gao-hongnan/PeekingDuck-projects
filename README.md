<div align="center">
<h1>PeekingDuck Projects</a></h1>
by Hongnan Gao
1st May, 2022
<br>
</div>


<h4 align="center">
  <a href="https://reighns92.github.io/PeekingDuck-projects/workflows/">Workflows</a>
  <span> · </span>
  <a href="https://reighns92.github.io/PeekingDuck-projects/exercise_counter/">Exercise Counter Tutorial</a>
  <span> · </span>
  <a href="https://reighns92.github.io/PeekingDuck-projects/melanoma_gradcam/">Melanoma Prediction with Grad-CAM Tutorial</a>
</h4>

---

## Introduction

> PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV) inference. The name "PeekingDuck" is a play on: "Peeking" in a nod to CV; and "Duck" in duck typing. - Extracted from [PeekingDuck](https://github.com/aimakerspace/PeekingDuck).

This project uses the PeekingDuck framework to create two use cases:

- **Exercise Counter**: we will create custom node(s) using the PeekingDuck Framework to count the number of times a user has performed a push-up. The example can be extended to other exercises such as sit-ups, pull-ups and other general exercises by altering the **counting logic**.
- **Melanoma Prediction with Grad-CAM**: we will create custom node(s) using the PeekingDuck Framework to predict the presence of melanoma in an image and output the image with Grad-CAM highlights.

## Installation

Install this project using the following command:

```bash
git clone                       # clone or download the repo to your working dir
pip install -e .                # install the dependencies
cd custom_hn_exercise_counter   # change to the project directory
peekingduck run                 # run the project
```

Note that this is only tested on Ubuntu-latest and Windows-latest with python version 3.8/3.9 through GitHub Actions.

For a more detailed workflow, see the [Workflows](https://reighns92.github.io/PeekingDuck-projects/workflows/) section.

## Tutorials

The tutorials below walkthrough how to use the PeekingDuck framework to create custom nodes for the two use cases.

- [PeekingDuck Exercise Counter](https://reighns92.github.io/PeekingDuck-projects/exercise_counter/)
- [PeekingDuck Grad-CAM](https://reighns92.github.io/PeekingDuck-projects/melanoma_gradcam/)

## Gallery

### Push-up Counter Demo

Seeing push-up counter in action: person doing a total of $7$ push-ups - photo taken by me. The counter is displayed in yellow font on the top left corner of the video/gif.
 
  ![Demo Count Push Ups](https://storage.googleapis.com/reighns/peekingduck/videos/seven_push_ups_by_jun_demo_gif.gif)

Below is a download link for the push-up counter demo. There are two demos, one is the one above, and the other is a video from [YouTube](https://www.youtube.com/watch?v=1D_HvjxB3Ps), uploaded by user Pedro Neto.

- [Download Push Up Demo Zip](https://storage.googleapis.com/reighns/peekingduck/videos/push_ups_demo_zip.zip)


### Melanoma Grad-CAM Demo

The demo of Melanoma Grad-CAM demo is shown below:

<img src="https://storage.googleapis.com/reighns/peekingduck/images/gradcam_demo.PNG" width="566" height="350">



## References

This project took references from the [Official PeekingDuck Tutorials](https://peekingduck.readthedocs.io/en/stable/).
