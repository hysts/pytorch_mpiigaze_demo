# A demo program of gaze estimation models (MPIIGaze, MPIIFaceGaze, ETH-XGaze)

[![PyPI version](https://badge.fury.io/py/ptgaze.svg)](https://pypi.org/project/ptgaze/)
[![Downloads](https://pepy.tech/badge/ptgaze)](https://pepy.tech/project/ptgaze)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hysts/pytorch_mpiigaze_demo/blob/master/demo.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hysts/pytorch_mpiigaze_demo.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze_demo)

This program runs gaze estimation on images and videos. By default, it uses the video from a webcam.

![ETH-XGaze video01 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video01.gif)
![ETH-XGaze video02 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video02.gif)
![ETH-XGaze video03 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video03.gif)

![MPIIGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_video00.gif)
![MPIIFaceGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiifacegaze_video00.gif)

![MPIIGaze image00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_image00.jpg)

To train models on MPIIGaze and MPIIFaceGaze, use [this repository](https://github.com/hysts/pytorch_mpiigaze). You can also use [this repository](https://github.com/hysts/pl_gaze_estimation) to train a model on the ETH-XGaze dataset.

## Quick start

This program has only been tested on Ubuntu.

### Installation

```bash
pip install ptgaze
```


### Run demo

```bash
ptgaze --mode eth-xgaze
```


### Usage


```
usage: ptgaze [-h] [--config CONFIG]
              [--mode {mpiigaze,mpiifacegaze,eth-xgaze}]
              [--face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO]
              [--camera CAMERA] [--output-dir OUTPUT_DIR] [--ext {avi,mp4}]
              [--no-screen] [--debug]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Config file. When using a config file, all the other
                        commandline arguments are ignored. See
                        https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/configs/eth-xgaze.yaml
  --mode {mpiigaze,mpiifacegaze,eth-xgaze}
                        With 'mpiigaze', MPIIGaze model will be used. With
                        'mpiifacegaze', MPIIFaceGaze model will be used. With
                        'eth-xgaze', ETH-XGaze model will be used.
  --face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}
                        The method used to detect faces and find face
                        landmarks (default: 'mediapipe')
  --device {cpu,cuda}   Device used for model inference.
  --image IMAGE         Path to an input image file.
  --video VIDEO         Path to an input video file.
  --camera CAMERA       Camera calibration file. See
                        https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/calib/sample_params.yaml
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        If specified, the overlaid video will be saved to this
                        directory.
  --ext {avi,mp4}, -e {avi,mp4}
                        Output video file extension.
  --no-screen           If specified, the video is not displayed on screen,
                        and saved to the output directory.
  --debug
```

While processing an image or video, press the following keys in the window to show or hide intermediate results:

- `l`: landmarks
- `h`: head pose
- `t`: projected points of the 3D face model
- `b`: face bounding box
- `n`: normalized images fed to the model

## Pretrained models

When run with `--mode`, the program downloads the pretrained weights from the Hugging Face Hub into the Hugging Face cache directory on first use:

| Mode | Model repo |
|------|------------|
| `mpiigaze` | [hysts/ptgaze-mpiigaze-resnet-preact](https://huggingface.co/hysts/ptgaze-mpiigaze-resnet-preact) |
| `mpiifacegaze` | [hysts/ptgaze-mpiifacegaze-resnet-simple](https://huggingface.co/hysts/ptgaze-mpiifacegaze-resnet-simple) |
| `eth-xgaze` | [hysts/ptgaze-eth-xgaze-resnet18](https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18) |

The weights are stored in the safetensors format, and ptgaze loads safetensors checkpoints only. If you have a `.pth` checkpoint trained with [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze), convert it once and point `gaze_estimator.checkpoint` in your config file to the converted file:

```bash
ptgaze-convert checkpoint.pth model.safetensors --mode mpiigaze
```

`ptgaze-convert` accepts both the `{"model": state_dict}` checkpoint layout and a bare state dict. With `--mode`, it also validates the state dict against the architecture used by that mode and records the mode in the file metadata.

## Notes on data normalization

The preprocessing in this repository follows the data normalization scheme that was used to create the training data of each model ([Zhang et al., TPAMI 2019, Section 4.2](https://arxiv.org/abs/1711.09017), [Zhang et al., ETRA 2018](https://perceptualui.org/publications/zhang18_etra.pdf)). A few details are easy to miss:

- The normalized camera parameters in `ptgaze/data/normalized_camera_params/` may look different from the values reported in the papers, but they are equivalent. The image warp depends only on the ratio of the focal length to the normalized distance, so, for example, the face parameters (focal length 1600, distance 1.0 m, 224x224 pixels) are equivalent to the ETH-XGaze normalization (960, 300 mm, 448x448) after downscaling. The values match the reference C++ implementation by the authors of the papers, [OpenGaze](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze).
- The normalized head pose fed to the MPIIGaze model follows the dataset annotation convention: the pitch and yaw of the z-axis of the normalized head rotation matrix.
- The predicted gaze vector is converted back to the camera coordinate system with the rotation only, following the modified normalization of Zhang et al. (ETRA 2018) and OpenGaze. Note that the gaze labels of the MPIIGaze and MPIIFaceGaze datasets were generated with the original normalization, which also applies the scaling matrix to the gaze vector, so the outputs of these two models can contain a small systematic bias (on the order of a degree) when interpreted as 3D directions. The ETH-XGaze dataset uses the modified normalization and is not affected.
- These conventions are covered by the regression tests in `tests/`, which compare the implementation against the label-generation code of the datasets.


## References

- Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." In European Conference on Computer Vision (ECCV), 2020. [arXiv:2007.15837](https://arxiv.org/abs/2007.15837), [Project Page](https://ait.ethz.ch/projects/2020/ETH-XGaze/), [GitHub](https://github.com/xucong-zhang/ETH-XGaze)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE Transactions on Pattern Analysis and Machine Intelligence 41 (2019). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
- Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)
