# A demo program of MPIIGaze and MPIIFaceGaze

With this program, you can runs gaze estimation on images and videos.
By default, the video from a webcam is used.

![MPIIGaze video result](figures/mpiigaze_video.gif)
![MPIIFaceGaze video result](figures/mpiifacegaze_video.gif)

(The original video is from [this public domain](https://www.pexels.com/video/woman-in-a-group-having-a-drink-while-listening-3201742/).)

![MPIIGaze image result](figures/mpiigaze_image.jpg)

(The original image is from [this public domain](https://www.pexels.com/photo/photography-of-a-beautiful-woman-smiling-1024311/).)

To train a model, use [this repository](https://github.com/hysts/pytorch_mpiigaze).

## Quick start

### Installation

```bash
pip install ptgaze
```


### Run demo

```bash
ptgaze --mode eye
```


### Usage


```
usage: ptgaze [-h] [--config CONFIG] [--mode {eye,face}]
              [--face-detector {dlib,face_alignment_dlib,face_alignement_sfd}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO]
              [--camera CAMERA] [--output-dir OUTPUT_DIR] [--ext {avi,mp4}]
              [--no-screen] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file for YACS. When using a config file, all
                        the other commandline arguments are ignored. See https
                        ://github.com/hysts/pytorch_mpiigaze_demo/configs/demo
                        _mpiigaze.yaml
  --mode {eye,face}     With 'eye', MPIIGaze model will be used. With 'face',
                        MPIIFaceGaze model will be used. (default: 'eye')
  --face-detector {dlib,face_alignment_dlib,face_alignement_sfd}
                        The method used to detect faces and find face
                        landmarks (default: 'dlib')
  --device {cpu,cuda}   Device used for model inference.
  --image IMAGE         Path to an input image file.
  --video VIDEO         Path to an input video file.
  --camera CAMERA       Camera calibration file. See https://github.com/hysts/
                        pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.
                        yaml
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        If specified, the overlaid video will be saved to this
                        directory.
  --ext {avi,mp4}, -e {avi,mp4}
                        Output video file extension.
  --no-screen           If specified, the video is not displayed on screen,
                        and saved to the output directory.
  --debug
```

While processing an image or video, press the following keys on the window
to show or hide intermediate results:

* `l`: landmarks
* `h`: head pose
* `t`: projected points of 3D face model
* `b`: face bounding box


## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)



