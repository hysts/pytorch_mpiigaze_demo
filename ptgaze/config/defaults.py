from yacs.config import CfgNode

config = CfgNode()

# option: MPIIGaze, MPIIFaceGaze
config.mode = 'MPIIGaze'
config.device = 'cuda'

# transform
config.transform = CfgNode()
config.transform.mpiifacegaze_face_size = 224
config.transform.mpiifacegaze_gray = False

config.model = CfgNode()
config.model.name = 'resnet_preact'
config.model.backbone = CfgNode()
config.model.backbone.name = 'resnet_simple'
config.model.backbone.pretrained = 'resnet18'
config.model.backbone.resnet_block = 'basic'
config.model.backbone.resnet_layers = [2, 2, 2]

# Face detector
config.face_detector = CfgNode()
# options: dlib, face_alignment_dlib, face_alignment_sfd
config.face_detector.mode = 'dlib'
config.face_detector.dlib = CfgNode()
config.face_detector.dlib.model = ''

# Gaze estimator
config.gaze_estimator = CfgNode()
config.gaze_estimator.checkpoint = ''
config.gaze_estimator.camera_params = ''
config.gaze_estimator.normalized_camera_params = ''
config.gaze_estimator.normalized_camera_distance = 0.6

# demo
config.demo = CfgNode()
config.demo.use_camera = True
config.demo.display_on_screen = True
config.demo.wait_time = 1
config.demo.image_path = ''
config.demo.video_path = ''
config.demo.output_dir = ''
config.demo.output_file_extension = 'avi'
config.demo.head_pose_axis_length = 0.05
config.demo.gaze_visualization_length = 0.05
config.demo.show_bbox = True
config.demo.show_head_pose = False
config.demo.show_landmarks = False
config.demo.show_normalized_image = False
config.demo.show_template_model = False


def get_default_config():
    return config.clone()
