from mf_module.utils.class_builder import ClassBuilder
from mf_module.model.model import MotionDetectorModel

def build_default_motion_detector(args):
    model = MotionDetectorModel(**args)
    return model


class MotionDetectionModelBuilder(ClassBuilder):
    all_class = {}
    def register_all(self):
        self.register("default", build_default_motion_detector)
