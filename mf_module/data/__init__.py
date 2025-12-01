from mf_module.utils.class_builder import ClassBuilder
from mf_module.data.processor import MotionDataProcessor
from mf_module.data.randomizer import MotionDataRandomizer


def build_default_motion_detection_parser(args):
    randomizer = MotionDataRandomizer()
    return MotionDataProcessor(randomizer, static_camera=True)


class MotionDetectionParserBuilder(ClassBuilder):
    all_class = {}
    
    def register_all(self):
        self.register("default", build_default_motion_detection_parser)
