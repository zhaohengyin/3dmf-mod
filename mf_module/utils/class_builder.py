import inspect

def check_type(obj):
    if inspect.isclass(obj):
        return "class"
    elif inspect.isfunction(obj):
        return "function"

class ClassBuilder:
    all_class = {}

    def __init__(self, cls_str="type", arg_str="args"):
        self.cls_str = cls_str
        self.arg_str = arg_str
        self.register_all()
        return 

    def register_all(self):
        '''
            to be implemented by user.
        '''
        return 

    def register(self, name, register_fn_or_class):
        self.all_class[name] = register_fn_or_class

    def build(self, args):
        class_name = args[self.cls_str].lower()
        if class_name not in self.all_class:
            raise ValueError(f"Class {class_name} is not registered. Available: {list(self.all_class.keys())}")
        
        result_cls_builder = self.all_class[class_name]
        if check_type(result_cls_builder) == "class":
            return result_cls_builder(**args[self.arg_str])
        else:
            return result_cls_builder(args[self.arg_str])
