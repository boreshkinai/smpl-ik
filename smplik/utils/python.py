from typing import ClassVar


def get_full_class_reference(c: ClassVar):
    module = c.__module__
    if module is None or module == str.__class__.__module__:
        return c.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + c.__name__
