from copy import deepcopy

from .base_constructors import get_class2dict, get_dict2class


class Pickler:
    def __init__(self, deleted_attrs=[], complex_attr_definition_dict={}):
        """
        Helper for pickling and un-pickling classes in .sorf_calculation and .models.
        """
        self.deleted_attrs = deleted_attrs
        self.complex_attr_definition_dict = complex_attr_definition_dict

    def getstate(self, input_obj):
        d = input_obj.__dict__

        output = {}
        for attr, value in d.items():
            if attr in self.deleted_attrs:
                new_value = None
            elif (attr in self.complex_attr_definition_dict) and (value is not None):
                new_value = get_class2dict(self.complex_attr_definition_dict[attr])(value)
            else:
                new_value = value
            output[attr] = new_value
        return output

    def state_dict(self, d):
        output = deepcopy(d)
        for attr, attr_definition in self.complex_attr_definition_dict.items():
            converter = get_dict2class(attr_definition)
            if output[attr] is not None:
                output[attr] = converter(output[attr])
        return output
