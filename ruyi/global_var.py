#!/usr/bin/env python
# Copyright (c) Institute of Artificial Intelligence (TeleAI), China Telecom, 2025. All Rights Reserved.

def _init():
    global _global_dict
    _global_dict = {}


def set_global_val(key, value):
    if '_global_dict' not in globals():
        _init()
    _global_dict[key] = value


def get_global_val(key, default_value=None):
    try:
        return _global_dict[key]
    except (KeyError, NameError):
        return default_value
