#!/usr/bin/env python
# Copyright (c) Institute of Artificial Intelligence (TeleAI), China Telecom, 2025. All Rights Reserved.

import os 

__version__ = "0.0.1"

def root_dir() -> str:
    current_path = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_path), ".."))
    return project_root
