#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import os
import sys

sys.path.append('.')

from modeling.utils import (
        build_model_id,
        build_model_path,
        setup_model_dir,
        load_model_json,
        setup_logging, 
        save_model_info,
        ModelConfig)
import modeling.parser

def main(args):
    model_id = build_model_id(args)

    model_path = build_model_path(args, model_id)

    setup_model_dir(args, model_path)

    json_cfg = load_model_json(args, x_train=None, n_classes=None)
    config = ModelConfig(**json_cfg)

    sys.path.append(args.model_dir)
    import model
    from model import fit

    model.fit(config)

if __name__ == '__main__':
    parser = modeling.parser.build_keras()
    sys.exit(main(parser.parse_args()))
