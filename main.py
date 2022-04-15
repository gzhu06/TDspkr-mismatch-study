import os
import json
from argparse import ArgumentParser
from configuration import get_config

config, parser = get_config()

def saveparser(params_file, parser):
    
    if not os.path.exists(config.train_path):
        os.mkdir(config.train_path)
        
    args = parser.parse_args()
    with open(params_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    parameters = vars(parser.parse_args())
    return parameters

if __name__ == '__main__':
    
    parser.add_argument("-stage", type=float, default=1, help="stage of development")
    config, unparsed = parser.parse_known_args()
    
    # create parameter file
    params_file = os.path.join(config.train_path, 'parameters.json')
    os.makedirs(config.train_path, exist_ok=True)
    parser_exp = saveparser(params_file, parser)
    
    # feature stage
    if config.stage == 0:
        run_cmd = 'python preprocessing/wavform_extract.py ' + '-config_file ' + params_file

    # apex train stage
    if config.stage == 1:
        run_cmd = 'python trainer/train_apex.py ' + '-config_file ' + params_file

    if config.stage == 4:
        run_cmd = 'python evaluate/extract_vector.py ' + '-config_file ' + params_file
    
    print(run_cmd)
    os.system(run_cmd)
