import os
import numpy as np
from datetime import datetime
import json
import argparse
import luigi
from cluster_tools.inference import InferenceLocal, InferenceSlurm


def parse_line(line, block_dict):
    line = line.split()
    if ''.join(line[2:5]) == 'startprocessingblock':
        type_ = 'start'
        block_id = int(line[5])
    elif ''.join(line[2:4]) == 'processedblock':
        type_ = 'stop'
        block_id = int(line[4])
    else:
        return block_dict

    time_stamp = line[0] + ' ' + line[1][:-1]
    time_stamp = datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S.%f')

    res_dict = block_dict.get(block_id, {})
    res_dict[type_] = time_stamp

    block_dict[block_id] = res_dict
    return block_dict


def evaluate_time(tmp_folder):
    log_file = os.path.join(tmp_folder, 'logs', 'inference_0.log')

    block_dict = {}

    with open(log_file) as f:
        for line in f:
            block_dict = parse_line(line, block_dict)

    times = []
    for res in block_dict.values():
        start = res['start']
        stop = res['stop']
        diff = (stop - start)
        diff = diff.seconds + diff.microseconds * 1e-6
        times.append(diff)

    print("Mean iteration time:", np.mean(times), "+-", np.std(times))
    print("Max iteration time:", np.max(times))
    print("Min iteration time:", np.min(times))
    print("Sum iteration times:", np.sum(times))


def run_prediction(weigth_folder, gpu, local, mixed_precision):
    task = InferenceLocal if local else InferenceSlurm
    name = os.path.split(weigth_folder)[1]
    ckpt = os.path.join(weigth_folder, 'Weights', 'model.nn')
    assert os.path.exists(ckpt)

    tmp_folder = os.path.join('tmp_folders', name)

    config_dir = os.path.join(tmp_folder, 'config')
    os.makedirs(config_dir, exist_ok=True)

    block_shape = [32, 256, 256]
    halo = [8, 32, 32]
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch13/bin/python'

    conf = task.default_global_config()
    conf.update({'block_shape': block_shape, 'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    gpu_type = name.split('_')[0]
    assert gpu_type in ('1080Ti', '2080Ti', 'P40'), gpu_type

    device_mapping = {gpu: 0} if local else None

    conf = task.default_task_config()
    conf.update({'gpu_type': gpu_type, 'use_best': False, 'mixed_precision': mixed_precision,
                 'device_mapping': device_mapping, 'threads_per_job': 6, 'mem_limit': 32})
    with open(os.path.join(config_dir, 'inference.config'), 'w') as f:
        json.dump(conf, f)

    input_path = '../data/sample_A_padded_20160501.hdf'
    output_path = os.path.join(tmp_folder, 'out.n5')

    out_key_dict = {'prediction': [0, 1]}

    t = task(max_jobs=1, tmp_folder=tmp_folder, config_dir=config_dir,
             input_path=input_path, input_key='volumes/raw',
             output_path=output_path, output_key=out_key_dict,
             checkpoint_path=ckpt, halo=halo, framework='pytorch')
    luigi.build([t], local_scheduler=True)

    evaluate_time(tmp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_folder', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--local', default=1, type=int)
    parser.add_argument('--mixed_precision', default=0, type=int)
    args = parser.parse_args()
    run_prediction(args.weight_folder, args.gpu,
                   bool(args.local), bool(args.mixed_precision))
