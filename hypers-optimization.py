import numpy as np
import pandas as pd
np.random.seed(1234)
import matplotlib.pyplot as plt
import os
import sys
import subprocess
from skopt import Optimizer as SKOptimizer
from skopt.space import Categorical, Integer, Real
from skopt import dump as skdump
from skopt import load as skload

def get_random_params_sample(params_grid, params_grid_sample_keys):
    size = len(params_grid_sample_keys)
    sample = []
    for k in params_grid_sample_keys:
        l = len(params_grid[k]["values"])
        r = np.random.randint(0, l, (1))[0]
        s = params_grid[k]["values"][r]
        sample.append(s)
    return sample

def set_new_params_lines(exapmle_params_lines, params_sample, params_grid, params_grid_sample_keys):
    new_lines = []
    hypers_rs_lines = False
    for line in exapmle_params_lines:
        new_line = line
        
        if "hypers_rs" in line:
            hypers_rs_lines = True
        if hypers_rs_lines:
            if line.strip() == "},":
                hypers_rs_lines = False

        if len(line.split(":")) == 2 and not hypers_rs_lines:
            s1, s2 = line.split(":")
            q1, q2, q3 = s1.split("\"")
            if q2 in params_grid.keys():
                dtype = params_grid[q2]["dtype"]
                value_to_set = None
                if q2 in params_grid_sample_keys:
                    index = np.argwhere(np.array(params_grid_sample_keys) == q2).reshape(1)[0]
                    value_to_set = params_sample[index]
                else:
                    value_to_set = params_grid[q2]["values"][0]
                if dtype == np.float or dtype == np.int:
                    value_to_set = str(value_to_set)
                elif dtype == np.bool:
                    value_to_set = str(value_to_set).lower()
                else: # dtype == np.str
                    value_to_set = "\"" + value_to_set + "\""
                s2 = " " + value_to_set + ",\n"
            new_line = ":".join([s1, s2])
        new_lines.append(new_line)
    return new_lines

def make_prefix(params_sample, params_grid_sample_keys, short_names_map):
    prefix = "_".join(["_".join([str(short_names_map[k]), str(short_names_map.get(params_sample[i], params_sample[i]))]) for i, k in enumerate(params_grid_sample_keys)])
    prefix = prefix.replace(".", "-")
    return prefix

def write_new_params_json(exapmle_params_file_name, params_sample, params_grid, params_grid_sample_keys, folder=os.getcwd()):
    exapmle_params_file = open(exapmle_params_file_name, 'r')
    exapmle_params_lines = exapmle_params_file.readlines()
    exapmle_params_file.close()

    new_lines = set_new_params_lines(exapmle_params_lines, params_sample, params_grid, params_grid_sample_keys)
    prefix = params_grid["prefix"]["values"][0]

    new_params_file_name = os.path.join(folder, f"{prefix}.json")
    new_params_file = open(new_params_file_name, 'w')
    new_params_file.writelines(new_lines)
    new_params_file.close()
    # print(f"New params json file \"{prefix}.json\" is written!")

def to_wrapped_hypers_space(hypers_sample, params_grid_sample_keys):
    hypers_sample_wrapped = hypers_sample.copy()
    for i, k in enumerate(params_grid_sample_keys):
        dtype = params_grid[k]["dtype"]
        if dtype != np.float and dtype != np.int:
            continue
        hypers_space_k = list(range(len(params_grid[k]["values"])))
        index = np.argwhere(np.array(hypers_space_k) == hypers_sample[i]).reshape(1)[0]
        hypers_sample_wrapped[i] = params_grid[k]["values"][index]
    return hypers_sample_wrapped

def to_hypers_space(hypers_sample_wrapped, params_grid_sample_keys):
    hypers_sample = hypers_sample_wrapped.copy()
    for i, k in enumerate(params_grid_sample_keys):
        dtype = params_grid[k]["dtype"]
        if dtype != np.float and dtype != np.int:
            continue
        hypers_sample_wrapped_k = params_grid[k]["values"]
        index = np.argwhere(np.array(hypers_sample_wrapped_k) == hypers_sample_wrapped[i]).reshape(1)[0]
        hypers_sample[i] = list(range(len(params_grid[k]["values"])))[index]
    return hypers_sample

def test_to__space_functions(params_grid, params_grid_sample_keys):
    # test "to_..._space()" functions
    test_is_passed = True
    for i in range(10):
        hypers_sample_w_0 = get_random_params_sample(params_grid, params_grid_sample_keys)
        hypers_sample_0 = to_hypers_space(hypers_sample_w_0, params_grid_sample_keys)
        hypers_sample_w_1 = to_wrapped_hypers_space(hypers_sample_0, params_grid_sample_keys)
        hypers_sample_1 = to_hypers_space(hypers_sample_w_1, params_grid_sample_keys)
        hypers_sample_w_2 = to_wrapped_hypers_space(hypers_sample_1, params_grid_sample_keys)

        if hypers_sample_w_0 != hypers_sample_w_1 or hypers_sample_w_0 != hypers_sample_w_2 or hypers_sample_0 != hypers_sample_1:
            test_is_passed = False
            print("hypers_sample_w_0 = ", hypers_sample_w_0)
            print("hypers_sample_w_1 = ", hypers_sample_w_1)
            print("hypers_sample_w_2 = ", hypers_sample_w_2)
            print("hypers_sample_0   = ", hypers_sample_0)
            print("hypers_sample_1   = ", hypers_sample_1)
    if test_is_passed:
        print("Test is passed!")
    else:
        print("Test is NOT passed!")


params_grid = {
    "prefix": {"values": ["example"], "dtype": np.str},
    "n_train": {"values": [1000], "dtype": np.int},
    "learning_rate": {"values": [0.1], "dtype": np.float},
    "n_combined_basis": {"values": [4, 6, 8, 10, 12], "dtype": np.int},
    "max_radial": {"values": [4, 6, 8, 10, 12], "dtype": np.int},
    "composition_regularizer": {"values": [1e-2, 1e-1, 1e0], "dtype": np.float},
    "radial_spectrum_regularizer": {"values": [1e-2, 1e-1, 1e0], "dtype": np.float},
    "power_spectrum_regularizer": {"values": [1e-2, 1e-1, 1e0], "dtype": np.float},
    "power_spectrum_combiner_regularizer": {"values": [0.0, 1e-4, 1e-2, 1e0], "dtype": np.float},
    "nn_layer_size": {"values": [0], "dtype": np.int},
    # "combiner": {"values": ["CombineRadialSpecies", "CombineRadialSpeciesWithAngular",
    #              "CombineRadialSpeciesWithAngularAdaptBasis", "CombineRadialSpeciesWithAngularAdaptBasisRadial"], "dtype": np.str},
    "combiner": {"values": ["CombineRadialSpecies", "CombineRadialSpeciesWithAngular"], "dtype": np.str},
    "do_gradients": {"values": [False], "dtype": np.bool}
}

# sort numerical values in params_grid
[params_grid[p]["values"].sort() for p in params_grid if params_grid[p]["dtype"] == np.float or params_grid[p]["dtype"] == np.int]

short_names_map = {
    "n_train": "ts",
    "n_combined_basis": "ncb",
    "max_radial": "mr",
    "composition_regularizer": "comp_reg",
    "radial_spectrum_regularizer": "rs_reg",
    "power_spectrum_regularizer": "ps_reg",
    "power_spectrum_combiner_regularizer": "psc_reg",
    "nn_layer_size": "nn",
    "learning_rate": "lr",
    "combiner": "comb",
    "CombineRadialSpecies": "RS",
    "CombineRadialSpeciesWithAngular": "RSL",
    "CombineRadialSpeciesWithAngularAdaptBasis": "RSLAB",
    "CombineRadialSpeciesWithAngularAdaptBasisRadial": "RSLABR"
}

params_grid_sample_keys = ["n_combined_basis", "max_radial", "combiner", "composition_regularizer", "radial_spectrum_regularizer",
    "power_spectrum_regularizer", "power_spectrum_combiner_regularizer"]

# set hypers optimizer space
hypers_space = [
            Integer(low=0, high=len(params_grid[k]["values"])-1, name=k) if params_grid[k]["dtype"] == np.float or params_grid[k]["dtype"] == np.int
            else Categorical(params_grid[k]["values"], name=k)
            for k in params_grid_sample_keys
        ]

os.makedirs(os.path.join(os.getcwd(), "json_files"), exist_ok=True)
path_to_folder_with_fit_script = os.path.join(os.getcwd(), "..", "alchemical-learning")
exapmle_params_file_name = os.path.join(os.getcwd(), "example.json")

np.random.seed(1234)
hypers_opt = SKOptimizer(hypers_space, "GP", acq_func="EI",
                acq_optimizer="sampling",
                initial_point_generator="lhs", n_initial_points=5)
next_sample = hypers_opt.ask()
f_vals = []
for i in range(100):
    next_sample_w = to_wrapped_hypers_space(next_sample, params_grid_sample_keys)
    print("next_sample_w = ", next_sample_w)

    # save params json file using next_sample_w
    ## make prefix
    prefix = make_prefix(next_sample_w, params_grid_sample_keys, short_names_map)
    params_grid["prefix"]["values"][0] = prefix
    # print("prefix = ", prefix)

    ## write new params json
    write_new_params_json(exapmle_params_file_name, next_sample_w, params_grid, params_grid_sample_keys, folder=os.path.join(os.getcwd(), "json_files"))

    # run fit-alchemical-potential.py script
    script_py_path = str(os.path.join(path_to_folder_with_fit_script, "fit-alchemical-potential.py"))
    data_xyz_path = str(os.path.join(path_to_folder_with_fit_script, "data", "data_shuffle.xyz"))
    params_json_path = str(os.path.join(os.getcwd(), "json_files", prefix + ".json"))
    # os.system(f"python3 {script_py_path} {data_xyz_path} {params_json_path} --device cpu").wait()
    
    cmd = ['python', script_py_path, data_xyz_path, params_json_path, "--device", "cpu"]
    # output = subprocess.Popen(cmd).wait()
    # print("output = ", output)

    result = subprocess.run(cmd, stderr = subprocess.PIPE, text = True)
    print(result.stderr)
    
    # find f_val as a min of test_mae
    f_val = None
    for folder in os.listdir(os.getcwd()):
        if prefix in folder:
            df = pd.read_table(os.path.join(os.getcwd(), folder, "epochs.dat"), sep="\s+")
            f_val = df.test_mae.min()
            del df
            break

    res = hypers_opt.tell(next_sample, f_val)
    next_sample = hypers_opt.ask()
    f_vals.append(f_val)

    while next_sample in res.x_iters:
        print("Warning: Next point has been visited. Reset!")
        f_val = res.func_vals[np.min(np.argwhere([v.tolist() == [str(s) for s in next_sample] for v in np.array(res.x_iters)]))]
        res = hypers_opt.tell(next_sample, f_val)
        next_sample = hypers_opt.ask()
    
    break