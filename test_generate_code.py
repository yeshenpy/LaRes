import inspect
function_code = inspect.getsource(compute_reward)

# print(function_code)
import numpy as np
import ast
from numpy import array
import pickle


def is_pure_float(value):
    # 判断是否是float类型且不是numpy数组或list
    if isinstance(value, float):
        return True
    elif isinstance(value, np.ndarray):  # 如果是numpy数组类型
        return False
    elif isinstance(value, list):  # 如果是list类型
        return False
    return False


import argparse
parse = argparse.ArgumentParser()
parse.add_argument('path', type=str)
args = parse.parse_args()
with open(args.path, 'rb') as f:
    temp_org_info_list = pickle.load(f)

tree = ast.parse(function_code)

input_name = None
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        function_name = node.name
        args = [arg.arg for arg in node.args.args]
        if function_name == "compute_reward":
            input_name = args
#        print(f'函数名：{function_name}')
#        print(f'输入参数：{args}')
#        print('---')

for temp_org_info in temp_org_info_list:
    input = []
    for name in input_name:
        input.append(temp_org_info[name])

    res = compute_reward(*input)

    if is_pure_float(res[0]) is False:
        print("Error: The return of the reward function does not follow the original format. Please ensure that the first element of the return (i.e., reward) is a scalar, not any other type of element.")
        exit()

    #print("Results", res)
print("Success!")