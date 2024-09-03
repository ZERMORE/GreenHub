import sys
import os
import subprocess


def run_model(sumocfg_file):
    # 读取并处理 .sumocfg 文件
    with open(sumocfg_file, 'r', encoding='utf-8') as f:
        config = f.read()
    # 假设你的模型是使用SUMO仿真工具进行仿真
    # 调用SUMO命令行进行仿真并将结果保存到输出目录
    output_dir = os.path.join(os.path.dirname(sumocfg_file), 'output')
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(['sumo', '-c', sumocfg_file, '--output-prefix', output_dir], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"SUMO simulation failed with error: {result.stderr}")

    # 可以在此处处理仿真结果文件，并返回感兴趣的数据
    # 例如，假设结果文件输出为 output/output.txt
    with open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8') as f:
        f.write("Simulation complete. Results are saved in the output directory.\n")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Please provide the .sumocfg file path as the argument.")
    sumocfg_file = sys.argv[1]
    run_model(sumocfg_file)

# import sys
#
# def run_model(input_file, output_file):
#     with open(input_file, 'r') as f:
#         data = f.readlines()
#
#     lane1 = int(data[0].strip())
#     lane2 = int(data[1].strip())
#     lane3 = int(data[2].strip())
#
#     # 假设模型是计算车道流量的总和
#     total_traffic = lane1 + lane2 + lane3
#
#     # 将输出写入指定的输出文件
#     with open(output_file, 'w') as f:
#         f.write(f"Total Traffic: {total_traffic}\n")
#
#     return total_traffic
#
# if __name__ == '__main__':
#     # 从命令行参数获取文件名
#     input_filename = sys.argv[1]
#     output_filename = sys.argv[2]
#     run_model(input_filename, output_filename)