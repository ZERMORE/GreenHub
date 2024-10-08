from flask import Flask, request, jsonify, render_template
import subprocess
import uuid
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        data = request.json
        lane1 = data['lane1']
        lane2 = data['lane2']
        lane3 = data['lane3']

        # 生成唯一的目录名
        unique_id = uuid.uuid4()
        directory = f'run_{unique_id}'

        # 创建目录
        os.makedirs(directory, exist_ok=True)

        # 定义输入和输出文件的路径
        route_filename = os.path.join(directory, 'flow.rou.xml')
        sumocfg_filename = os.path.join(directory, 'config.sumocfg')

        # 写入 .rou.xml 文件
        with open(route_filename, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<routes>\n")
            f.write(
                '    <vType id="ACCType" color="22,255,255" length="5.0" carFollowModel="IDM" laneChangeModel="LC2013" lcSpeedGain="1"/>\n')
            f.write(
                '    <vType id="CACCType" color="22,55,255" length="5.0" carFollowModel="IDM" laneChangeModel="LC2013" lcSpeedGain="1"/>\n')
            f.write(
                '    <vType id="standard_car" color="255,225,0" length="5.0" maxSpeed="22.2" carFollowModel="IDM" laneChangeModel="LC2013" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain="1"/>\n')
            f.write('    <route id="0" edges="R1 R2 R3 R4 R5"/>\n')
            f.write('    <route id="1" edges="R1 R2 R3 R4 RL3"/>\n')
            f.write('    <route id="2" edges="L1 RL1 RL2 R3 R4 R5"/>\n')
            f.write('    <!-- Vehicles, persons and containers (sorted by depart) -->\n')
            f.write(
                f'    <flow id="f_3" begin="1.00" departLane="free" departPos="free" departSpeed="speedLimit" route="0" end="7200.00" vehsPerHour="{lane1}.00"/>\n')
            f.write(
                f'    <flow id="f_1" begin="1.00" departLane="free" departPos="free" departSpeed="speedLimit" route="1" end="7200.00" vehsPerHour="{lane2}.00"/>\n')
            f.write(
                f'    <flow id="f_2" begin="1" departLane="free" departPos="free" departSpeed="speedLimit" route="2" end="7200.00" vehsPerHour="{lane3}.00"/>\n')
            f.write("</routes>\n")

        # 写入 .sumocfg 文件
        with open(sumocfg_filename, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration>\n')
            f.write('    <input>\n')
            f.write(f'        <net-file value="D:\项目文件\人工智能创意赛\webUI\wuhandata\\map2.net.xml"/>\n')
            f.write(f'        <route-files value="flow.rou.xml"/>\n')
            f.write('        <additional-files value="D:\项目文件\人工智能创意赛\webUI\wuhandata\\nsdet.add.xml"/>\n')
            f.write('    </input>\n')
            f.write('    <time>\n')
            f.write('        <begin value="0"/>\n')
            f.write('        <end value="10000"/>\n')
            f.write('        <step-length value="1"/>\n')
            f.write('    </time>\n')
            f.write('</configuration>\n')

        # 调用模型脚本并传递 .sumocfg 文件路径
        result = subprocess.run(['python', 'model.py', sumocfg_filename], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Model script failed with error: {result.stderr}")

        output_dir = os.path.join(directory, 'output')
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output folder {output_dir} was not created.")

        # 假设你的模型在输出目录中生成结果文件，你可以读取这些文件并返回内容
        # 下面是一个示例，假设有一个output.txt结果文件
        output_path = os.path.join(output_dir, 'output.txt')
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                model_output = f.read()
            return jsonify({"output": model_output})
        else:
            return jsonify({"output": "Model ran successfully but no output found."})

    except Exception as e:
        # 记录错误并返回
        app.logger.error(f"Error running model: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import subprocess
# import uuid
# import os
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/run-model', methods=['POST'])
# def run_model():
#     try:
#         data = request.json
#         lane1 = data['lane1']
#         lane2 = data['lane2']
#         lane3 = data['lane3']
#
#         # 生成唯一的目录名
#         unique_id = uuid.uuid4()
#         directory = f'run_{unique_id}'
#
#         # 创建目录
#         os.makedirs(directory, exist_ok=True)
#
#         # 定义输入和输出文件的路径
#         input_filename = os.path.join(directory, 'input.txt')
#         output_filename = os.path.join(directory, 'output.txt')
#
#         # 写入输入文件
#         with open(input_filename, 'w') as f:
#             f.write(f"{lane1}\n{lane2}\n{lane3}\n")
#
#         # 调用模型脚本并传递输入和输出文件的路径
#         result = subprocess.run(['python', 'model.py', input_filename, output_filename], capture_output=True, text=True)
#
#         if result.returncode != 0:
#             raise RuntimeError(f"Model script failed with error: {result.stderr}")
#
#         if not os.path.exists(output_filename):
#             raise FileNotFoundError(f"Output file {output_filename} was not created.")
#
#         # 读取并返回输出文件的内容
#         with open(output_filename, 'r') as f:
#             model_output = f.read()
#
#         return jsonify({"output": model_output})
#
#     except Exception as e:
#         # 记录错误并返回
#         app.logger.error(f"Error running model: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True)