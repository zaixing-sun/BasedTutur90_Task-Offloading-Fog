import os
import json
import random

def generate_configs(scenario, nodes_num):
    """
    Generate configuration files for the given scenario and number of nodes.
    :param scenario: Scenario name (e.g., '25N50E', '50N50E', '100N150E', 'MilanCityCenter').
    :param nodes_num: Number of nodes in the scenario.
    :return: None
    """
   
    # Load the graph data from the specified scenario file.
    with open(f"eval/benchmarks/Topo4MEC/source/{scenario}/graph.txt", 'r') as f:
        graph_lines = f.readlines()
        edges_lines = [line.split() for line in graph_lines]
        edges_lines = [[int(line[0]) - 1, int(line[1]) - 1, float(line[2])]
                       for line in edges_lines]  # RayCloudSim is 0-index

    # Create a directory to save the configuration files.
    save_as = f"eval/benchmarks/Topo4MEC/data/{scenario}/config.json"


    # Generate nodes and edges configurations.
    nodes = []
    for node_id in range(nodes_num):
        idle_energy_coef = max(0.01 * random.random(), 0.001)
        exe_energy_coef = 10 * idle_energy_coef
        param_Latency = (0, 0.005)  # s

        nodes.append(
            {
                'NodeType': 'Node',
                'NodeName': f'n{node_id}',
                'NodeId': node_id,
                'MaxCpuFreq': random.randint(10, 50) * 100,  # MHz, 1 GHz = 1000 MHz
                'MaxBufferSize': random.randint(5, 40) * 10,  # Mb
                'IdleEnergyCoef': round(idle_energy_coef, 4),
                'ExeEnergyCoef': round(exe_energy_coef, 4),
                'Latency': round(random.uniform(*param_Latency), 4),  # s
            }
        )

    edges = []
    for src, dst, bw in edges_lines:
        edges.append(
            {
                'EdgeType': 'SingleLink', 
                'SrcNodeID': src,
                'DstNodeID': dst, 
                'Bandwidth': 10 * bw,  # Mbps
            }
        )

    # Save the generated configurations to a JSON file.
    data = {
        'Nodes': nodes,
        'Edges': edges,
    }
    json_object = json.dumps(data, indent=4)
    
    with open(save_as, 'w+') as fw:
            fw.write(json_object)
    # if not os.path.exists(save_as):
    #     with open(save_as, 'w+') as fw:
    #         fw.write(json_object)
    # else:
    #     print(f"File {save_as} already exists!")

    # Load the configurations from the JSON file.
    with open(save_as, 'r') as fr:
        json_object = json.load(fr)
        nodes, edges = json_object['Nodes'], json_object['Edges']
    print(f"{len(nodes)} nodes, {len(edges)} edges")
    




def main():
    # 1. loading source files
    scenarios = ['25N50E', '50N50E', '100N150E', 'MilanCityCenter']
    nodes_nums = [25, 50, 100, 30]

    # 2. generate configs
    for scenario, nodes_num in zip(scenarios, nodes_nums):
        generate_configs(scenario, nodes_num)
    print("Finished generating configuration files.")



if __name__ == '__main__':
    random.seed(878289)
    main()
