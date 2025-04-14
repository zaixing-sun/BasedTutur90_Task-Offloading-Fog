import os
import json
import numpy as np
import pandas as pd


def generate_tasks(num_arrivals_per_ingress, lambda_, ingress_line, param_TaskSize, param_CyclesPerBit, param_TransBitRate, param_DDL):
    """
    Generate synthetic tasks based on the given parameters.
    
    :param num_arrivals_per_ingress: Number of arrivals per ingress node
    :param lambda_: Average rate of arrivals per time unit for each ingress node
    :param ingress_line: List of ingress node IDs
    :param param_TaskSize: Tuple of min and max task size
    :param param_CyclesPerBit: Tuple of min and max cycles per bit
    :param param_TransBitRate: Tuple of min and max transmission bit rate           
    :param param_DDL: Tuple of min and max deadline
    :return: List of generated tasks
    """
    tasks = []
    lambdas = [lambda_] * len(ingress_line)  # Average rate of arrivals per time unit for each ingress node
    for node_id, lambda_ in zip(ingress_line, lambdas):
        # # Generate inter-arrival times
        # inter_arrival_times = np.random.exponential(1 / lambda_, num_arrivals_per_ingress)
        # # Compute arrival times
        # arrival_times = np.cumsum(inter_arrival_times)
        # arrival_times = np.round(arrival_times).tolist()

        num_arrivals_0 = list(np.random.poisson(lambda_, num_arrivals_per_ingress))
        num_arrivals = []
        k = 0
        while k<num_arrivals_per_ingress:
            k1 = num_arrivals_0.pop(0)+1
            num_arrivals.append(k1)
            k += k1
        sum1 = sum(num_arrivals)
        if sum1>num_arrivals_per_ingress:
            num_arrivals[-1] -= (sum1 - num_arrivals_per_ingress)
        interarrival_times = np.round( np.random.exponential(scale=1/lambda_, size=len(num_arrivals)), 2)
        arrival_times = list(np.round(np.cumsum(interarrival_times), 2))

        for numWF in num_arrivals:
            arrival_time = arrival_times.pop(0)
            for _ in range(numWF):
                tasks.append(
                    [
                        f't{0}', # TaskName (invalid)
                        np.round(arrival_time+1,2),  # arrival_time, #    # GenerationTime 
                        0,  # TaskID (invalid)
                        np.random.randint(*param_TaskSize),  # TaskSize
                        np.random.randint(*param_CyclesPerBit),  # CyclesPerBit
                        10 * np.random.randint(*param_TransBitRate),  # TransBitRate
                        np.random.randint(*param_DDL),  # DDL
                        f'n{node_id}',  # SrcName
                    ]
                )
    ordered_tasks = sorted(tasks, key=lambda x: x[1])
    for i, item in enumerate(ordered_tasks):
        item[0] = f't{i}'  # TaskName (valid)
        item[2] = i  # TaskID (valid)

    return ordered_tasks

def save_tasks_to_csv(tasks, save_as, header):
    """
    Save the generated tasks to a CSV file.
    
    :param tasks: List of generated tasks
    :param save_as: Path to save the CSV file
    :param header: Header for the CSV file
    """
    data = pd.DataFrame(tasks, columns=header)
    data.to_csv(save_as, index=False)
    # if not os.path.exists(save_as):
    #     data = pd.DataFrame(tasks, columns=header)
    #     data.to_csv(save_as, index=False)
    # else:
    #     print(f"File {save_as} already exists!")

def save_params_to_json(params, params_save_as):
    """
    Save the parameters to a JSON file.
    
    :param params: Dictionary of parameters
    :param params_save_as: Path to save the JSON file
    """
    with open(params_save_as, 'w+') as fw:
        json.dump(params, fw, indent=4)
    # if not os.path.exists(params_save_as):
    #     with open(params_save_as, 'w+') as fw:
    #         json.dump(params, fw, indent=4)
    # else:
    #     print(f"File {params_save_as} already exists!")




def main():
    """
    Main function to generate and save synthetic tasks.
    """

    # 1. parameters
    header = ['TaskName', 'GenerationTime', 'TaskID', 'TaskSize', 'CyclesPerBit', 
              'TransBitRate', 'DDL', 'SrcName']  # field names
    
    '''
    [1] Fan W. Blockchain-Secured Task Offloading and Resource Allocation for Cloud-Edge-End Cooperative Networks[J]. IEEE Transactions on Mobile Computing, 2024, 23(8): 8092–8110.
    [1] Fan W, Zhao L, Liu X, Su Y, Li S, Wu F, Liu Y. Collaborative Service Placement, Task Scheduling, and Resource Allocation for Task Offloading with Edge-Cloud Cooperation[J]. IEEE Transactions on Mobile Computing, 2024, 23(1): 238–256.
    '''
    param_TaskSize = (10, 100 + 1)  # Mb
    param_CyclesPerBit = (100, 1000 + 1)  # per-MBit
    param_TransBitRate = (1, 5)  # Mbps
    param_DDL = (20, 100 + 1)  # s
    # param_Latency = (0, 0.005)  # s

    # 2. scenarios
    scenarios = ['25N50E', '50N50E', '100N150E', 'MilanCityCenter']
    scenarios_dict_train = {
        '25N50E': {'nodes': 25, 
                   'edges': 50, 
                   'lambda_': np.round(0.04*6.25,  2),
                   'num_arrivals_per_ingress': 100, 
                   'ingress_line': list(range(25))},
        '50N50E': {'nodes': 50,
                     'edges': 50,
                     'lambda_': np.round(0.02*6.25, 2),
                     'num_arrivals_per_ingress': 200,
                     'ingress_line': list(range(50))},
        '100N150E': {'nodes': 100,
                    'edges': 150,
                    'lambda_': np.round(0.01*6.25, 2),
                    'num_arrivals_per_ingress': 400,
                    'ingress_line': list(range(100))},
        'MilanCityCenter': {'nodes': 30,
                            'edges': 35,
                            'lambda_': np.round(0.03333*6.25, 2),
                            'num_arrivals_per_ingress': 120,
                            'ingress_line': list(range(30))}
    }

    scenarios_dict_test = { 
        '25N50E': {'nodes': 25,
                   'edges': 50,
                   'lambda_': np.round(0.04*6.25,2),
                   'num_arrivals_per_ingress': int(100/2),
                   'ingress_line': list(range(25))},
        '50N50E': {'nodes': 50,
                   'edges': 50,
                   'lambda_': 0.02*6.25,
                   'num_arrivals_per_ingress': int(200/2),
                   'ingress_line': list(range(50))},
        '100N150E': {'nodes': 100,
                     'edges': 150,
                     'lambda_': np.round(0.01*6.25,2),
                     'num_arrivals_per_ingress': int(400/2),
                     'ingress_line': list(range(100))},
        'MilanCityCenter': {'nodes': 30,
                           'edges': 35,
                           'lambda_': np.round(0.03333*6.25,2),
                           'num_arrivals_per_ingress': int(120/2),
                           'ingress_line': list(range(30))}
    }


    # 3. generate train data
    for flag, params in scenarios_dict_train.items():
        num_arrivals_per_ingress = params['num_arrivals_per_ingress']
        lambda_ = params['lambda_']
        ingress_line = params['ingress_line']     

        save_as = f"eval/benchmarks/Topo4MEC/data/{flag}/trainset.csv"      
        params_save_as = f"eval/benchmarks/Topo4MEC/data/{flag}/trainset_configs.json"

        # 3.1. synthetic tasks
        tasks = generate_tasks(num_arrivals_per_ingress, lambda_, ingress_line, param_TaskSize, param_CyclesPerBit, param_TransBitRate, param_DDL)

        # 3.2. saving
        save_tasks_to_csv(tasks, save_as, header)
        params = {
            'num_arrivals_per_ingress': num_arrivals_per_ingress,
            'lambdas': [lambda_] * len(ingress_line),
            'ingress': ingress_line,
            'param_TaskSize': param_TaskSize,
            'param_CyclesPerBit': param_CyclesPerBit,
            'param_TransBitRate': param_TransBitRate,
            'param_DDL': param_DDL,
        }
        save_params_to_json(params, params_save_as)

    # 4. generate test date
    for flag, params in scenarios_dict_test.items():
        num_arrivals_per_ingress = params['num_arrivals_per_ingress']
        lambda_ = params['lambda_']

        with open(f"eval/benchmarks/Topo4MEC/source/{flag}/ingress.txt", 'r') as f:
            ingress_line = f.readlines()[1].split()
            ingress_line = [int(item) - 1 for item in ingress_line]  # RayCloudSim is 0-index

        save_as = f"eval/benchmarks/Topo4MEC/data/{flag}/testset.csv"
        params_save_as = f"eval/benchmarks/Topo4MEC/data/{flag}/testset_configs.json"
        
        # 4.1. synthetic tasks
        tasks = generate_tasks(num_arrivals_per_ingress, lambda_, ingress_line, param_TaskSize, param_CyclesPerBit, param_TransBitRate, param_DDL)
        
        # 4.2. saving
        save_tasks_to_csv(tasks, save_as, header)
        params = {
            'num_arrivals_per_ingress': num_arrivals_per_ingress,
            'lambdas': [lambda_] * len(ingress_line),
            'ingress': ingress_line,
            'param_TaskSize': param_TaskSize,
            'param_CyclesPerBit': param_CyclesPerBit,
            'param_TransBitRate': param_TransBitRate,
            'param_DDL': param_DDL,
        }
        save_params_to_json(params, params_save_as)
    # 5. print
    print("===============================================")
    print("All tasks generated and saved successfully.")
    # # 6. loading
    # data = pd.read_csv(save_as)
    # ordered_tasks = list(data.iloc[:].values)





if __name__ == '__main__':
    random_seed = 985689
    np.random.seed(random_seed)
    main()
