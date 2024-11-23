import pathlib as pl
import pandas as pd
import numpy as np
import csv
import random


# ACCESS DIFFERENCE FOLDERS FOR DIFFERENT FILES
src = pl.Path.cwd().parent
data_repo = src/"data"
dump = src/"dump"
icer = dump/"icer"
matrices = dump/"matrices"
graphs = dump/"graphs"
owsa = dump/"owsa"
psa = dump/"psa"

# MODEL PARAMETERS
#specify model start age as 20,50,75
START_AGE = 50
RUN_TIME = 100-START_AGE
CYCLE_LENGTH = 1
iterations=1
#specify testing adherence level: 25, 75, 100%
adherence=100


# MODEL STATES
ALL_STATES = {
        0: "start", 
        1: "msi", 
        2: "ihc",
        3: "ts",
        4: "germline", 
        5: "cancer", 
        6: "cancer_death",
        7: "all_cause"
        }

WTP = 100000 # willingness to pay threshold
discount = .97

strats = list(range(1))

# Pre calibrated transition matrices
t_matrix_dict = {
        
# THIS IS FOR CASCADE PORTION OF MODEL     

        "gen20": np.load(matrices/"gen20.npy"),
        "gen20t": np.load(matrices/"gen20.npy"),
        "nh20": np.load(matrices/"nh20.npy"),
        "mlh20q1": np.load(matrices/"mlh20q1.npy"),
        "mlh20q2": np.load(matrices/"mlh20q2.npy"),
        "pms20q1": np.load(matrices/"pms20q1.npy"),
        "pms20q2": np.load(matrices/"pms20q2.npy"),
        "pms20q3": np.load(matrices/"pms20q3.npy"),
              
        
        "gen50": np.load(matrices/"gen50.npy"),
        "gen50t": np.load(matrices/"gen50.npy"),
        "nh50": np.load(matrices/"nh50.npy"),
        "mlh50q1": np.load(matrices/"mlh50q1.npy"),
        "mlh50q2": np.load(matrices/"mlh50q2.npy"),
        "pms50q1": np.load(matrices/"pms50q1.npy"),
        "pms50q2": np.load(matrices/"pms50q2.npy"),
        "pms50q3": np.load(matrices/"pms50q3.npy"),     

        
        "gen75": np.load(matrices/"gen75.npy"),
        "gen75t": np.load(matrices/"gen75.npy"),
        "nh75": np.load(matrices/"nh75.npy"),
        "mlh75q1": np.load(matrices/"mlh75q1.npy"),
        "mlh75q2": np.load(matrices/"mlh75q2.npy"),
        "pms75q1": np.load(matrices/"pms75q1.npy"),
        "pms75q2": np.load(matrices/"pms75q2.npy"),
        "pms75q3": np.load(matrices/"pms75q3.npy"),
        
   
        }

# DISEASE AND DEATH STATES IN THE MODEL
disease_states = [5, 6, 7]
death_states = [6, 7]
life_states = [0,1,2,3,4,5]


start_state = 0 
csy_cost = 1203  
cancer_death_cost = 105900 
init_cancer_cost = 63900  
cancer_cost = 6100
healthy_util = 1
cancer_util = 0.73
germline_test_cost = 328.25
csy_complication_cost = 3981.18
csy_complication = 0.007
csy_death_prob= 0.0061
couns_cost = 250
csy_disutility = 0.0384


strategies = {
    "Strategy1": {
        "nh": f"nh{START_AGE}",
        "gen": f"gen{START_AGE}",
        "gen_t": f"gen{START_AGE}t",
        "mlh": f"mlh{START_AGE}q1",
        "pms": f"pms{START_AGE}q1",
    },
    "Strategy2": {
        "nh": f"nh{START_AGE}",
        "gen": f"gen{START_AGE}",
        "gen_t": f"gen{START_AGE}t",
        "mlh": f"mlh{START_AGE}q1",
        "pms": f"pms{START_AGE}q2",
    },
    "Strategy3": {
        "nh": f"nh{START_AGE}",
        "gen": f"gen{START_AGE}",
        "gen_t": f"gen{START_AGE}t",
        "mlh": f"mlh{START_AGE}q1",
        "pms": f"pms{START_AGE}q3",
    },
    "Strategy4": {
        "nh": f"nh{START_AGE}",
        "gen": f"gen{START_AGE}",
        "gen_t": f"gen{START_AGE}t",
        "mlh": f"mlh{START_AGE}q2",
        "pms": f"pms{START_AGE}q2",
    },
    "Strategy5": {
        "nh": f"nh{START_AGE}",
        "gen": f"gen{START_AGE}",
        "gen_t": f"gen{START_AGE}t",
        "mlh": f"mlh{START_AGE}q2",
        "pms": f"pms{START_AGE}q3",
    },
}

df = pd.read_csv(f"{adherence}_adherence.csv")

class Person:
    def __init__(self, pid, df, strategy_name, strategies, t_matrix_dict):
        # Find the group of the patient from the dataset (df)
        patient_row = df[df['pid'] == pid]
        if patient_row.empty:
            raise ValueError(f"Patient ID {pid} not found in dataset.")
        
        group = patient_row.iloc[0]['group']  # Get the group for this patient

        # Check if the strategy exists
        if strategy_name not in strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found in strategies.")
        
        # Get the group-to-matrix mapping for the given strategy
        group_to_matrix_mapping = strategies[strategy_name]
        
        # Check if the group exists in the strategy's mapping
        if group not in group_to_matrix_mapping:
            raise ValueError(f"Group '{group}' not found in strategy '{strategy_name}' mapping.")
        
        # Get the corresponding matrix name for this group from the mapping
        matrix_name = group_to_matrix_mapping[group]
        
        # Check if the matrix name exists in the transition matrices dictionary (t_matrix_dict)
        if matrix_name not in t_matrix_dict:
            raise ValueError(f"Matrix '{matrix_name}' not found in transition matrices.")
        
        # Assign the transition matrix based on the group
        self.t_matrix = t_matrix_dict[matrix_name]
        
        # Store group, matrix name, strategy name, and patient ID
        self.group = group
        self.chosen_matrix_name = matrix_name
        self.strategy_name = strategy_name
        
        # Initialize the starting state (can be modified based on your logic)
        self.current_state = 0  # Starting state
        self.pid = pid
        # # Debugging: Log assignment
        # self.log_assignment(pid, group, matrix_name)


