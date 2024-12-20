###FOR MULTIPLE STRATGIES###
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm import tqdm
import presets_population_based_testing as ps
import os

def update_screening_costs(dis_cost, un_cost, dis_screening, un_screening, csy_counter, ps, discount_rate):
    dis_cost += ps.csy_cost * discount_rate
    un_cost += ps.csy_cost 
    
    dis_screening += ps.csy_cost * discount_rate
    un_screening += ps.csy_cost
    csy_counter += 1
    return dis_cost, un_cost, dis_screening, un_screening, csy_counter



def apply_csy_and_check_for_complications(t, ps, person, dis_cost, un_cost, dis_screening, un_screening, csy_counter, discount_rate):
    """
    Function to handle CSY logic and complications for each patient at each time step.
    """
    csy_complication_flag = 0  # Initialize flag with default value
    csy_death_flag = 0  # Initialize flag with default value
    # Define CSY intervals and age ranges for each chosen_matrix_name
    csy_settings = {
        "gen20": {"interval": 5, "start": 19, "end": 64},
        "gen20t": {"interval": 5, "start": 19, "end": 64},
        "nh20": {"interval": 5, "start": 19, "end": 64},
        "nh20t": {"interval": 5, "start": 19, "end": 64},
        "gen50": {"interval": 5, "start": 0, "end": 34},
        "gen50t": {"interval": 5, "start": 0, "end": 34},
        "nh50": {"interval": 5, "start": 0, "end": 34},
        "nh50t": {"interval": 5, "start": 0, "end": 34},
        "gen75": {"interval": 5, "start": 0, "end": 9},
        "gen75t": {"interval": 5, "start": 0, "end": 9},
        "nh75": {"interval": 5, "start": 0, "end": 9},
        "nh75t": {"interval": 5, "start": 0, "end": 9},
        "mlh20q1": {"interval": 1, "start": 0, "end": 64},
        "mlh20q2": {"interval": 2, "start": 0, "end": 64},
        "pms20q1": {"interval": 1, "start": 9, "end": 64},
        "pms20q2": {"interval": 2, "start": 9, "end": 64},
        "pms20q3": {"interval": 3, "start": 9, "end": 64},
        "mlh50q1": {"interval": 1, "start": 0, "end": 34},
        "mlh50q2": {"interval": 2, "start": 0, "end": 34},
        "pms50q1": {"interval": 1, "start": 0, "end": 34},
        "pms50q2": {"interval": 2, "start": 0, "end": 34},
        "pms50q3": {"interval": 3, "start": 0, "end": 34},
        "mlh75q1": {"interval": 1, "start": 0, "end": 9},
        "mlh75q2": {"interval": 2, "start": 0, "end": 9},
        "pms75q1": {"interval": 1, "start": 0, "end": 9},
        "pms75q2": {"interval": 2, "start": 0, "end": 9},
        "pms75q3": {"interval": 3, "start": 0, "end": 9},
    }

    # Retrieve CSY interval and range for the current matrix name
    settings = csy_settings.get(person.chosen_matrix_name)
    if settings and t % settings["interval"] == 0 and settings["start"] <= t <= settings["end"]:
        dis_cost, un_cost, dis_screening, un_screening, csy_counter = update_screening_costs(
            dis_cost, un_cost, dis_screening, un_screening, csy_counter, ps, discount_rate
        )
        if np.random.random() < ps.csy_complication:  # Complication probability
            dis_cost += ps.csy_complication_cost * discount_rate  # Discounted cost
            un_cost += ps.csy_complication_cost  # Undiscounted cost
            
            dis_screening += ps.csy_complication_cost * discount_rate
            un_screening += ps.csy_complication_cost
            csy_complication_flag = 1  # Complication occurred
            
            # Check if the complication results in death
            if np.random.random() < ps.csy_death_prob:
                csy_death_flag = 1  # Mark as death from CSY

    return dis_cost, un_cost, dis_screening, un_screening, csy_counter, csy_complication_flag, csy_death_flag

# Function to handle a subset of patients
def microsim_worker(args):
    strategy_name, group_to_matrix_mapping, t_matrix_dict, run_time, patient_ids = args

    records = []  # Local records for this subset of patients

    # Fetch the group-to-matrix mapping for the selected strategy
    group_to_matrix_mapping = ps.strategies[strategy_name]

    for pid in tqdm(patient_ids, desc=f"Simulating Patients {patient_ids[0]}-{patient_ids[-1]}"):
        # Use the patient's group and assign the correct matrix
        person = ps.Person(pid, ps.df, strategy_name, ps.strategies, ps.t_matrix_dict)
        chosen_matrix_name = person.chosen_matrix_name  # Matrix name selected for the patient

        # Initialize matrices and states
        p_matrix = np.zeros(run_time)
        p_matrix[0] = ps.start_state

        life_years = dis_cost = un_cost = utilities = ACM = 0
        cd_t = acm_t = cancer_t = cancer_count = cancer_death_count = 0
        un_cancer = dis_cancer = un_screening = dis_screening = csy_counter = 0

        csy_complication_flag = 0
        csy_death_flag = 0

        for t in range(run_time - 1):
            discount_rate = 1 / (1 + (1 - ps.discount)) ** (t * ps.CYCLE_LENGTH)
            csy_complication_flag = 0
            csy_death_flag = 0

            # Transition state
            p_matrix[t + 1] = np.random.choice(
                list(ps.ALL_STATES.keys()), 
                p=person.t_matrix[t][int(p_matrix[t])]
            )
            state_name = ps.ALL_STATES[int(p_matrix[t + 1])]

            # Life years count
            if p_matrix[t + 1] in ps.life_states:
                life_years += 1

            # Apply initial germline test costs
            if t == 0 and chosen_matrix_name not in ["nh20", "nh50", "nh75", "gen20", "gen50", "gen75"]:
                dis_cost += (ps.germline_test_cost + ps.couns_cost) * discount_rate
                un_cost += ps.germline_test_cost + ps.couns_cost
                dis_screening += (ps.germline_test_cost + ps.couns_cost) * discount_rate
                un_screening += ps.germline_test_cost + ps.couns_cost

            # Apply CSY and check for complications
            if state_name == "start":
                utilities += ps.healthy_util * discount_rate
                dis_cost, un_cost, dis_screening, un_screening, csy_counter, csy_complication_flag, csy_death_flag = apply_csy_and_check_for_complications(
                    t, ps, person, dis_cost, un_cost, dis_screening, un_screening, csy_counter, discount_rate
                )

            if csy_complication_flag:
                utilities -= 0.0384 * discount_rate

            if csy_death_flag:
                p_matrix[t + 1] = list(ps.ALL_STATES.keys())[list(ps.ALL_STATES.values()).index("all_cause")]
                state_name = "all_cause"
                ACM = 1
                acm_t += t

            # Cancer and cancer death states
            if state_name == "cancer" and ps.ALL_STATES[int(p_matrix[t])] != "cancer":
                dis_cost += ps.init_cancer_cost * discount_rate
                un_cost += ps.init_cancer_cost
                dis_cancer += ps.init_cancer_cost * discount_rate
                un_cancer += ps.init_cancer_cost
                cancer_count = 1
                cancer_t += t
                cancer_first_year = t
            #  For subsequent years in the cancer state, apply subsequent_cancer_cost
            elif state_name == "cancer" and t > cancer_first_year:
                utilities += ps.cancer_util * discount_rate
                dis_cost += ps.cancer_cost * discount_rate
                un_cost += ps.cancer_cost
                
                dis_cancer += ps.cancer_cost * discount_rate
                un_cancer += ps.cancer_cost

            if state_name == "cancer_death" and ps.ALL_STATES[int(p_matrix[t])] != "cancer_death":
                dis_cost += ps.cancer_death_cost * discount_rate
                un_cost += ps.cancer_death_cost
                dis_cancer += ps.cancer_death_cost * discount_rate
                un_cancer += ps.cancer_death_cost
                cd_t += t
                cancer_death_count = 1

            if state_name == "all_cause" and ps.ALL_STATES[int(p_matrix[t])] != "all_cause":
                ACM = 1
                acm_t += t

        # Append patient record
        records.append({
                "Patient ID": pid,
                "Time": t,
                "State": state_name,
                "Discounted Cost": dis_cost,
                "Undiscounted Cost": un_cost,
                "Undiscounted Screening Cost": un_screening,
                "Discounted Screening Cost": dis_screening,
                "Undiscounted Cancer Cost": un_cancer,
                "Discounted Cancer Cost": dis_cancer,
                "QALY": utilities,
                "Life Years": life_years,
                "Cancer_Death": cancer_death_count,
                "Cancer": cancer_count,
                "csy_counter": csy_counter,
                "ACM_AGE": acm_t,
                "CANCER_INCIDENCE": cancer_t,
                "ACM": ACM,
                "CD_AGE": cd_t,
                "Chosen Matrix Name": chosen_matrix_name, 
                "Strategy Name": strategy_name 
                
        })

    return records

def microsim_parallel(strategies, t_matrix_dict, run_time, n, n_cores=None):
    if n_cores is None:
        n_cores = cpu_count()  # Default to all available cores

    # Split the patient population into chunks for each process
    patient_ids = np.array_split(range(n), n_cores)

    # Prepare the worker_args, ensuring the order matches what's expected by microsim_worker
    worker_args = [
        (strategy_name,  # Strategy name
         strategies[strategy_name],  # group_to_matrix_mapping (mapping for the strategy)
         t_matrix_dict,  # t_matrix_dict (the dictionary of matrices)
         run_time,  # run_time
         chunk)  # patient_ids chunk
        for strategy_name in strategies.keys()
        for chunk in patient_ids
    ]
    
    # Create a pool of workers and execute the simulation
    with Pool(n_cores) as pool:
        results = pool.map(microsim_worker, worker_args)

    # Combine results from all processes
    all_records = [record for result in results for record in result]

    # Convert the combined results to a DataFrame
    state_df = pd.DataFrame(all_records)

    # Separate results by strategy
    strategy_dfs = {strategy_name: state_df[state_df['Strategy Name'] == strategy_name] 
                    for strategy_name in strategies.keys()}

    return strategy_dfs

# # Main microsimulation function with parallel processing
# def microsim_parallel(strategy, run_time, n, n_cores=None):
#     if n_cores is None:
#         n_cores = cpu_count()  # Default to all available cores

#     # Split the patient population into chunks for each process
#     patient_ids = np.array_split(range(n), n_cores)

#     with Pool(n_cores) as pool:
#         results = pool.map(microsim_worker, [(strategy, run_time, chunk) for chunk in patient_ids])

#     # Combine results from all processes
#     all_records = [record for result in results for record in result]
#     state_df = pd.DataFrame(all_records)
#     return state_df

def analyze_results(strategy_dfs):
    """
    Analyzes the microsimulation results DataFrame for each strategy.
    Args:
        strategy_dfs (dict): A dictionary where keys are strategy names, and values are DataFrames with simulation results.
    Returns:
        dict: A dictionary containing summary metrics for each strategy.
    """
    summaries = []
    
    for strategy_name, df in strategy_dfs.items():
        dis_count_per_pt = df['Discounted Cost'].mean()
        undis_count_per_pt = df['Undiscounted Cost'].mean()
        
        dis_screening_per_pt = df['Discounted Screening Cost'].mean()
        undis_screening_per_pt = df['Undiscounted Screening Cost'].mean()
        
        dis_cancer_per_pt = df['Discounted Cancer Cost'].mean()
        undis_cancer_per_pt = df['Undiscounted Cancer Cost'].mean()
        
        qaly_per_pt = df['QALY'].mean()
        ly_per_pt = df['Life Years'].mean()
        csy_avg = df['csy_counter'].mean()
        
       # csy_flag_sum = df['CSY_Flag'].sum()
       # csy_death_sum = df['CSY_Death'].sum()
        
        # ACM-related calculations
        acm_df = df[df['ACM'] == 1]
        sum_acm_age = acm_df['ACM_AGE'].sum()
        count_acm = acm_df['ACM'].sum()
        average_acm_age = (sum_acm_age / count_acm)+ps.START_AGE if count_acm != 0 else None
        
        # Cancer Death-related calculations
        cancer_death_df = df[df['Cancer_Death'] == 1]
        sum_cd_age = cancer_death_df['CD_AGE'].sum() 
        count_cd = cancer_death_df['Cancer_Death'].sum()
        average_cd_age = (sum_cd_age / count_cd)+ps.START_AGE if count_cd != 0 else None
        
        # Cancer Incidence-related calculations
        cancer_incidence_df = df[df['Cancer'] == 1]
        sum_cancer_incidence_age = cancer_incidence_df['CANCER_INCIDENCE'].sum()
        count_cancer = cancer_incidence_df['Cancer'].sum()
        average_cancer_incidence_age = (sum_cancer_incidence_age / count_cancer)+ps.START_AGE if count_cancer != 0 else None
        
        summaries.append({
            'Strategy Name': strategy_name,
            'Life Years': ly_per_pt,
            'QALY': qaly_per_pt,
            'Number Cancer Diagnoses': count_cancer,
            'Number Cancer Deaths': count_cd,
            'Average_Cancer_Diagnosis_AGE': average_cancer_incidence_age,
            'Average_Cancer_Death_AGE': average_cd_age,
            'CSY': csy_avg,
            'Discounted Cost': dis_count_per_pt,
            'Undiscounted Cost': undis_count_per_pt,
            "Discounted Screening Cost": dis_screening_per_pt,
            "Undiscounted Screening Cost": undis_screening_per_pt,
            "Discounted Cancer Cost": dis_cancer_per_pt,
            "Undiscounted Cancer Cost": undis_cancer_per_pt,
            'Number ACM': count_acm,
            'Average_ACM_AGE': average_acm_age,
       #     "CSY Flags": csy_flag_sum,
        #    "CSY Deaths": csy_death_sum
        })

    return summaries

def save_results_to_excel(strategy_dfs, summaries, filename="microsim_results.xlsx"):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write individual strategy results to separate sheets
        for strategy_name, df in strategy_dfs.items():
            # Save each strategy's results in a separate sheet
            df.to_excel(writer, sheet_name=f"Results_{strategy_name}", index=False)

        # Combine summaries and write to a summary sheet
        summary_df = pd.DataFrame(summaries)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Results saved to {filename}.")

def main():
    strategies = ps.strategies
    t_matrix_dict = ps.t_matrix_dict
    run_time = ps.RUN_TIME
    n = len(ps.df)
    iterations = ps.iterations  # Number of iterations for sensitivity analysis or validation
    n_cores = 8  # Number of cores to use for parallel processing

    # List to store all iteration summaries
    all_iteration_summaries = []

    for i in range(iterations):
        print(f"Running microsimulation iteration {i+1}...")
        df = microsim_parallel(strategies, t_matrix_dict, run_time, n, n_cores)

        print(f"Analyzing results for iteration {i+1}...")
        summaries = analyze_results(df)

        # Append the current iteration's summary
        all_iteration_summaries.append(summaries)

    # Convert all summaries into a DataFrame
    # Flatten the list of lists of summaries into a single list of dicts
    flattened_summaries = [summary for iteration in all_iteration_summaries for summary in iteration]

    # Convert the flattened list of summaries into a DataFrame
    summaries_df = pd.DataFrame(flattened_summaries)

    # Calculate the average for each strategy across all iterations
    avg_summaries_df = summaries_df.groupby('Strategy Name').mean()

    # Also, save the average results
    avg_summaries_df.to_excel(f"microsim_avg_results_age{ps.START_AGE}.xlsx", engine='openpyxl')

    print("Averages per strategy saved to microsim_avg_results.xlsx.")
# Run the main function
if __name__ == "__main__":
    main()
