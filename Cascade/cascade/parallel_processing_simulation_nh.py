#For one iteration of microsim only with parallel processing###
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm import tqdm
import presets_population_based_nh as ps
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
    """
    Runs microsimulation for a subset of patients.
    Args:
        strategy: Strategy being simulated.
        run_time: Number of cycles to simulate.
        patient_ids: List of patient IDs for this worker.
    Returns:
        A DataFrame containing simulation results for the assigned patients.
    """
    strategy, run_time, patient_ids = args
    records = []  # Local records for this subset
   # np.random.seed(0)
    for pid in tqdm(patient_ids, desc=f"Simulating Patients {patient_ids[0]}-{patient_ids[-1]}"):
        person = ps.Person(pid, ps.df, ps.group_to_matrix_mapping, ps.t_matrix_dict)
        chosen_matrix_name = person.chosen_matrix_name
        
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

            # Cost and utility calculations
            if state_name in ps.cu_dict.keys():
                utilities += ps.cu_dict[state_name][1] * discount_rate
                dis_cost += ps.cu_dict[state_name][0] * discount_rate
                un_cost += ps.cu_dict[state_name][0]
                dis_cancer += ps.cu_dict[state_name][0] * discount_rate
                un_cancer += ps.cu_dict[state_name][0]

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

            elif state_name == "cancer_death" and ps.ALL_STATES[int(p_matrix[t])] != "cancer_death":
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
                "CSY_Flag": csy_complication_flag, 
                "CSY_Death": csy_death_flag
        })

    return records

# Main microsimulation function with parallel processing
def microsim_parallel(strategy, run_time, n, n_cores=None):
    if n_cores is None:
        n_cores = cpu_count()  # Default to all available cores

    # Split the patient population into chunks for each process
    patient_ids = np.array_split(range(n), n_cores)

    with Pool(n_cores) as pool:
        results = pool.map(microsim_worker, [(strategy, run_time, chunk) for chunk in patient_ids])

    # Combine results from all processes
    all_records = [record for result in results for record in result]
    state_df = pd.DataFrame(all_records)
    return state_df

def analyze_results(results_df):
    """
    Analyzes the microsimulation results DataFrame and calculates summary metrics.
    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
    Returns:
        pd.DataFrame: Summary metrics and chosen matrix-specific results.
    """
    # Ensure results are in a DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("Input results must be a pandas DataFrame.")

    # Calculate summary metrics
    dis_count_per_pt = results_df['Discounted Cost'].mean()
    undis_count_per_pt = results_df['Undiscounted Cost'].mean()
    
    dis_screening_per_pt = results_df['Discounted Screening Cost'].mean()
    undis_screening_per_pt = results_df['Undiscounted Screening Cost'].mean()
    
    dis_cancer_per_pt = results_df['Discounted Cancer Cost'].mean()
    undis_cancer_per_pt = results_df['Undiscounted Cancer Cost'].mean()
    
    qaly_per_pt = results_df['QALY'].mean()
    ly_per_pt = results_df['Life Years'].mean()
    csy_avg = results_df['csy_counter'].mean()
    
    csy_flag_sum = results_df['CSY_Flag'].sum()
    csy_death_sum = results_df['CSY_Death'].sum()
    
    # ACM-related calculations
    acm_df = results_df[results_df['ACM'] == 1]
    sum_acm_age = acm_df['ACM_AGE'].sum()
    count_acm = acm_df['ACM'].sum()
    average_acm_age = (sum_acm_age / count_acm)+ps.START_AGE if count_acm != 0 else None
    
    # Cancer Death-related calculations
    cancer_death_df = results_df[results_df['Cancer_Death'] == 1]
    sum_cd_age = cancer_death_df['CD_AGE'].sum() 
    count_cd = cancer_death_df['Cancer_Death'].sum()
    average_cd_age = (sum_cd_age / count_cd)+ps.START_AGE if count_cd != 0 else None
    
    # Cancer Incidence-related calculations
    cancer_incidence_df = results_df[results_df['Cancer'] == 1]
    sum_cancer_incidence_age = cancer_incidence_df['CANCER_INCIDENCE'].sum()
    count_cancer = cancer_incidence_df['Cancer'].sum()
    average_cancer_incidence_age = (sum_cancer_incidence_age / count_cancer)+ps.START_AGE if count_cancer != 0 else None
    
    # Create a DataFrame with the summary results
    summary = {
        "Total Patients": len(results_df),
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
        "CSY Flags": csy_flag_sum,
        "CSY Deaths": csy_death_sum
    }
    
    # Convert summary to DataFrame for better formatting
    summary_df = pd.DataFrame([summary])

    # Add specific summary for chosen matrices if needed
    chosen_matrix_summary = results_df.groupby("Chosen Matrix Name").agg({
        "Discounted Cost": "mean",
        "QALY": "mean",
        "Life Years": "mean",
        "Cancer": "sum",
        "Cancer_Death": "sum",
        "ACM": "sum"
    }).rename(columns={
        "Discounted Cost": "Mean Cost",
        "QALY": "Mean QALY",
        "Life Years": "Mean Life Years",
        "Cancer": "Total Cancer Incidence",
        "Cancer_Death": "Total Cancer Deaths",
        "ACM": "Total All-Cause Mortality"
    })

    # Return the summary DataFrame and chosen matrix summary
    return summary_df, chosen_matrix_summary

def main():
    # Set parameters
    strategy = ps.strats
    run_time = ps.RUN_TIME
    n = len(ps.df)  # Use predefined cohort size from `ps.df`
    iterations = 1  # Number of iterations for sensitivity analysis or validation
    n_cores = 8  # Number of cores to use for parallel processing

    # Create a list to store summary results for all iterations
    all_iteration_summaries = []

    for i in range(iterations):
        print(f"Running microsimulation iteration {i+1}...")
        results_df = microsim_parallel(strategy=strategy, run_time=run_time, n=n, n_cores=n_cores)
        
        # Optionally, analyze results for each iteration (if you want to track it)
        print(f"Analyzing results for iteration {i+1}...")
        summary, matrix_summary = analyze_results(results_df)

        # Append the summary for the current iteration to the all_iteration_summaries list
        all_iteration_summaries.append(summary)
        
        # Optionally, save the summary for the current iteration to a separate file
       # summary.to_csv(f"summary_results_iteration_{i+1}.csv", index=False)

    # Concatenate all summary results from all iterations into one DataFrame
    final_summary_df = pd.concat(all_iteration_summaries, ignore_index=True)

    # Calculate the average row (mean of each column across all iterations)
    average_row = final_summary_df.mean()

    # Add the average row as the last row in the final summary DataFrame
    final_summary_df.loc['Average'] = average_row

    # Save the combined summary results with the average row to a single CSV file
    final_summary_df.to_excel(f"microsim_avg_results_age_nh{ps.START_AGE}.xlsx", engine='openpyxl')


    print("All iterations complete. Summary results with average row saved to 'microsim_summary_all_iterations.csv'.")

# Run the main function
if __name__ == "__main__":
    main()
