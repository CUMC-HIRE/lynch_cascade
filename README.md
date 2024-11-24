# Lynch Syndrome Cascade Model
This repository contains a microsimulation model designed to simulate the health outcomes, costs, and transitions of patients in the context of Lynch syndrome cascade screening strategies. The model is built to explore different screening approaches and their long-term impacts on patient health and healthcare costs.

## Overview
The Lynch Syndrome Cascade Model uses a microsimulation approach to track a cohort of patients with predefined characteristics through their health state transitions. Please run model for least a few hundred iterations to ensure robust and reproducible results.

## Key Features:
* Microsimulation of 15,000 Patients: Simulates individual patient trajectories through various health states.
* Surveillance Intervals: Patients are assigned transition matrices reflecting the following surveillance strategies:
    * MLH1 and MSH2: Every year or every 2 years.
    * MSH6 and PMS2: Every year, every 2 years, or every 3 years.
* Model Start Age: We begin our simulation model at 3 different ages: 20, 50, and 75 years old, and patients are tracked until age 100 or death. 
* Predefined Cohort: Patients are initialized using an imported dataset containing their baseline characteristics and are assigned specific transition matrices.
* Transition Matrices: Define the probabilities of moving between health states based on patient characteristics and screening strategies.

## Function
1. Predefined Cohort: The cohort indicates how patients are assigned transition matrices. 
2. Microsimulation:
    * The simulation tracks 15,000 patients over time, recording life years, quality adjusted life years (QALY), cancer cases, cancer deaths, average age of cancer diagnosis, average age of cancer death, number of colonoscopies a patient received, total costs, screening costs, and cancer-related costs. 
3. Outputs: The model outputs detailed patient-level data and aggregate results. 



