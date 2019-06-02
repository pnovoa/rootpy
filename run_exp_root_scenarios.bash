#!/bin/bash

# declare an array variable
#declare -a arr_rbf=("rbf_gaussian" "rbf_thinplate" "shepard" "svr_gaussian_kernel" "svr_laplacian_kernel")
#declare -a arr_rbf=("svr_gaussian_kernel" "svr_thinplatespline_kernel")
#declare -a arr_rbf=("svr_laplacian_kernel")

declare -a samplesizes=(60)
declare -a a_num_iter=(30)
declare -a change_types=(1 2)
declare -a time_windows=(2 6 10)

framework_list=("JinFramework")
learning_period=20
num_runs=15
num_changes=50
num_processess=3
pop_size=1

foldername="output/$(date +%Y_%m_%d_%H_%M_%S)/"
mkdir -p ${foldername}

jobb=""

let frm_id=-1

for framework in "${framework_list[@]}";
do
    ((frm_id+=1))
    for num_iter in "${a_num_iter[@]}";
    do
	for ss in "${samplesizes[@]}";
	do
		for j in "${change_types[@]}";
		do
			for k in "${time_windows[@]}";
			do
				jobb="python3 run_experiment_scenarios.py ${num_processess} ${ss} ${framework} ${frm_id} ${j} ${k} ${learning_period} ${num_runs} ${num_changes} ${num_iter} ${pop_size} ${foldername}";
				echo Processing $jobb;
				$jobb & pid=$!;
				PID_LIST+=" $pid";
			done;
			trap "kill $PID_LIST" SIGINT
			echo "Parallel processes have started";
			wait $PID_LIST;
			echo;
			echo "All processes have been completed for change type ${j}";
		done;
	done;
    done;
done
