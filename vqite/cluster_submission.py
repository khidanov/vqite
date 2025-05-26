"""
This script generates job scripts to run "qiskit_varqite_run.py" on
multiple computing nodes on a cluster.
The parameters for "qiskit_varqite_run.py" are specified in the
"set_of_params" list below.
Separate job is created and executed for each parameter set.
"""

import numpy as np
import os

"""
Specifying parameter values for "qiskit_varqite_run.py"
"""

set_of_params=[(filename,
                init_params,
                optimize_m,
                optimize_v,
                simplify_sequence) 
               for filename in ["N12g0.8"]
               for init_params in ["zeros"]
               for optimize_m in ["greedy"]
               for optimize_v in ["greedy"]
               for simplify_sequence in ["ADCRS"]
              ]

# set_of_params=[(filename,
#                 init_params,
#                 optimize_m,
#                 optimize_v,
#                 simplify_sequence) 
#                for filename in ["N12g1.0"]
#                for init_params in ["random"]
#                for optimize_m in ["greedy"]
#                for optimize_v in ["greedy"]
#                for simplify_sequence in ['""']
#               ]

# set_of_params=[(filename,
#                 init_params,
#                 optimize_m,
#                 optimize_v,
#                 simplify_sequence) 
#                for filename in ["N12g0.8"]
#                for init_params in ["zeros"]
#                for optimize_m in ["hyper"]
#                for optimize_v in ["greedy","hyper"]
#                for simplify_sequence in ['""',"CRS"]
#               ]

# set_of_params=[(filename,
#                 init_params,
#                 optimize_m,
#                 optimize_v,
#                 simplify_sequence) 
#                for filename in ["N12g0.1","N12g0.5","N12g0.8","N12g1.0","N12g1.2","N12g1.5"]
#                for init_params in ["zeros","random"]
#                for optimize_m in ["greedy","hyper"]
#                for optimize_v in ["greedy","hyper"]
#                for simplify_sequence in ['""',"CRS","ADCRS"]
#               ]


#n_tasks_per_node = 64 (i.e. 2 CPUs per task) appears to be the best option.
#Calculating M is significantly faster in this case than for n_tasks_per_node = 128
#Further increasing cpus_per_task does not seem to improve the speed.
n_tasks_per_node = 64
n_nodes = 20
n_tasks = n_tasks_per_node*n_nodes
#Despite the fact that in NERSC manual cpus_per_task is indicated to be 
#equal to 2*int(128/n_tasks_per_node), SLURM gives an error when submitting
#a job with that value. Instead, the below value is working:
cpus_per_task = int(128/n_tasks_per_node)

file_dir = os.path.dirname(os.path.abspath(__file__))

"""
For each parameter value, creating an sbatch job to execute.
Execution logs are saved into a directory logs/.
"""
for params in set_of_params:
    (filename, init_params, optimize_m, optimize_v, simplify_sequence) = params
    job_file = "cluster_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -A m529\n")
        fh.writelines("#SBATCH -C cpu\n")

        fh.writelines("#SBATCH --qos=premium\n")
        fh.writelines("#SBATCH -t 24:00:00\n")
        fh.writelines("#SBATCH --nodes=%s\n" % n_nodes)
        fh.writelines("#SBATCH --ntasks-per-node=%s\n" % n_tasks_per_node)
        fh.writelines("#SBATCH --cpus-per-task=%s\n" % cpus_per_task)
        # fh.writelines("#SBATCH --cpu-bind=cores\n")

        # fh.writelines("#SBATCH --qos=regular\n")
        # fh.writelines("#SBATCH -t 24:00:00\n")
        # fh.writelines("#SBATCH --nodes=1\n")
        # fh.writelines("#SBATCH --ntasks-per-node=64\n")
        # fh.writelines("#SBATCH --cpus-per-task=2\n")
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o "+file_dir+"/out_err/"+filename+init_params+
                      "_om"+optimize_m+"_ov"+optimize_v+"_s"+simplify_sequence+"_out\n")
        fh.writelines("#SBATCH -e "+file_dir+"/out_err/"+filename+init_params+
                      "_om"+optimize_m+"_ov"+optimize_v+"_s"+simplify_sequence+"_err\n")
        fh.writelines("#SBATCH --job-name=\""+filename+init_params+
                      "_om"+optimize_m+"_ov"+optimize_v+"_s"+simplify_sequence+"\"\n")
        # fh.writelines("#SBATCH --mail-user=\n")
        # fh.writelines("#SBATCH --mail-type=FAIL\n")
        # fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("module load python py-numpy py-scipy py-joblib"
        #     "ml-gpu/20230427\n")
        fh.writelines("{ time mpiexec -n %s python " % n_tasks +
                        file_dir + "/run.py -f %s -i %s -om %s -ov %s -s %s " % params  +
                        (" ; } 2> logs/time%sn%s%s_om%s_ov%s_s%s.txt \n") % (filename, n_tasks, init_params, optimize_m, optimize_v, simplify_sequence)
                    )
    """
    Executing the job.
    """
    os.system("sbatch %s" %job_file)

"""
Deleting the .sbatch file in the end.
"""
os.remove(job_file)
