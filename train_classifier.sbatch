#!/bin/bash
#SBATCH --job-name=train_classifier    # Job name
#SBATCH --partition=spgpu                # GPU partition
#SBATCH --account=hafiz1               # Great Lakes account name
#SBATCH --gpus-per-node=1              # GPU per node
#SBATCH --mem=40G                      # CPU RAM
#SBATCH --cpus-per-task=4             # CPU cores (matches your num_workers)
#SBATCH --time=01:00:00                # time requested (hh:mm:ss)
#SBATCH --output=vtrain_classifier_log/%x-%j.log             # (%x=job_name, %j=job_id)

# --- Email Notifications (Optional but Recommended) ---
#SBATCH --mail-user=jsudan@umich.edu  # Replace with your email
#SBATCH --mail-type=BEGIN,END,FAIL           # Send email on job start, end, and failure

#----------------------------------------------------------------#
#  SETUP THE ENVIRONMENT
#----------------------------------------------------------------#
echo "Setting up the environment..."
echo "Job started on $(hostname) at $(date)"

# Load necessary modules. This ensures a clean and consistent environment.
module purge
module load python/3.9.12
module load cuda/11.8.0

# Activate your Python virtual environment
source ~/myenv/bin/activate

# Print some diagnostic information to the log file for debugging
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"


#----------------------------------------------------------------#
#  RUN THE SCRIPT
#----------------------------------------------------------------#
echo "Starting the visualize_subspace script..."

# Navigate to your project directory if the script is not in your home directory
cd /home/jsudan/deepfake_moe

# Execute your main training script
python train_classifier.py --encoder style
python train_classifier.py --encoder linguistic

echo "----------------------------------------------------------------"
echo "visualize_subspace script finished and results are saved."
echo "Job finished at: $(date)"
