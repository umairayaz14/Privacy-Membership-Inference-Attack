Assignment 1: Privacy-Membership-Inference-Attack 
Team: atml_team072

How to Reproduce the Best Result

1. Clone the repository

git clone https://github.com/umairayaz14/Privacy-Membership-Inference-Attack.git

2. Install dependencies

pip install -r requirements.txt

3. Download the data and model

wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
"https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
"https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"

4. Run the attack

python task_template.py

This will generate submission.csv and automatically submit to the leaderboard.

Expected Process:
The script will sequentially train the 8 shadow models for 20 epochs, extract the logit-scaled confidence scores, apply the offline calibration strategy, and generate a submission.csv file. It will then automatically submit this file to the evaluation server using the embedded API key.

Note on Ongoing Experiments:
It is worth noting that a scaled-up configuration of 128 shadow models trained for 50 epochs is currently executing on a compute cluster; should this job complete and yield a higher score prior to the final deadline, that updated result will stand as our final submission.
