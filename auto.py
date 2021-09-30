import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--device', default="cuda")
parser.add_argument('--nohupNN', required=True)
args = parser.parse_args()

for subject_id in range(1,10):
    CODE = f"nohup python -u {args.main} --name nohup{args.nohupNN}{subject_id}_{args.name}_subject{subject_id} --subject {subject_id} --device {args.device}" \
         + f" > nohup{args.nohupNN}{subject_id}_{args.name}_subject{subject_id}.out"
    print(CODE)
    os.system(CODE)
    
print("finish!")