import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--device', default="cuda")
parser.add_argument('--nohupNNN', required=True)
parser.add_argument('--seed', default="n")
parser.add_argument('--sparse_lambda', default=None)
args = parser.parse_args()

for subject_id in range(1,10):
    CODE = f"nohup python -u {args.main} --name nohup{args.nohupNNN}{subject_id}_{args.name}_subject{subject_id} --subject {subject_id} --device {args.device} --seed {args.seed}"
    
    if args.sparse_lambda != None:
        CODE += f" --sparse_lambda {args.sparse_lambda}" 
    
    CODE += f" > nohup{args.nohupNNN}{subject_id}_{args.name}_subject{subject_id}.out"
    
    print(CODE)
    os.system(CODE)
    
print("finish!")

