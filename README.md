# CS768-Learning-with-Graphs
Course Project - reproduction of results and ablation study of NOSMOG(https://openreview.net/pdf?id=Cs3r5KLdoj) paper

# Enviroment setup
create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
install the requirements
```bash
pip install -r requirements.txt
```
# Running student
python train_student.py \
  --dataset Citeseer \
  --mode transductive \
  --teacher GCN \
  --num_layers 4 \
  --hidden_dim 128 \
  --gt_weight 1.0 \
  --sl_weight 0.5 \
  --rsd_weight 0.1 \
  --adv_weight 0.3 \
  --pgd_eps 0.05 \
  --pgd_iters 5 \
  --temperature 2.0 \
  --device cuda

# Running teacher
python train_teacher.py \
  --num_runs 5 \
  --setting ind \
  --data_path ./data/citeseer \
  --model_name GAT \
  --num_layers 3 \
  --hidden_dim 256 \
  --drop_out 0.5 \
  --batch_sz 1024 \
  --learning_rate 0.005 \
  --output_path ./teacher_outputs

