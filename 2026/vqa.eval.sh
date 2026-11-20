
## call the scoring script on runs (run names without json extension) and save in output folder
## GT includes human AND AI alternative correct answers
python3 score_vqa_answer_generation.py --ground-truth vqa-ag-GT.human.AI.json --runs-dir vqa.ag.runs --output-dir vqa-ag-outputs
