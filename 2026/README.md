# VQA
Video Question Answering Evaluation Based on TREC VQA 2026 guidelines (https://www-nlpir.nist.gov/projects/tv2026/vqa.html)

# Task 1 (Answer Generation) Validator
Now available is a validator for task 1 (AG) runs. Just run the script validate.ag.run_2026.py and pass the testing query json file and your run. It generates a report stating if the run passes or if there are errors that need to be fixed in your run.

NOTE: the validator index queries starting from 0, while the testing queries starts from Q_ID = 1. Please take this into consideration when debugging an error reported by the validator (e.g. you may need to skip to the next query to find the error)

# Task 2 (Multiple Choice) Validator
The script validate_vqa_mc_2026.py can be used to validate your runs. It takes as input the query file and your run and report any errors if they exist.

