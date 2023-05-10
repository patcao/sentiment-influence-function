# 100 influence points around the 30th percentile
python scripts/loo_top_influence.py --output-dir results_10k_reg001/loo --config-path model_params/bert-epoch9-reg0.001-10000.yaml --influence-dir results_10k_reg001/influence --test-guid 218 --num-influence-points 10000 --num-workers 100 --worker-id 30

# Compute influence functions if finished
python scripts/compute_influence_functions.py --output-dir results_10k_reg001/influence --config-path model_params/bert-epoch9-reg0.001-10000.yaml --num-workers 1 --worker-id 1
