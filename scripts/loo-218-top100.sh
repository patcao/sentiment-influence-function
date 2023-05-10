# 100 points of the top influence scores
python scripts/loo_top_influence.py --epochs 5 --output-dir results_10k_reg001/loo --config-path model_params/bert-epoch9-reg0.001-10000.yaml --influence-dir results_10k_reg001/influence --test-guid 218 --num-influence-points 100 --num-workers 1 --worker-id 1
