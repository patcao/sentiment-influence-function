# 100 points of the top influence scores
python scripts/loo_top_influence.py --epochs 6 --learning-rate 0.0001 --output-dir results_10k_reg001/loo6 --config-path model_params/bert-epoch9-reg0.001-10000.yaml --influence-dir results_10k_reg001/influence --test-guid 218 --num-influence-points 50 --num-workers 1 --worker-id 1


# Retrain from the start
# python scripts/loo_top_influence.py --epochs 4 --output-dir results_10k_reg001/loo4 --config-path model_params/bert-classifier-init.yaml --influence-dir results_10k_reg001/influence --test-guid 218 --num-influence-points 50 --num-workers 1 --worker-id 1
