import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils import shuffle

from utils import load_data, introduce_missing_data, normalization, renormalization
from utils import compute_rmse

from knnxkde import KNNxKDE
from GAIN.gain import gain
from softimpute.softimpute import softimpute


# ===============================
# CONFIGURATION
# ===============================
LIST_DATASETS = [
    '2d_linear',
    '2d_sine',
    '2d_ring',
    'geyser',
    'penguin',
    'planets',
]

OUT_DIR = 'output'
MISSING_SCENARIO = 'mar'

NB_REPEAT = 20
LIST_MISS_RATES = [0.2]
LIST_TAUS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
LIST_NB_NEIGHBORS = [1, 2, 5, 10, 20, 50, 100]
LIST_NB_TREES = [1, 2, 3, 5, 10, 15, 20]
LIST_NB_ITERS = [100, 200, 400, 700, 1000, 2000, 4000]
LIST_LAMBDAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


# ===============================
# INITIALIZE COMBINED CSV STORAGE
# ===============================
combined_summary_rows = []

# ===============================
# MAIN LOOP
# ===============================
for data_name in LIST_DATASETS:
    original_data = load_data(data_name)
    original_data = shuffle(original_data)
    print(f'\n=====================')
    print(f'|| ORIGINAL DATA NAME: {data_name} / SHAPE={original_data.shape}')
    print(f'=====================')
    print(f'({time.asctime()})')

    # Prepare RMSE dictionary
    rmse_dict = {
        'knnxkde': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_TAUS))),
        'knnimputer': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_NEIGHBORS))),
        'missforest': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_TREES))),
        'softimpute': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_LAMBDAS))),
        'gain': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_ITERS))),
        'mice': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'mean': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'median': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
    }

    # Loop over missing rates
    for i2, cur_miss_rate in enumerate(LIST_MISS_RATES):
        print(f'\n~~~ MISSING RATE = {cur_miss_rate} ~~~', flush=True)

        for i3 in range(NB_REPEAT):
            t0 = time.time()
            original_data = load_data(data_name)
            original_data = shuffle(original_data)

            # Introduce missingness
            miss_data = introduce_missing_data(
                original_data=original_data,
                miss_rate=cur_miss_rate,
                mode=MISSING_SCENARIO,
                data_name=data_name,
            )

            # Normalize
            norm_miss_data, norm_params = normalization(miss_data)
            norm_original_data, _ = normalization(original_data, parameters=norm_params)

            # --- kNNxKDE ---
            for i4, tau_val in enumerate(LIST_TAUS):
                print(f'Iteration {i3+1}/{NB_REPEAT} => kNNxKDE... {i4+1}/{len(LIST_TAUS)}', end='\r', flush=True)
                cur_tau = 1.0 / tau_val
                knnxkde = KNNxKDE(h=0.03, tau=cur_tau, metric='nan_std_eucl')
                norm_imputed_data = knnxkde.impute_mean(norm_miss_data)
                rmse_dict['knnxkde'][i2, i3, i4] = np.nan if norm_imputed_data is None else compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- kNNImputer ---
            for i4, neigh in enumerate(LIST_NB_NEIGHBORS):
                print(f'Iteration {i3+1}/{NB_REPEAT} => kNNImputer... {i4+1}/{len(LIST_NB_NEIGHBORS)}', end='\r', flush=True)
                knnimputer = KNNImputer(n_neighbors=neigh)
                norm_imputed_data = knnimputer.fit_transform(norm_miss_data)
                rmse_dict['knnimputer'][i2, i3, i4] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- MissForest ---
            for i4, nb_trees in enumerate(LIST_NB_TREES):
                print(f'Iteration {i3+1}/{NB_REPEAT} => MissForest... {i4+1}/{len(LIST_NB_TREES)}', end='\r', flush=True)
                estimator = ExtraTreesRegressor(n_estimators=nb_trees)
                missforest = IterativeImputer(estimator=estimator, max_iter=10, tol=2e-1, verbose=0)
                norm_imputed_data = missforest.fit_transform(norm_miss_data)
                rmse_dict['missforest'][i2, i3, i4] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- SoftImpute ---
            for i4, lam in enumerate(LIST_LAMBDAS):
                print(f'Iteration {i3+1}/{NB_REPEAT} => SoftImpute... {i4+1}/{len(LIST_LAMBDAS)}', end='\r', flush=True)
                norm_imputed_data = softimpute(norm_miss_data, lam)[1]
                rmse_dict['softimpute'][i2, i3, i4] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- GAIN ---
            for i4, nb_iter in enumerate(LIST_NB_ITERS):
                print(f'Iteration {i3+1}/{NB_REPEAT} => GAIN... {i4+1}/{len(LIST_NB_ITERS)}', end='\r', flush=True)
                gain_params = {"batch_size": 128, "hint_rate": 0.9, "alpha": 100, "iterations": nb_iter}
                norm_imputed_data = gain(norm_miss_data, gain_params)
                rmse_dict['gain'][i2, i3, i4] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- MICE ---
            mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, tol=2e-1, verbose=0)
            norm_imputed_data = mice.fit_transform(norm_miss_data)
            rmse_dict['mice'][i2, i3] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- Mean ---
            mean_imputer = SimpleImputer(strategy='mean')
            norm_imputed_data = mean_imputer.fit_transform(norm_miss_data)
            rmse_dict['mean'][i2, i3] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            # --- Median ---
            median_imputer = SimpleImputer(strategy='median')
            norm_imputed_data = median_imputer.fit_transform(norm_miss_data)
            rmse_dict['median'][i2, i3] = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)

            t1 = time.time()
            print(' ' * 60, end='\r')
            print(f'Iteration {i3+1}/{NB_REPEAT} -> time = {(t1 - t0):.3f} s', flush=True)


    # ===============================
    # SAVE PKL FILE
    # ===============================
    save_dir = f'{OUT_DIR}/{MISSING_SCENARIO}/rmse'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/rmse_{data_name}.pkl', 'wb') as f:
        pickle.dump(rmse_dict, f)
    print(f'\nSaved RMSE dictionary: {save_dir}/rmse_{data_name}.pkl')

    # ===============================
    # COMPUTE SUMMARY FOR CSV
    # ===============================
    print('Creating summary RMSE entries...')
    for method, values in rmse_dict.items():
        arr = np.array(values)
        if arr.ndim == 3:
            mean_rmse = np.nanmean(arr, axis=(1, 2))
        elif arr.ndim == 2:
            mean_rmse = np.nanmean(arr, axis=1)
        else:
            mean_rmse = np.nanmean(arr)

        for i_miss, miss_rate in enumerate(LIST_MISS_RATES):
            combined_summary_rows.append({
                'Dataset': data_name,
                'Method': method,
                'Missing_Rate': miss_rate,
                'Mean_RMSE': mean_rmse[i_miss] if isinstance(mean_rmse, np.ndarray) else mean_rmse
            })

    print(f'({time.asctime()})')

# ===============================
# SAVE ONE COMBINED CSV
# ===============================
df_combined = pd.DataFrame(combined_summary_rows)
save_dir = f'{OUT_DIR}/{MISSING_SCENARIO}/rmse'
csv_path = f'{save_dir}/rmse_summary_all.csv'
df_combined.to_csv(csv_path, index=False)
print(f'\n✅ Combined summary saved -> {csv_path}')
print(df_combined.head())

print('\nFINISH!')
print('Bye \\o/\\o/')
 