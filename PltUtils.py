import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import subprocess

def plot_slice_sim_distrib(sim_data_frame, slice, errorbar_arg):
    metrics=sim_data_frame.columns[3:]
    s_data=sim_data_frame.query(f'slice_id=="{slice}"')
    print(f's_data:\n{s_data}')
    print(f'metrics={metrics}')
    f, axs = plt.subplots(len(metrics), figsize=(7, len(metrics)), layout="tight")
    for ax, x_var in zip(axs, metrics):
        sns.pointplot(data=s_data, x=s_data[x_var], errorbar=errorbar_arg, ax=ax, color='purple', alpha=0.7)
        sns.stripplot(data=s_data, x=s_data[x_var], jitter=True, ax=ax, color='green', size=3, alpha=0.8)
    plt.savefig(f"slice_{slice}_sim_scores_distrib.png")
    plt.show()

def plot_all_vols_sim_metrics(sim_df, sim_metric):
    vols_df=pd.DataFrame(sim_df['vol_id'].unique())
    g=sns.FacetGrid(vols_df, col=0, col_wrap=6, sharex=False)
    for ax,vol_id in zip(g.axes, vols_df[0]):
        print(f'vol={vol_id}')
        v_data=sim_df.query(f'vol_id=="{vol_id}"')
        plot=sns.scatterplot(data=v_data, x='index', y=sim_metric, ax=ax, size=1, palette='husl', hue=sim_metric, legend=False)
        # plot=sns.scatterplot(data=v_data, x='index', y=sim_metric, ax=ax, size=1, palette='ch:r=-.5, l=.75', hue=sim_metric, legend=False)
        g.tight_layout()
    plt.savefig(f"all_vols_sim_metric_{sim_metric}.png")
    plt.show()

def plot_per_vol_sim_metrics(sim_df, vol_id):
    v_data=sim_df.query(f'vol_id=="{vol_id}"')
    metric_names_df=pd.DataFrame(v_data.columns[3:])
    metric_names_df.columns = ["metric_name"]
    metric_names_df.reset_index(inplace=True)
    g = sns.FacetGrid(metric_names_df, col='metric_name', col_wrap=4, sharey=False, sharex=True)
    for ax, y_var in zip(g.axes, metric_names_df["metric_name"]):
        sns.scatterplot(data=v_data, x='index', y=y_var, ax=ax, hue=y_var, palette="husl", legend=False, alpha=1, size=1)
        g.tight_layout()
    plt.savefig(f"vol_{vol_id}_sim_metrics.png")
    plt.show()

def plot_iqr_outliers(sim_df, vol_id, sim_metric, rootDir):
    v_data=sim_df.query(f'vol_id=="{vol_id}"')[['index', 'vol_id', 'slice_id', sim_metric]]
    q1 = v_data[sim_metric].quantile(0.25)
    q3 = v_data[sim_metric].quantile(0.75)
    iqr = q3 - q1
    print(f'q1={q1}, q3={q3}, iqr={iqr}')
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print(f'lower={lower}, upper={upper}')
    outliers = v_data.query(f'{sim_metric} < {lower} | {sim_metric} > {upper}')
    print(f'outliers:\n{outliers}')
    for _, row in outliers.iterrows():
        subprocess.run(["open", f'{rootDir}/{row["vol_id"]}/{row["slice_id"]}.png'])

def plot_lower_quantile_outliers(sim_df, vol_id, sim_metric, rootDir):
    v_data=sim_df.query(f'vol_id=="{vol_id}"')[['index', 'vol_id', 'slice_id', sim_metric]]
    q1 = v_data[sim_metric].quantile(0.25)
    outliers = v_data.query(f'{sim_metric} < {q1}').sort_values(by=[sim_metric], ascending=True)
    print(f'outliers:\n{outliers}')
    for _, row in outliers.iterrows():
        subprocess.run(["open", f'{rootDir}/{row["vol_id"]}/{row["slice_id"]}.png'])
        yn = input("Continue? (y/n)")
        if yn == 'n' or yn == 'N':
            break

def plot_outliers_below_threshold(sim_df, vol_id, sim_metric, rootDir, threshold):
    v_data=sim_df.query(f'vol_id=="{vol_id}"')[['index', 'vol_id', 'slice_id', sim_metric]]
    outliers = v_data.query(f'{sim_metric} < {threshold}').sort_values(by=[sim_metric], ascending=True)
    print(f'outliers:\n{outliers}')
    for _, row in outliers.iterrows():
        subprocess.run(["open", f'{rootDir}/{row["vol_id"]}/{row["slice_id"]}.png'])
        yn = input("Continue? (y/n)")
        if yn == 'n' or yn == 'N':
            break

def thresholded_outliers_summary(sim_df, sim_metric, threshold):
    vol_ids=sim_df['vol_id'].unique()
    for vid in vol_ids:
        ssim_lo=sim_df.query(f'vol_id=="{vid}" & {sim_metric}<{threshold}')[['vol_id', 'slice_id', 'sksim']]
        print(f'{vid}: ssim_lo_count={len(ssim_lo)}')

if __name__ == "__main__":
    help_str = """
    per_vol_sim_metrics: plot how all similarity metrics vary across slices in a volume
        Required: vol_id
        Eg: python PltUtils.py --rootDir . --task per_vol_sim_metrics  --vol_id I238629

    all_vols_sim_metric: Plot how a sim_metric varies across all slices and across all volumes 
        Required: sim_mtrics
        Eg:  python PltUtils.py --rootDir . --task all_vols_sim_metric --sim_metric sksim

    slice_sim_distrib: plot how a slice's similarity metrics vary across all volumes
        Required: slice_id
        Eg:  python PltUtils.py --rootDir . --task slice_sim_distrib --slice_id 128_90_128

    outliers: plot outliers in a volume's similarity metric
        Required: vol_id, sim_metric
        Eg:  python PltUtils.py --rootDir . --task outliers --vol_id I238629 --sim_metric sksim
    """
    args = argparse.ArgumentParser(help_str)
    args.add_argument("--task",
                      type=str,
                      required=True,
                      help = "per_vol_sim_metrics | all_vols_sim_metric | slice_sim_distrib|outliers")
    args.add_argument("--vol_id", type=str, required=False, help="Required if task==all_slices_metrics_in_a_vol|all_sim_metrics_in_vol")
    args.add_argument("--slice_id", type=str, required=False, help="Required if task==sliceid_metrics_across_all_vols")
    args.add_argument("--sim_metric", type=str, required=False, help="Required if task==allvols_sim_metric.\n can be sim|psnr|fid|mssim|lpips|vsi")
    args.add_argument("--rootDir", type=str, required=True)
    args = args.parse_args()

    #Input validation
    if args.task == "per_vol_sim_metrics":
        if args.vol_id is None:
            print("Please provide vol_id")
            exit()
    elif args.task == "all_vols_sim_metric":
        if args.sim_metric is None:
            print("Please provide sim_metric: sim|psnr|mssim|fid|mssim|lpips|vsi")
            exit()
    elif args.task == 'slice_sim_distrib':
        if args.slice_id is None:
            print("Please provide slice_id")
            exit()
    elif args.task == 'outliers_lower_quartile' or args.task == 'outliers_iqr':
        if (args.vol_id is None) or (args.sim_metric is None):
            print("Please provide vol_id and a sim_metric to fetch outliers")
            exit()
    elif args.task == 'thresholded_outliers':
        if (args.vol_id is None) or (args.sim_metric is None) or (args.threshold is None):
            print("Please provide vol_id and a sim_metric to fetch outliers")
            exit()

    sim_scores = []
    for d in os.listdir(args.rootDir):
        dpath=os.path.join(args.rootDir, d)
        if os.path.isdir(dpath):
            slice_sim_scores_fpath = os.path.join(dpath, "slice_sim_scores.json")
            if os.path.exists(slice_sim_scores_fpath):
                with open(slice_sim_scores_fpath) as f:
                    slice_sim_scores = json.load(f)
                sim_scores += [[d] + s for s in slice_sim_scores]

    sim_df = pd.DataFrame(sim_scores)
    # if 'sewar_sim' in sim_df.columns:
        # del sim_df['sewar_sim']
    # print(sim_df)
    sim_df.columns = ["vol_id", "slice_id", "sksim", "piqa_sim", "sewar_sim", "psnr", "fid", "mssim", "lpips", "vsi"]
    sim_df.reset_index(inplace=True)

    if args.task == 'per_vol_sim_metrics':
        plot_per_vol_sim_metrics(sim_df, args.vol_id)
    elif args.task == "all_vols_sim_metric":
        plot_all_vols_sim_metrics(sim_df, args.sim_metric)
    elif args.task == "slice_sim_distrib":
        plot_slice_sim_distrib(sim_df, args.slice_id, 'sd')
    elif args.task == 'outliers_iqr':
        plot_iqr_outliers(sim_df, args.vol_id, args.sim_metric, args.rootDir)
    elif args.task == 'outliers_lower_quartile':
        plot_lower_quantile_outliers(sim_df, args.vol_id, args.sim_metric, args.rootDir)
    elif args.task == 'outliers':
        plot_outliers_below_threshold(sim_df, args.vol_id, args.sim_metric, args.rootDir, args.threshold)
    else:
        print("Please provide valid task")
