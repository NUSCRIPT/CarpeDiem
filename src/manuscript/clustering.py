import itertools

import numpy as np
import pandas as pd
import seaborn as sns

import umap
import pynndescent

import scipy.cluster.hierarchy
import scipy.spatial.distance
import scipy.stats

import sklearn.metrics
import matplotlib.pyplot as plt



import matplotlib as mpl
import matplotlib.pyplot as plt


import matplotlib.patheffects

def get_reference_data_columns():
    data_columns = [
        'ECMO_flag', 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag',
        'Temperature', 'Heart_rate', 'Systolic_blood_pressure',
        'Diastolic_blood_pressure', 'Mean_arterial_pressure',
        'Norepinephrine_flag',
        'Norepinephrine_rate', 'Respiratory_rate', 'Oxygen_saturation',
        'Urine_output', 'GCS_eye_opening', 'GCS_motor_response',
        'GCS_verbal_response', 'RASS_score', 'PEEP', 'FiO2', 'Plateau_Pressure',
        'Lung_Compliance', 'PEEP_changes', 'Respiratory_rate_changes',
        'FiO2_changes', 'ABG_pH', 'ABG_PaCO2', 'ABG_PaO2',
        'PaO2FIO2_ratio', 'WBC_count', 'Lymphocytes', 'Neutrophils',
        'Hemoglobin', 'Platelets', 'Bicarbonate', 'Creatinine', 'Albumin',
        'Bilirubin', 'CRP', 'D_dimer', 'Ferritin', 'LDH', 'Lactic_acid',
        'Procalcitonin',
    ]
    return data_columns


def index_by_patient_ids_icu_stay_day(df):
    """

    Will replace index with a unique identifier.

    Input:
        df    df with 'Patient_id', 'ICU_stay', 'ICU_day'

    Output:
        df   with index being composite of 'Patient_id', 'ICU_stay', 'ICU_day'

    """


    v = df['Patient_id'].astype(str) + '/' + df['ICU_stay'].astype(str) + '/' + df['ICU_day'].astype(str)
    if v.value_counts().max() > 1:
        raise AssertionError('Patient_id, ICU_stay, ICU_day is amiguous')
    else:
        df.index = v

    return df


def get_distances(data_mtx, approach):

    if approach == 'nan_euclidean':
        data_dist = sklearn.metrics.pairwise.nan_euclidean_distances(data_mtx)
    elif approach == 'euclidean':
        data_dist = sklearn.metrics.pairwise.euclidean_distances(data_mtx)
    else:
        raise AssertionError('Approach not implemented yet.')

    return data_dist


def get_tree(df_dist, approach):

    if approach == 'ward':
        tree = scipy.cluster.hierarchy.ward(df_dist[np.triu_indices_from(df_dist, k=1)])

    else:
        raise AssertionError('Approach not implemented.')

    return tree



def table_with_assignments(tree, labels, threshold_range=range(3, 41)):
    """
    Creates table with clusters assigned at different levels
    of cutoff.

    Input:
        tree   distance tree
        labels labels of records (e.g.: patient ids) provided in same order as distance matrix used for trees
        threshold_range   optional 

    """



    assignments = {}
    for max_cluster in threshold_range:
        assignments[max_cluster] = scipy.cluster.hierarchy.cut_tree(tree, max_cluster).ravel()
        
    out = pd.DataFrame(
        index=labels,
        data=assignments
    ).stack().rename_axis(['pt_day', 'max_cluster']).to_frame('cluster_id').reset_index()
    return out, assignments


def identify_related_features(data):
    """
    Indentify features that are similar at different cutoffs
    """


    data_corr_mtx = data.corr()
    na_idx = data_corr_mtx.isna()
    data_corr_mtx[na_idx] = 0

    HIGH_CORR_CUTOFFS = [0.5, 0.7]


    cutoff_groups = {}
    for cutoff in HIGH_CORR_CUTOFFS:
        groups = {}
        for c, cols in data_corr_mtx.apply(lambda x: x.index[(x.abs() > cutoff) & (x.abs() < 1)]).items():
            if cols.size > 0:
                group = None
                if c in groups:
                    group = groups[c]
                else:
                    for col in cols:
                        if col in groups:
                            group = groups[col]
                if group is None:
                    group = []
                group.append(c)
                groups[c] = group
                for col in cols:
                    group.append(col)
                    groups[col] = group
        for k, v in groups.items():
            groups[k] = tuple(set(v))
        groups = list(set(groups.values()))
        cutoff_groups[cutoff] = groups

    return cutoff_groups



def reweight_related_features(df_features, approach, groups):
    
    df_features = df_features.copy()
    data_columns = get_reference_data_columns()

    if approach=='square_root':

        for group in groups:
            group = pd.Series(group)
            group = group[group.isin(data_columns)]
            df_features.loc[:, group] /= len(group) ** 0.5

    elif approach=='mean_rank':

        for group in groups:
            group = pd.Series(group)
            group = group[group.isin(data_columns)]

            df_features.loc[
                :, '_AND_'.join(group.values)
            ] = df_features.loc[:, group].mean(1).rank(pct=True)
            df_features = df_features.drop(labels=group, axis='columns')

    return df_features





def get_sign_mortalities(df_assigned_clusters, df_with_mortality):

    out_df = df_assigned_clusters.copy()
    super_agg = []
    for max_cluster in range(3, 41):
        out_slice = out_df.loc[out_df.max_cluster.eq(max_cluster), :].set_index('pt_day').copy()
        out_slice['patient'] = df_with_mortality.Patient_id[out_slice.index]
        out_slice['binary_outcome'] = df_with_mortality.Binary_outcome[out_slice.index]

        clusters = out_slice.cluster_id.unique()
        results = []
        for c1, c2 in itertools.combinations(clusters, 2):
            cont = out_slice.loc[out_slice.cluster_id.isin([c1, c2]), :].groupby(
                ['cluster_id', 'binary_outcome']
            ).patient.nunique().to_frame('v').reset_index().pivot(
                index='cluster_id',
                columns='binary_outcome',
                values='v'
            ).reindex(columns=[0, 1], index=[c1, c2]).fillna(0)

            results.append(
                scipy.stats.fisher_exact(cont.values)[1]
            )

        d = pd.Series(results).to_frame('pval')
        d.loc[:, 'significant'] = d['pval'] < 0.01
        d.loc[:, 'max_cluster'] =  max_cluster

        super_agg.append(d)
        
    return pd.concat(super_agg)    




def histogram(columns, data, data_columns, data_mtx):

    cols = np.pad(columns, (0, 44 - columns.size), constant_values=np.nan).reshape(11, 4)
    fig, axes = plt.subplots(
            nrows=cols.shape[0], 
            ncols=cols.shape[1], 
            figsize=(16, 28), 
            gridspec_kw={"wspace": 0.4, "hspace": 0.5}
        )
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            column = cols[row, col]
            ax = axes[row, col]
            if isinstance(column, str) and column in data:
                ax.hist(
                    data_mtx[column], 
                    bins=50,
    #                 log=True
                )
                ax.set_title(column + f" {data_mtx[column].isna().sum() * 100 / data.shape[0]:.2f}% NA")
            else:
                ax.remove()


def heatmap(df_assigned_clusters, df_with_mortality, data_mtx, tree, col_tree):


    out=df_assigned_clusters.copy()
    data = df_with_mortality.copy()

    df = out[out['max_cluster'].isin(range(10, 16))].pivot(index='pt_day', columns='max_cluster', values='cluster_id').reindex(
        axis=data.index
    )
    df.columns = [f'cl_{c}' for c in df.columns]
    df = df.reset_index(drop=True)

    df_color = df.apply(
        lambda x: [mpl.colors.rgb2hex(mpl.cm.tab20(int(i))) if not np.isnan(i) else "white" for i in x]
    )

    df_color['Binary_outcome'] = data.Binary_outcome.map(
        {0: "skyblue", 1: "darkred"}
    )

    mpl.rcParams["figure.figsize"] = (16, 16)
    cg = sns.clustermap(
        data_mtx.T, 
        col_linkage=tree,
        row_linkage=col_tree,
        col_colors=df_color.to_numpy().T,
        xticklabels=False,
        yticklabels=data_mtx.columns,
        cmap="coolwarm",
        cbar_pos=None,
        vmin=0,
        vmax=1,
        standard_scale="row"
    )
    cg.ax_heatmap.collections[0].set_rasterized(True)
    cg.ax_col_colors.collections[0].set_rasterized(True)
    for y, i in enumerate(range(10, 16)):
        for c, x in (df[f"cl_{i}"][cg.dendrogram_col.reordered_ind].reset_index().reset_index()
                     .groupby(f"cl_{i}").level_0.mean().iteritems()):
            cg.ax_col_colors.text(x, y + 0.5, str(c + 1), ha="center", va="center")
    cg.ax_col_colors.set_yticks([i + 0.5 for i in range(7)])
    cg.ax_col_colors.set_yticklabels([f"{x} clusters" for x in range(10, 16)] + ["Mortality"])    



def show_umap(data, assignment):


    umap.umap_.check_array = lambda x, **kwargs: x
    nndescent = pynndescent.NNDescent(np.zeros((1, 1)))

    knn = umap.umap_.nearest_neighbors(
        data, 
            n_neighbors=7, 
            metric="precomputed",
            metric_kwds=None,
            angular=False,
            random_state=42
        )

    knn = (knn[0], knn[1], nndescent)

    umap_model = umap.UMAP(
        n_neighbors=7, 
        random_state=42, 
        precomputed_knn=knn,
        metric="precomputed",
        min_dist=0.2,
    )


    umap_data = umap_model.fit_transform(data)


    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        umap_data[:, 0], 
        umap_data[:, 1], 
        s=5, 
        c=[mpl.cm.tab20(i) for i in assignment],
        alpha=0.5
    )
    for c in np.unique(assignment):
        u1 = np.median(umap_data[assignment == c, 0])
        u2 = np.median(umap_data[assignment == c, 1])
        ax.text(u1, u2, c + 1, size=14, weight="bold", path_effects=[
            mpl.patheffects.Stroke(linewidth=3, foreground="white"),
            mpl.patheffects.Normal()
        ])
    ax.set_xlabel("UMAP1", size=16)
    ax.set_ylabel("UMAP2", size=16);

    return umap_data




def show_umap2(data, assignment):


    umap.umap_.check_array = lambda x, **kwargs: x
    nndescent = pynndescent.NNDescent(np.zeros((1, 1)))

    knn = umap.umap_.nearest_neighbors(
        data, 
            n_neighbors=7, 
            metric="precomputed",
            metric_kwds=None,
            angular=False,
            random_state=42
        )

    knn = (knn[0], knn[1], nndescent)

    umap_model = umap.UMAP(
        n_neighbors=2, 
        random_state=42, 
        precomputed_knn=knn,
        metric="precomputed",
        min_dist=0.1,
    )


    umap_data = umap_model.fit_transform(data)


    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        umap_data[:, 0], 
        umap_data[:, 1], 
        s=5, 
        c=[mpl.cm.tab20(i) for i in assignment],
        alpha=0.5
    )
    for c in np.unique(assignment):
        u1 = np.median(umap_data[assignment == c, 0])
        u2 = np.median(umap_data[assignment == c, 1])
        ax.text(u1, u2, c + 1, size=14, weight="bold", path_effects=[
            mpl.patheffects.Stroke(linewidth=3, foreground="white"),
            mpl.patheffects.Normal()
        ])
    ax.set_xlabel("UMAP1", size=16)
    ax.set_ylabel("UMAP2", size=16);

    return umap_data



def infer_clustermortality_and_add_cluster_id(data_with_mortality, data_mtx, assignment):

    data = data_with_mortality.copy()
    data_mtx = data_mtx.copy()
    data.loc[data_mtx.index, 'cluster'] = assignment


    cluster_mortality = data.copy()

    cluster_mortality = cluster_mortality.groupby("cluster").apply(
        lambda x: pd.Series(
            [x.Patient_id.nunique(), x.Patient_id[x.Binary_outcome.eq(1)].nunique()]
        )
    )

    cluster_mortality.rename({0: "Total", 1: "Died"}, axis=1, inplace=True)
    cluster_mortality["mortality"] = cluster_mortality.Died / cluster_mortality.Total
    cluster_mortality = cluster_mortality.sort_values("mortality").reset_index()\
                            .reset_index().set_index("cluster")

    data["cluster_order"] = cluster_mortality["index"][data.cluster].values
    data = data.sort_values("cluster_order")

    return data, cluster_mortality




def heatmap_by_mortality(data_with_mortality, data_mtx, assignment):

    
    data, cluster_mortality = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    display(cluster_mortality)


    COLUMN_ORDER = [
        # // Intubated
        'Intubation_flag', 
        # // Sedation
        'RASS_score',
        'GCS_eye_opening', 'GCS_motor_response', 'GCS_verbal_response',
        # // Lung injury severity
        'PaO2FIO2_ratio', 'ABG_PaO2', 
        'FiO2', 'PEEP', 'Oxygen_saturation', 
        'Plateau_Pressure', 'Lung_Compliance', 'ECMO_flag',
        # // Hemodynamics/shock
        'Norepinephrine_flag', 
        'Norepinephrine_rate', 'Mean_arterial_pressure',
        'Systolic_blood_pressure',
        'Diastolic_blood_pressure',
        'Lactic_acid', 'Hemoglobin', 
        'ABG_pH', 
        'ABG_PaCO2', 
        'Bicarbonate', 
        # // Renal
        'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', 'Urine_output', 
        # // Inflammatory biomarkers
        'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 
        'Bilirubin',  'Albumin', 'Lymphocytes',
        # // Vitals
        'Temperature', 
        'Heart_rate', 
        'Respiratory_rate',
        # // Instability
        'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes'
    ];


    df_color = pd.DataFrame(data.cluster_order.apply(
        lambda x: mpl.colors.rgb2hex(mpl.cm.tab20(int(x)))
    ))

    df_color['Binary_outcome'] = data.Binary_outcome.map(
        {0: "skyblue", 1: "darkred"}
    )

    mpl.rcParams["figure.figsize"] = (16, 16)
    cg = sns.clustermap(
        data_mtx.loc[data.index, COLUMN_ORDER].T, 
        col_cluster=False,
        row_cluster=False,
        col_colors=df_color.to_numpy().T,
        xticklabels=False,
        yticklabels=[x.replace("_", " ") for x in COLUMN_ORDER],
        cmap="coolwarm",
        cbar_pos=(0.07, 0.16, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        colors_ratio=0.025,
        standard_scale="row"
    )
    cg.ax_heatmap.collections[0].set_rasterized(True)
    cg.ax_col_colors.collections[0].set_rasterized(True)
    mort_x = []
    for i, (c, x) in enumerate(data.cluster_order.reset_index().reset_index()
                 .groupby("cluster_order").level_0.mean().iteritems()):
        cg.ax_col_colors.text(x, 0.4, str(i + 1), ha="center", va="center", size=14)
        mort_x.append(x)
    cg.ax_col_colors.set_yticks([0.5, 1.5])
    cg.ax_col_colors.set_yticklabels(["Cluster", "Mortality"], size=12)
    # cg.ax_col_colors.set_title("Rank-Euclidean", size=16)
    cg.ax_col_colors.set_ylim(0, 2)
    cg.ax_heatmap.tick_params(
        labelleft=True, left=True, right=False, labelright=False, labelsize=13
    )
    p = cg.ax_heatmap.get_position()
    p.x0 += 0.1
    p.x1 += 0.1
    cg.ax_heatmap.set_position(p)
    p = cg.ax_col_colors.get_position()
    p.y0 += 0.01
    p.y1 += 0.01
    p.x0 += 0.1
    p.x1 += 0.1
    cg.ax_col_colors.set_position(p)
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)
    # for y, (i, r) in enumerate(cluster_mortality.iterrows()):
    #     x = y // 8 * 2000
    #     y = 6 - y % 8 * 0.5
    #     cg.ax_col_colors.text(x, y, f"{int(i)}: {r[3]:.3f}")
    p = cg.ax_col_colors.get_position()
    mort_ax = cg.figure.add_axes((p.x0, p.y0 + 0.08, p.x1 - p.x0, 0.08))
    mort_ax.patch.set_visible(False)
    mort_ax.set_xlim(*cg.ax_heatmap.get_xlim())
    mort_ax.plot(mort_x, cluster_mortality.mortality, c="#999")
    mort_ax.scatter(mort_x, cluster_mortality.mortality, c="#999")
    mort_ax.set_ylim(0, 1)
    mort_ax.tick_params(bottom=False, labelbottom=False)
    mort_ax.set_ylabel("Mortality rate", size=12)
    mort_ax.spines[["bottom", "top", "right"]].set_visible(False)
    mort_ax.axhline(0, ls="--", c="#aaa")
    mort_ax.axhline(1, ls="--", c="#aaa")



def quilt(data_with_mortality, data_mtx, assignment):
    
    data_mtx = data_mtx.copy()
    data = data_with_mortality.copy()
    assignment = assignment.copy()

    data, _ = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    #data = data.loc[data_mtx.index]

    data.cluster = data.cluster_order + 1
    data.drop("cluster_order", axis="columns", inplace=True)

    # First element is name
    # Other elements are features with optional `-` in the beginning to flip their sign

    FEATURE_GROUPS = [
        ["Neurologic", '-RASS_score', '-GCS_eye_opening', '-GCS_motor_response', '-GCS_verbal_response'],
        ["Respiratory", 'Intubation_flag', '-PaO2FIO2_ratio', '-ABG_PaO2', 'FiO2', 'PEEP', 
         '-Oxygen_saturation', 'Plateau_Pressure', '-Lung_Compliance', 'ECMO_flag', 'Respiratory_rate'],
        ["Shock", 'Norepinephrine_flag', 'Norepinephrine_rate', '-Mean_arterial_pressure',
        '-Systolic_blood_pressure', '-Diastolic_blood_pressure', 'Lactic_acid', '-Hemoglobin', 
        '-ABG_pH', 'ABG_PaCO2', 'Heart_rate', '-Bicarbonate'],
        ["Renal", 'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', '-Urine_output'],
        ["Inflammatory", 'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 'Bilirubin',  '-Albumin', '-Lymphocytes', 'Temperature'],
        ["Ventilator instability", 'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes']
    ]

    df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.cluster.unique()))
    for group in FEATURE_GROUPS:
        for c in df.columns:
            for feature in group[1:]:
                sign = 1
                if feature.startswith("-"):
                    sign = -1
                    feature = feature[1:]
                idx = data.index[data.cluster.eq(c)]
                feature_min = data_mtx[feature].min()
                feature_max = data_mtx[feature].max()
                feature_values = data_mtx.loc[idx, feature]
                # feature_values = (feature_values - feature_min) / (feature_max - feature_min)
                feature_mean = feature_values.mean() * sign
                df.loc[group[0], c] += feature_mean

    cg = sns.clustermap(
        df,
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )
    cg.ax_heatmap.set_xlabel("Cluster", size=16, labelpad=10)
    cg.ax_heatmap.xaxis.set_label_position("top")
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)




def quilt2(data_with_mortality, data_mtx, assignment):


    data_mtx = data_mtx.copy()
    data = data_with_mortality.copy()
    assignment = assignment.copy()

    data, _ = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    #data = data.loc[data_mtx.index] tried to troubleshoot why quilt2 doesn't work

    data.cluster = data.cluster_order + 1
    data.drop("cluster_order", axis="columns", inplace=True)

    # First element is name
    # Other elements are features with optional `-` in the beginning to flip their sign

    FEATURE_GROUPS = [
        ["Neurologic", '-RASS_score', '-GCS_eye_opening', '-GCS_motor_response', '-GCS_verbal_response'],
        ["Respiratory", 'Intubation_flag', '-PaO2FIO2_ratio', '-ABG_PaO2', 'FiO2', 'PEEP', 
         '-Oxygen_saturation', 'Plateau_Pressure', '-Lung_Compliance', 'ECMO_flag', 'Respiratory_rate'],
        ["Shock", 'Norepinephrine_flag', 'Norepinephrine_rate', '-Mean_arterial_pressure',
        '-Systolic_blood_pressure', '-Diastolic_blood_pressure', 'Lactic_acid', '-Hemoglobin', 
        '-ABG_pH', 'ABG_PaCO2', 'Heart_rate', '-Bicarbonate'],
        ["Renal", 'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', '-Urine_output'],
        ["Inflammatory", 'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 'Bilirubin',  '-Albumin', '-Lymphocytes', 'Temperature'],
        ["Ventilator instability", 'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes']
    ]

    df = pd.DataFrame(0, index=data.index, columns=[x[0] for x in FEATURE_GROUPS])
    for group in FEATURE_GROUPS:
        for feature in group[1:]:
            sign = 1
            if feature.startswith("-"):
                sign = -1
                feature = feature[1:]
            idx = ~data_mtx[feature].isna()
            df.loc[idx, group[0]] += data_mtx.loc[idx, feature] * sign
            if f"{group[0]}_cnt" not in df.columns:
                df[f"{group[0]}_cnt"] = 0
            df.loc[idx, f"{group[0]}_cnt"] += 1

    worst_resp, worst_resp_cnt = df.Respiratory.max(), df.Respiratory_cnt[df.Respiratory.idxmax()]
    worst_renal, worst_renal_cnt = df.Renal.max(), df.Renal_cnt[df.Renal.idxmax()]
    df.loc[data.ECMO_flag.eq(1), "Respiratory"] = worst_resp
    df.loc[data.ECMO_flag.eq(1), "Respiratory_cnt"] = worst_resp_cnt
    df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal"] = worst_renal
    df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal_cnt"] = worst_renal_cnt
    quilt_df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.cluster.unique()))
    for group in FEATURE_GROUPS:
        for c in quilt_df.columns:
            idx = data.index[data.cluster.eq(c)]
            quilt_df.loc[group[0], c] = df.loc[idx, group[0]].sum() / df.loc[idx, f"{group[0]}_cnt"].sum()
    cg = sns.clustermap(
        quilt_df,
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )
    cg.ax_heatmap.set_xlabel("Cluster", size=16, labelpad=10)
    cg.ax_heatmap.xaxis.set_label_position("top")
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)





def quilt3(data_with_mortality, data_mtx, assignment): #weighting
    
    data_mtx = data_mtx.copy()
    data = data_with_mortality.copy()
    assignment = assignment.copy()

    data, _ = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    #data = data.loc[data_mtx.index]

    data.cluster = data.cluster_order + 1
    data.drop("cluster_order", axis="columns", inplace=True)

    # First element is name
    # Other elements are features with optional `-` in the beginning to flip their sign

    FEATURE_GROUPS = [
        ["Neurologic", '-RASS_score', '-GCS_eye_opening', '-GCS_motor_response', '-GCS_verbal_response'],
        ["Respiratory", 'Intubation_flag', '-PaO2FIO2_ratio', '-ABG_PaO2', 'FiO2', 'PEEP', 
         '-Oxygen_saturation', 'Plateau_Pressure', '-Lung_Compliance', 'Respiratory_rate',
         'ECMO_flag', 'ECMO_flag','ECMO_flag','ECMO_flag','ECMO_flag','ECMO_flag','ECMO_flag','ECMO_flag','ECMO_flag',],
        ["Shock", 'Norepinephrine_flag', 'Norepinephrine_rate', '-Mean_arterial_pressure',
        '-Systolic_blood_pressure', '-Diastolic_blood_pressure', 'Lactic_acid', '-Hemoglobin', 
        '-ABG_pH', 'ABG_PaCO2', 'Heart_rate', '-Bicarbonate'],
        ["Renal", 
        'CRRT_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Hemodialysis_flag', 
        'Creatinine', '-Urine_output'],
        ["Inflammatory", 'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 'Bilirubin',  '-Albumin', '-Lymphocytes', 'Temperature'],
        ["Ventilator instability", 'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes']
    ]

    df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.cluster.unique()))
    for group in FEATURE_GROUPS:
        for c in df.columns:
            for feature in group[1:]:
                sign = 1
                if feature.startswith("-"):
                    sign = -1
                    feature = feature[1:]
                idx = data.index[data.cluster.eq(c)]
                feature_min = data_mtx[feature].min()
                feature_max = data_mtx[feature].max()
                feature_values = data_mtx.loc[idx, feature]
                # feature_values = (feature_values - feature_min) / (feature_max - feature_min)
                feature_mean = feature_values.mean() * sign
                df.loc[group[0], c] += feature_mean

    cg = sns.clustermap(
        df,
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )
    cg.ax_heatmap.set_xlabel("Cluster", size=16, labelpad=10)
    cg.ax_heatmap.xaxis.set_label_position("top")
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)





def quilt4(data_with_mortality, data_mtx, assignment):


    data_mtx = data_mtx.copy()
    data = data_with_mortality.copy()
    assignment = assignment.copy()

    data, _ = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    #data = data.loc[data_mtx.index] tried to troubleshoot why quilt2 doesn't work

    data.cluster = data.cluster_order + 1
    data.drop("cluster_order", axis="columns", inplace=True)

    # First element is name
    # Other elements are features with optional `-` in the beginning to flip their sign

    FEATURE_GROUPS = [
        ["Neurologic", '-RASS_score', '-GCS_eye_opening', '-GCS_motor_response', '-GCS_verbal_response'],
        ["Respiratory", 'Intubation_flag', '-PaO2FIO2_ratio', '-ABG_PaO2', 'FiO2', 'PEEP', 
         '-Oxygen_saturation', 'Plateau_Pressure', '-Lung_Compliance', 'ECMO_flag', 'Respiratory_rate'],
        ["Shock", 'Norepinephrine_flag', 'Norepinephrine_rate', '-Mean_arterial_pressure',
        '-Systolic_blood_pressure', '-Diastolic_blood_pressure', 'Lactic_acid', '-Hemoglobin', 
        '-ABG_pH', 'ABG_PaCO2', 'Heart_rate', '-Bicarbonate'],
        ["Renal", 'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', '-Urine_output'],
        ["Inflammatory", 'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 'Bilirubin',  '-Albumin', '-Lymphocytes', 'Temperature'],
        ["Ventilator instability", 'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes']
    ]

    df = pd.DataFrame(0, index=data.index, columns=[x[0] for x in FEATURE_GROUPS])
    for group in FEATURE_GROUPS:
        for feature in group[1:]:
            sign = 1
            if feature.startswith("-"):
                sign = -1
                feature = feature[1:]
            idx = ~data_mtx[feature].isna()
            if feature in ("ECMO_flag", 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag'):
                df.loc[idx, group[0]] += data.loc[idx, feature] * sign
            else:
                df.loc[idx, group[0]] += data_mtx.loc[idx, feature] * sign
            if f"{group[0]}_cnt" not in df.columns:
                df[f"{group[0]}_cnt"] = 0
            df.loc[idx, f"{group[0]}_cnt"] += 1

    # worst_resp, worst_resp_cnt = df.Respiratory.max(), df.Respiratory_cnt[df.Respiratory.idxmax()]
    # worst_renal, worst_renal_cnt = df.Renal.max(), df.Renal_cnt[df.Renal.idxmax()]
    # df.loc[data.ECMO_flag.eq(1), "Respiratory"] = worst_resp
    # df.loc[data.ECMO_flag.eq(1), "Respiratory_cnt"] = worst_resp_cnt
    # df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal"] = worst_renal
    # df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal_cnt"] = worst_renal_cnt
    quilt_df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.cluster.unique()))


    for group in FEATURE_GROUPS:
        for c in quilt_df.columns:
            idx = data.index[data.cluster.eq(c)]
            quilt_df.loc[group[0], c] = df.loc[idx, group[0]].sum() / df.loc[idx, f"{group[0]}_cnt"].sum()
    cg = sns.clustermap(
        quilt_df,
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )
    cg.ax_heatmap.set_xlabel("Cluster", size=16, labelpad=10)
    cg.ax_heatmap.xaxis.set_label_position("top")
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)







def quilt5(data_with_mortality, data_mtx, assignment):


    data_mtx = data_mtx.copy()
    data = data_with_mortality.copy()
    assignment = assignment.copy()

    data, _ = infer_clustermortality_and_add_cluster_id(
        data_with_mortality, data_mtx, assignment)

    #data = data.loc[data_mtx.index] tried to troubleshoot why quilt2 doesn't work

    data.cluster = data.cluster_order + 1
    data.drop("cluster_order", axis="columns", inplace=True)

    # First element is name
    # Other elements are features with optional `-` in the beginning to flip their sign

    FEATURE_GROUPS = [
        ["Neurologic", '-RASS_score', '-GCS_eye_opening', '-GCS_motor_response', '-GCS_verbal_response'],
        ["Respiratory", 'Intubation_flag', '-PaO2FIO2_ratio', '-ABG_PaO2', 'FiO2', 'PEEP', 
         '-Oxygen_saturation', 'Plateau_Pressure', '-Lung_Compliance', 'ECMO_flag',  'Respiratory_rate'],
        ["Shock", 'Norepinephrine_flag', 'Norepinephrine_rate', '-Mean_arterial_pressure',
        '-Systolic_blood_pressure', '-Diastolic_blood_pressure', 'Lactic_acid', '-Hemoglobin', 
        '-ABG_pH', 'ABG_PaCO2', 'Heart_rate', '-Bicarbonate'],
        ["Renal", 'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', '-Urine_output'],
        ["Inflammatory", 'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
        'D_dimer', 'LDH', 'Ferritin', 'Bilirubin',  '-Albumin', '-Lymphocytes', 'Temperature'],
        ["Ventilator instability", 'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes']
    ]

    df = pd.DataFrame(0, index=data.index, columns=[x[0] for x in FEATURE_GROUPS])
    for group in FEATURE_GROUPS:
        for feature in group[1:]:
            sign = 1
            if feature.startswith("-"):
                sign = -1
                feature = feature[1:]
            idx = ~data_mtx[feature].isna()
            if feature in ("ECMO_flag", 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag'):
                df.loc[idx, group[0]] += data.loc[idx, feature] * sign
            else:
                df.loc[idx, group[0]] += data_mtx.loc[idx, feature] * sign
            if f"{group[0]}_cnt" not in df.columns:
                df[f"{group[0]}_cnt"] = 0
            df.loc[idx, f"{group[0]}_cnt"] += 1

    # worst_resp, worst_resp_cnt = df.Respiratory.max(), df.Respiratory_cnt[df.Respiratory.idxmax()]
    # worst_renal, worst_renal_cnt = df.Renal.max(), df.Renal_cnt[df.Renal.idxmax()]
    # df.loc[data.ECMO_flag.eq(1), "Respiratory"] = worst_resp
    # df.loc[data.ECMO_flag.eq(1), "Respiratory_cnt"] = worst_resp_cnt
    # df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal"] = worst_renal
    # df.loc[data.Hemodialysis_flag.eq(1) | data.CRRT_flag.eq(1), "Renal_cnt"] = worst_renal_cnt

    quilt_df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.cluster.unique()))
    for group in FEATURE_GROUPS:
        for c in quilt_df.columns:
            idx = data.index[data.cluster.eq(c)]
            quilt_df.loc[group[0], c] = df.loc[idx, group[0]].sum() / df.loc[idx, f"{group[0]}_cnt"].sum()
    cg = sns.clustermap(
        quilt_df.iloc[:4, :],
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg2 = sns.clustermap(
        quilt_df.iloc[4:, :],
        col_cluster=False,
        row_cluster=False,
        standard_scale="row",
        cmap="coolwarm",
        cbar_pos=(0.01, 0.38, 0.015, 0.05),
        cbar_kws={"ticks": []},
        vmin=0,
        vmax=1,
        linewidths=2,
    )
    cg.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )
    cg2.ax_heatmap.remove()
    cg2.ax_heatmap.figure = cg.figure
    plt.close(cg2.figure)
    cg2.ax_heatmap.tick_params(
        bottom=False, 
        labelbottom=False, 
        top=True, 
        labeltop=True, 
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        labelsize=14
    )

    cg.figure.axes.append(cg2.ax_heatmap)
    cg.figure.add_axes(cg2.ax_heatmap)
    dummy = cg.figure.add_subplot(111)
    cg2.ax_heatmap.set_position(dummy.get_position())
    dummy.remove()
    cg.ax_heatmap.set_xlabel("Cluster", size=16, labelpad=10)
    cg.ax_heatmap.xaxis.set_label_position("top")
    cg.cax.annotate("row min.", (0, -0.1), va="top", annotation_clip=False, fontsize=12)
    cg.cax.annotate("row max.", (0, 1.1), va="bottom", annotation_clip=False, fontsize=12)
    p = cg.ax_heatmap.get_position()
    p.y0 += 0.3
    cg.ax_heatmap.set_position(p)
    p.y0 -= 0.3
    p.y1 = p.y0 + 0.2
    cg2.ax_heatmap.set_position(p)
    print(p)





def feature_plot(data, value, value_name, cmap=None):
    if cmap is None:
        cmap = sns.color_palette("magma", as_cmap=True)
    df = data.sort_values(value)
    idx = df[value].isna()
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(
        df.umap1[idx],
        df.umap2[idx],
        s=20,
        c="#a9c4d3",
        ec=None,
        alpha=0.2
    )
    scatter = ax.scatter(
        df.umap1[~idx], 
        df.umap2[~idx], 
        s=20, 
        c=df.loc[~idx, value],
        ec=None,
        cmap=cmap,
        alpha=0.5
    )
    ax.set_xlabel("UMAP1", size=16)
    ax.set_ylabel("UMAP2", size=16)
    ax.collections[0].set_rasterized(True)
    ax.collections[1].set_rasterized(True)
    cbar_ax = ax.figure.add_axes((0.8, 0.8, 0.02, 0.1))
    cbar = ax.figure.colorbar(scatter, cax=cbar_ax)
    cbar_ax.set_title(value_name, loc="left")
    cbar.solids.set(alpha=1)
    ax.legend(
        [mpl.patches.Patch(fc="#a9c4d3", ec="#333")],
        ["NA"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.8)
    )
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_clinical_course(data, pt, line_color="firebrick"):
    day_cmap = sns.dark_palette(line_color, reverse=True, as_cmap=True)
    df = data.loc[data.Patient_id == pt, :]
    fig, axes = plt.subplots(
        figsize=(13, 3), 
        nrows=2, 
        sharex=True, 
        gridspec_kw=dict(height_ratios=(1, 5)),
        constrained_layout=True
    )
    ax = axes[0]
    ax.set_xlim(df.ICU_day.min() - 1.2, df.ICU_day.max() + 0.2)
    ax.set_ylim(-0.2, 1.2)
    GAP = 0.2
    prev = None
    span_l = 0
    for _, r in df.iterrows():
        if prev is None:
            prev = r
        if prev.clusters != r.clusters:
            start = r.ICU_day - span_l - 1
            ax.add_patch(mpl.patches.Rectangle(
                (start, 0),
                width=span_l - GAP,
                height=1,
                fc=mpl.cm.tab20(int(prev.clusters - 1)),
                ec="#333"
            ))
            span_x = start + span_l / 2 - GAP / 2
            ax.text(span_x, 0.48, prev.clusters, ha="center", va="center", fontsize=12, weight="bold")
            span_l = 1
            prev = r
        else:
            span_l += 1
    start = r.ICU_day - span_l
    ax.add_patch(mpl.patches.Rectangle(
        (start, 0),
        width=span_l,
        height=1,
        fc=mpl.cm.tab20(int(r.clusters - 1)),
        ec="#333",
        lw=1
    ))
    ax.text(start + span_l / 2, 0.48, r.clusters, ha="center", va="center", fontsize=12, weight="bold")
    ax.set_yticks([0.5])
    ax.set_yticklabels(["ICU state"], size=14)
    ax.tick_params(bottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"Patient {pt}", size=16)
    ax = axes[1]
    points = np.array([df.ICU_day - 1, df.SOFA_score]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments, cmap=day_cmap, lw=2, capstyle="round")
    lc.set_array((df.ICU_day - 1) / df.shape[0])
    ax.add_collection(lc)
    ax.set_ylim(-0.2, df.SOFA_score.max() + 0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("SOFA\nscore", size=16)
    ax.set_xlabel("ICU day", size=16)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator())
    for _, r in df.iterrows():
        if r.has_bal:
            ax.axvline(r.ICU_day - 1, ymax=0.9, c="gray")
    return ax




def plot_clinical_course_umap(data, pt, line_color="firebrick", first_day_offset=(0, 0), last_day_offset=(0, 0)):
    day_cmap = sns.dark_palette(line_color, reverse=True, as_cmap=True)
    df = data.loc[data.Patient_id == pt, :]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    ax.scatter(
        data.umap1,
        data.umap2,
        c=[mpl.cm.tab20(i - 1) for i in data.clusters],
        s=2,
        alpha=0.4
    )
    ax.scatter(
        df.umap1,
        df.umap2,
        c=[mpl.cm.tab20(i - 1) for i in df.clusters],
        s=10,
        alpha=1,
        ec=[day_cmap(x) for x in (df.ICU_day - 1) / df.ICU_day.max()],
        lw=1.5
    )
    ax.set_xlabel("UMAP1", size=10)
    ax.set_ylabel("UMAP2", size=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelleft=False, labelbottom=False)
    for i in range(1, df.shape[0]):
        this_coord = df.umap1[i - 1], df.umap2[i - 1]
        next_coord = df.umap1[i], df.umap2[i]
        dist = ((this_coord[0] - next_coord[0])**2 + (this_coord[1] - next_coord[1])**2)**0.5
        if dist > 1:
            ax.annotate("", xytext=this_coord, xy=next_coord, arrowprops=dict(
                arrowstyle="->,head_width=0.3,head_length=0.3", 
                ec=day_cmap(i / df.shape[0]), 
                lw=1.5, 
                shrinkA=2, 
                shrinkB=2
            ))
        if i == 1:
            ax.annotate(i - 1, xy=this_coord, xytext=first_day_offset, 
                        textcoords="offset pixels", size=14)
        if i == df.shape[0] - 1:
            ax.annotate(i, xy=next_coord, xytext=last_day_offset, textcoords="offset pixels", size=14)
    ax.collections[0].set_rasterized(True)
    return ax



def stacked_hue_barplot(df, x, y, hue, stacks, ax, palette=None):
    if palette is None:
        palette = mpl.cm.tab10
    bar_gap = 1
    bar_width = 2
    col_pad = 2
    n_bars = len(stacks)
    n_cols = df[x].nunique()
    col_width = n_bars * bar_width + (n_bars - 1) * bar_gap + 2 * col_pad
    col_values = pd.Series(df[x].unique()).sort_values().reset_index(drop=True)
    hue_values = pd.Series(df[hue].unique()).sort_values().reset_index(drop=True)
    bar_values = pd.Series(range(len(stacks))).sort_values().reset_index(drop=True)
    to_display = pd.DataFrame(dict(
        col=np.repeat(col_values, n_bars), 
        bar=np.tile(bar_values, n_cols),
        bar_num=np.tile(bar_values.index, n_cols)
    )).reset_index().rename({"index": "col_num"}, axis=1)
    to_display["bar_pos"] = to_display.apply(
        lambda x: x.col_num * col_width + x.bar_num * bar_width + (x.bar_num - 1) * bar_gap + col_pad, 
        axis=1
    )
    max_stack = max([len(group) for group in stacks])
    bottom = np.zeros(to_display.shape[0])
    for i in range(max_stack):
        curr_stack = {j: group[-i - 1] for j, group in enumerate(stacks) if i < len(group)}
        to_display["bar_value"] = to_display.bar.map(curr_stack)
        count = to_display.merge(
            df.loc[df[hue].isin(curr_stack.values()), :], 
            left_on=["col", "bar_value"], 
            right_on=[x, hue],
            how="left"
        )[y].fillna(0)
        ax.bar(
            to_display.bar_pos, 
            count, 
            color=[palette(hue_values.index[hue_values == i]) 
                   for i in to_display.bar_value.fillna(hue_values.values[0])],
            ec="#333333",
            width=bar_width,
            align="edge",
            bottom=bottom
        )
        bottom += count
    ax.set_xticks(col_values.index * col_width + col_width / 2 + -1 * bar_gap)
    ax.set_xticklabels(col_values, size=16)
    ax.set_ylim((0, bottom.max() + bottom.max() * 0.1))

    bar_handles = []
    for i, b in enumerate(hue_values):
        bar_handles.append(mpl.patches.Patch(color=palette(i), label=b, ec="#333333"))
    ax.legend(
        handles=bar_handles, 
        loc="upper right", 
        title="Discharge disposition", 
        frameon=False,
        fontsize=14,
        title_fontsize=12
    )
    return ax


