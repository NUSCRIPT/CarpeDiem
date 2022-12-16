#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects
import seaborn as sns
import numpy as np
import itertools
import scipy.stats
import statsmodels.stats.multitest
import statannotations.Annotator
import math
import os
import time

import sys



sys.path.append('./../src/')
# from manuscript import sankey_side_by_side as sankey
from manuscript import clustering, datasets, inout, export

pd.options.display.max_columns = 200
mpl.rcParams["figure.figsize"] = (10, 8)
mpl.rcParams['pdf.fonttype'] = 42  # edit-able in illustrator
# mpl.rcParams['font.sans-serif'] = "Arial"
# mpl.rcParams["font.family"] = "sans-serif"



def over_04_clustering(user, outstem, data):

    data = data.copy()

    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    # get_ipython().run_line_magic('load_ext', 'autoreload')
    # get_ipython().run_line_magic('autoreload', '2')
    # get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    # In[2]:


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sys.path.append('./../src/')
    from manuscript import sankey_side_by_side as sankey
    from manuscript import clustering, datasets, inout, export

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")


    # In[3]:


    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/04_clustering'    # name of notebook
    save = True


    skip_part_of_code = True

    # In[4]:


    def dump_figure(name):
        if save:
            export.image(
                user,
                f'{outfolder}/{name}',
            )


    # In[5]:


    def dump_table(df, name):
        if save:
            export.full_frame(
                user, 
                f'{outfolder}/{name}', 
                df, 
                index=True,
                date=False
            )


    # # Get Data, as in reference

    # In[6]:


    # data = pd.read_csv(
    #     inout.get_material_path('general/03_overwrite_PF_Cr/03data-external_221104_1136.csv.gz'), 
    #     index_col=0)


    # In[7]:


    data.shape


    # In[8]:


    # data = data.reset_index()


    # List of columns for clustering

    # In[9]:


    data_columns = clustering.get_reference_data_columns()


    # Get data that we will run clustering on

    # In[10]:


    data_mtx_orig = data[data_columns].copy()


    # ## 0. Preparation of shared aspects

    # Let's create groups of variables which share those high correlations. Let's try different cutoffs for high correlation, as they produce different results

    # In[11]:


    cutoff_groups_on_orig = clustering.identify_related_features(data_mtx_orig)


    # In[12]:


    data_mtx_as_pct = data_mtx_orig.rank(axis=0, pct=True)
    data_dist_col = clustering.get_distances(
        data_mtx_as_pct.transpose(), approach='nan_euclidean')   
    col_tree = clustering.get_tree(data_dist_col, approach='ward')


    # In[13]:


    threshold_for_relatedness = 0.7
    cutoff_groups = cutoff_groups_on_orig[threshold_for_relatedness]


    # In[14]:


    approaches = {}


    # ## 1. Similarity approach



    # In[15]:


    data_mtx = data_mtx_orig.copy()
    data_mtx = data_mtx.rank(axis=0, pct=True)

    data_mtx_for_similarity = data_mtx.copy()
    data_mtx_for_similarity = clustering.reweight_related_features(
        data_mtx_for_similarity, 
        approach='mean_rank', 
        groups=cutoff_groups)


    # In[16]:


    corr_mtx = data_mtx_for_similarity.transpose().corr("pearson")
    data_dist = clustering.get_distances(corr_mtx, approach='euclidean')   
    tree = clustering.get_tree(df_dist=data_dist, approach='ward')


    # In[17]:


    out, assignments = clustering.table_with_assignments(
        tree=tree,
        labels=data.index
    )


    # In[18]:


    sign_mortality_similarity = clustering.get_sign_mortalities(
        df_assigned_clusters=out,
        df_with_mortality=data
    )
    sign_mortality_similarity.loc[:, 'approach'] = 'similarity'


    # In[19]:


    approaches = {
        'Similarity' : {
            'feature_matrix': data_mtx,   # data_mtx_orig.rank(axis=0, pct=True),
            'data_dist':data_dist,
            'tree':tree,
            'assignments_table':out,
            'assignments':assignments
        }
    }


    if not skip_part_of_code:

        # ## 2. Rank-euclidean approach

        # In[20]:


        data_mtx = data_mtx_orig.copy()
        data_mtx = data_mtx.rank(axis=0, pct=True)

        data_mtx = data_mtx.copy()
        data_mtx = clustering.reweight_related_features(
            data_mtx, 
            approach='square_root', 
            groups=cutoff_groups)


        # In[21]:


        rank_data_dist = clustering.get_distances(data_mtx, approach='nan_euclidean')
        rank_tree = clustering.get_tree(rank_data_dist, approach='ward')
        rank_out, rank_assignments = clustering.table_with_assignments(
            tree=rank_tree,
            labels=data.index
        )

        sign_mortality_rank = clustering.get_sign_mortalities(
            df_assigned_clusters=rank_out, 
            df_with_mortality=data)
        sign_mortality_rank.loc[:, 'approach'] = 'rank-euclidean'


        # In[22]:


        approaches['Ranked-Euclidean'] = {
                'feature_matrix':data_mtx,
                'data_dist':rank_data_dist,
                'tree':rank_tree,
                'assignments_table':rank_out,
                'assignments':rank_assignments
        }


        # ## 3. Normalized-euclidean approach 

        # In[23]:


        data_mtx = data_mtx_orig.copy()


        # In[24]:


        to_winsor_both = [
            'Temperature', 'Heart_rate', 'Systolic_blood_pressure',
            'Diastolic_blood_pressure', 'Mean_arterial_pressure',
            'Respiratory_rate', 'PEEP', 'Plateau_Pressure',
            'ABG_pH', 'ABG_PaCO2', 'ABG_PaO2', 'PaO2FIO2_ratio',
            'Hemoglobin', 'Bicarbonate', 'Albumin', 'LDH',
            'Lactic_acid'
        ]
        to_winsor_right = [
            'Norepinephrine_rate', 'Urine_output',
            'Lung_Compliance', 'WBC_count', 'Lymphocytes',
            'Neutrophils', 'Platelets', 'Creatinine',
            'Bilirubin', 'CRP', 'D_dimer', 'Ferritin',
            'Procalcitonin'
        ]
        to_winsor_left = [
            'Oxygen_saturation'
        ]


        WINSOR_THRESHOLD_PCT = 1

        for column in data_columns:
            col = data_mtx[column].dropna()
            lower = np.percentile(col, WINSOR_THRESHOLD_PCT)
            upper = np.percentile(col, 100 - WINSOR_THRESHOLD_PCT)
            if column in to_winsor_both or column in to_winsor_left:
                data_mtx.loc[data_mtx[column] < lower, column] = lower
            if column in to_winsor_both or column in to_winsor_right:
                data_mtx.loc[data_mtx[column] > upper, column] = upper
                
            

        to_log2 = [
            'Norepinephrine_rate', 'ABG_PaO2', 
            'Creatinine', 'Bilirubin', 'D_dimer', 
            'Ferritin', 'LDH', 'Lactic_acid',
            'Procalcitonin'
        ]

        for c in to_log2:
            data_mtx[c] = np.log2(data_mtx[c])

        to_quantize = [
            'ECMO_flag', 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag',
            'Norepinephrine_flag',
            'GCS_eye_opening', 'GCS_motor_response',
            'GCS_verbal_response', 'RASS_score', 'PEEP', 'FiO2',
            'PEEP_changes', 'Respiratory_rate_changes',
            'FiO2_changes'
        ]

        for c in to_quantize:
            data_mtx[c] = data_mtx[c].rank(pct=True)

        for c in list(set(data_columns) - set(to_quantize)):
            col = data_mtx[c]
            col = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col))
            data_mtx[c] = col


        # In[25]:


        data_mtx = clustering.reweight_related_features(
            data_mtx, 
            approach='square_root', 
            groups=cutoff_groups)


        # In[26]:


        norm_data_dist = clustering.get_distances(data_mtx, approach='nan_euclidean')
        norm_tree = clustering.get_tree(norm_data_dist, approach='ward')
        norm_out, norm_assignments = clustering.table_with_assignments(
            tree=norm_tree,
            labels=data.index
        )


        # In[27]:


        sign_mortality_norm = clustering.get_sign_mortalities(
            df_assigned_clusters=norm_out, 
            df_with_mortality=data)
        sign_mortality_norm.loc[:, 'approach'] = 'norm-euclidean'


        # In[28]:


        approaches['Normalized-Euclidean']  = {
                'feature_matrix':data_mtx,
                'data_dist':norm_data_dist,
                'tree':norm_tree,
                'assignments_table':norm_out,
                'assignments':norm_assignments
            }


        # # 3. Compare inter-cluster mortality differentiation across approaches

        # In[29]:


        mpl.rcParams["figure.figsize"] = (10, 6)
        sns.lineplot(
            x='max_cluster',
            y='significant',
            data=pd.concat(
                [
                    sign_mortality_rank,
                    sign_mortality_similarity,
                    sign_mortality_norm
                    
                ]).reset_index(drop=True),
            hue='approach'
        )
        dump_figure('approaches_significance.pdf')


    # ### Different approaches heatmaps

    # In[30]:


    key = 'Similarity'


    # In[31]:


    mpl.rcParams["figure.figsize"] = (10, 8)

    for key in approaches.keys():

        clustering.heatmap(
            df_assigned_clusters=approaches[key]['assignments_table'], 
            df_with_mortality=data, 
            data_mtx=approaches[key]['feature_matrix'], 
            tree=approaches[key]['tree'], 
            col_tree=col_tree)
        plt.title(f'{key}\n\n\n\n\n\n\n\n\n', fontsize=30, loc='center')

        dump_figure(
            f'heatmap-{key}.pdf')


    # ## 4. UMAP for best approach

    # In[32]:


    for key in approaches.keys():
        umap_data = clustering.show_umap(
        data=approaches[key]['data_dist'],
        assignment = approaches[key]['assignments'][14] )

        plt.title(f'{key}\n', fontsize=30, loc='center')
        
        #save umaps
        data["umap1"] = umap_data[:, 0]
        data["umap2"] = umap_data[:, 1]
        data = clustering.index_by_patient_ids_icu_stay_day(data)
        dump_table(data, f'{key}-umap.csv.gz')
        
        dump_figure(
            f'umap-{key}.pdf')


    # ## Manually order the heatmaps on 14 clusters

    # In[33]:


    data.drop(columns=['umap1', 'umap2'], inplace=True)
    data=data.reset_index()


    # In[34]:


    for key in approaches.keys():
        
        clustering.heatmap_by_mortality(
        data_with_mortality=data,
        data_mtx=approaches[key]['feature_matrix'],
        assignment=approaches[key]['assignments'][14]
            )
        
        plt.title(f'{key}\n', fontsize=30, loc='center')
            
        dump_figure(
            f'manual_order-{key}.pdf')

    # In[35]:



    #save clusters and cluster mortality

    for key in approaches.keys():  



        df_with_clusters, cluster_mortality = clustering.infer_clustermortality_and_add_cluster_id(
                data_with_mortality=data,
                data_mtx=approaches[key]['feature_matrix'],
                assignment=approaches[key]['assignments'][14]
                )
        dump_table(df_with_clusters, f'{key}-clusters.csv.gz')
        dump_table(cluster_mortality, f'{key}-cluster_mortality.csv.gz')


    # ## Quilts 

    # In[36]:

    for key in approaches.keys():
        
        clustering.quilt(
        data_with_mortality=data,
        data_mtx=approaches[key]['feature_matrix'],
        assignment=approaches[key]['assignments'][14]
            )

        plt.suptitle(f'{key}',fontsize=30, )
            
        dump_figure(
        f'quilt-{key}.pdf')


    # In[37]:



    for key in approaches.keys():
        
        clustering.quilt2(
        data_with_mortality=data,
        data_mtx=approaches[key]['feature_matrix'],
        assignment=approaches[key]['assignments'][14]
            )

        plt.suptitle(f'{key}',fontsize=30, )
            
        dump_figure(
        f'quilt2-{key}.pdf')
        

    # In[38]:




    for key in approaches.keys():
        
        clustering.quilt3(
        data_with_mortality=data,
        data_mtx=approaches[key]['feature_matrix'],
        assignment=approaches[key]['assignments'][14]
            )

        plt.suptitle(f'{key}',fontsize=30, )
            
        dump_figure(
        f'quilt3-{key}.pdf')
    
    # In[39]:



    for key in approaches.keys():
        
        clustering.quilt4(
        data_with_mortality=data,
        data_mtx=approaches[key]['feature_matrix'],
        assignment=approaches[key]['assignments'][14]
            )

        plt.suptitle(f'{key}',fontsize=30, )
            
        dump_figure(
        f'quilt4-{key}.pdf')

    return


def over_04x_spider_trimmed(user, outstem):

    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sys.path.append('./../src/')
    from manuscript import sankey_side_by_side as sankey
    from manuscript import clustering, datasets, inout, export

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")


    outfolder = f'{outstem}/04x_spider_trimmed'    # name of notebook
    save = True

    def dump_table(df, name):

        if save:
            export.full_frame(
                user, 
                f'{outfolder}/{name}', 
                df, 
                index=True,
                date=False
            )

    def dump_figure(name):
        if save:
            export.image(
                user,
                f'{outfolder}/{name}',
            )




    data = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/05_join_clusters_umap/05_data_umap_clusters.csv.gz'), 
        index_col=0)



    data_columns = clustering.get_reference_data_columns()
    data_mtx_orig = data[data_columns].copy()

    # cutoff_groups_on_orig = clustering.identify_related_features(data_mtx_orig)
    # data_mtx_as_pct = data_mtx_orig.rank(axis=0, pct=True)
    # data_dist_col = clustering.get_distances(
    #     data_mtx_as_pct.transpose(), approach='nan_euclidean')   
    # col_tree = clustering.get_tree(data_dist_col, approach='ward')
    # threshold_for_relatedness = 0.7
    # cutoff_groups = cutoff_groups_on_orig[threshold_for_relatedness]

    data_mtx = data_mtx_orig.copy()
    data_mtx = data_mtx.rank(axis=0, pct=True)

    # data_mtx_for_similarity = data_mtx.copy()
    # data_mtx_for_similarity = clustering.reweight_related_features(
    #     data_mtx_for_similarity, 
    #     approach='mean_rank', 
    #     groups=cutoff_groups)




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

    quilt_df = pd.DataFrame(0, index=[x[0] for x in FEATURE_GROUPS], columns=sorted(data.clusters.unique()))

    for group in FEATURE_GROUPS:
        for c in quilt_df.columns:
            idx = data.index[data.clusters.eq(c)]
            quilt_df.loc[group[0], c] = df.loc[idx, group[0]].sum() / df.loc[idx, f"{group[0]}_cnt"].sum()

    quiltdf2 = quilt_df.T

    def min_max_scaling(series):
        return (series - series.min()) / (series.max() - series.min())
    for col in quiltdf2.columns:
        quiltdf2[col] = min_max_scaling(quiltdf2[col])

    quiltdf3=quiltdf2.T

    quiltdf3 = quiltdf3.transpose().reset_index()



    quiltdf3['Relative Resp Severity'] = (quiltdf3['Respiratory']+quiltdf3['Ventilator instability'])/(quiltdf3['Respiratory']+quiltdf3['Ventilator instability']+quiltdf3['Neurologic']+quiltdf3['Shock']+quiltdf3['Renal']+quiltdf3['Inflammatory'])
    quiltdf3 = quiltdf3.astype(float)

    dump_table(quiltdf3, 'quiltdf3_normalized_then_relative_respiratory_score.csv')

    return






def over_05_join_clusters_umap(user, outstem):


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    sys.path.append('./../src/')
    from manuscript import sankey_side_by_side as sankey
    from manuscript import clustering, datasets, inout, export

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")

    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    # In[2]:


    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/05_join_clusters_umap'    # name of notebook
    save = True

    def dump_table(df, name):
        if save:
            export.full_frame(
                user, 
                f'{outfolder}/{name}', 
                df, 
                index=True,
                date=False
            )

    def dump_figure(name):
        if save:
            export.image(
                user,
                f'{outfolder}/{name}',
            )


    # In[3]:


    #load umap coordinates 

    umap = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/04_clustering/Similarity-umap.csv.gz'), 
        index_col=0)


    # In[4]:


    #load cluster information
    clusters = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/04_clustering/Similarity-clusters.csv.gz'), 
        index_col=0)


    # In[5]:


    clusters.sort_values(["Patient_id", "ICU_stay", "ICU_day"], inplace=True)


    # In[6]:


    #merge
    data = pd.merge(umap, clusters)


    # In[7]:


    data = data.set_index(
        data.Patient_id.astype(str) + "/" + data.ICU_stay.astype(str) + "/" + data.ICU_day.astype(str)
    )
    data.drop(columns=['index'], inplace=True)


    # In[8]:


    # we want the clusters numbered in increasing mortality
    # want the clusters 1-14 rather than 0-13

    data['clusters']=data['cluster_order']+1


    # In[9]:


    data.drop(columns=['cluster', 'cluster_order'], inplace=True)


    # In[10]:


    # Mortality per cluster


    # In[11]:


    cluster_mortality = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/04_clustering/Similarity-cluster_mortality.csv.gz'), 
        index_col=0)


    # In[12]:


    cluster_mortality['clusters']=cluster_mortality['index']+1


    # In[13]:


    cluster_mortality = cluster_mortality[['clusters','mortality']]


    # In[14]:


    dump_table(cluster_mortality, '05_cluster_mortality_renumbered.csv.gz')


    # In[15]:


    # Previous and next clusters


    # In[16]:


    data["prev_cluster"] = data.groupby(["Patient_id", "ICU_stay"]).shift(1).clusters
    data["next_cluster"] = data.groupby(["Patient_id", "ICU_stay"]).shift(-1).clusters
    data.prev_cluster = data.prev_cluster.fillna(-1).astype(int)
    data.next_cluster = data.next_cluster.fillna(-1).astype(int)
    data["is_transition"] = (data.clusters != data.next_cluster) & (data.next_cluster != -1)


    # In[17]:


    data.drop(columns=['level_0'], inplace=True)


    # In[18]:


    dump_table(data, '05_data_umap_clusters.csv.gz')


    # # Plots, feature plots

    # In[19]:


    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))
    scatter = ax.scatter(
        data.umap1, 
        data.umap2, 
        s=10, 
        c=[mpl.cm.tab20(i - 1) for i in data.clusters],
        alpha=0.5,
        ec=None
    )
    for c in data.clusters.unique():
        num = data.clusters[data.clusters == c].values[0]
        u1 = data.umap1[data.clusters == c].median()
        u2 = data.umap2[data.clusters == c].median()
        ax.text(u1, u2, c, size=18, weight="bold", path_effects=[
            mpl.patheffects.Stroke(linewidth=3, foreground="white"),
            mpl.patheffects.Normal()
        ])
    ax.set_xlabel("UMAP1", size=16)
    ax.set_ylabel("UMAP2", size=16)
    ax.collections[0].set_rasterized(True)
    ax.spines[["top", "right"]].set_visible(False)

    dump_figure('Similarity_umap_renumbered.pdf')


    # In[20]:


    clustering.feature_plot(data, "PaO2FIO2_ratio", "P:F ratio")
    dump_figure('pfratio.pdf')


    # In[21]:


    clustering.feature_plot(data, "Hemodialysis_flag", 'HD', cmap=mpl.colors.ListedColormap(["#ccc", "tab:orange"]))
    dump_figure('hemodialysis.pdf')


    # In[22]:


    clustering.feature_plot(data, "CRRT_flag", 'CRRT', cmap=mpl.colors.ListedColormap(["#ccc", "tab:red"]))
    dump_figure('crrt.pdf')


    # In[23]:


    clustering.feature_plot(data, "Intubation_flag", 'Intubated', cmap=mpl.colors.ListedColormap(["#ccc", "tab:green"]))
    dump_figure('intubated.pdf')


    # In[24]:


    clustering.feature_plot(data, "Creatinine", "Creatinine")
    dump_figure('creatinine.pdf')


    # In[25]:


    clustering.feature_plot(data, "D_dimer", "D-Dimer")
    dump_figure('ddimer.pdf')


    # In[26]:


    clustering.feature_plot(data, "ECMO_flag", "ECMO", cmap=mpl.colors.ListedColormap(["#ccc", "tab:blue"]))
    dump_figure('ecmo.pdf')


    # In[27]:


    clustering.feature_plot(data, "ECMO_flag", "ECMO", cmap=mpl.colors.ListedColormap(["#ccc", "tab:blue"]))
    dump_figure('ecmo.pdf')


    # In[28]:


    clustering.feature_plot(data, "COVID_status", "COVID", cmap=mpl.colors.ListedColormap(["#ccc", "lightcoral"]))
    dump_figure('covid.pdf')


    # In[29]:


    # set categorical order
    data['Patient_category'] = pd.Categorical(data['Patient_category'],
                                       categories=['Non-Pneumonia Control', 'Other Pneumonia', 'Other Viral Pneumonia', 'COVID-19'],
                                       ordered=True)

    Category_palette = [
        "tab:gray",
        "tab:blue",
        "forestgreen",
        "crimson",
    ]


    # In[30]:


    data2 = data.copy()


    # In[31]:



    color_map = {
         'Non-Pneumonia Control':'black',
         'Other Pneumonia':'lightgreen',
         'Other Viral Pneumonia':'lightblue',
         'COVID-19':'pink'
        }



    # In[32]:


    data2['color'] = data['Patient_category'].map(color_map)


    # In[33]:


    clustering.feature_plot(data2, "color", "Category")
    dump_figure('category.pdf')


    # # Pathway plots

    # In[34]:



    try:
        ax = clustering.plot_clinical_course(data, 271, line_color=mpl.cm.tab10.colors[0])
        dump_figure('271_course.pdf')
    except:
        print('271_course.pdf failed')

    # In[35]:


    try:
        ax = clustering.plot_clinical_course_umap(data, 271, line_color=mpl.cm.tab10.colors[0], 
                              first_day_offset=(-7, 7), last_day_offset=(-25, 2))
        dump_figure('271_umap.pdf')
    except:
        print('271_umap.pdf failed')







def over_06_covid_analysis(user, outstem):


    # import IPython.display
    # IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")


    # In[3]:


    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/06_covid-analysis'    # name of notebook
    save = True


    # In[4]:


    def dump_figure(name):
        if save:
            export.image(
                user,
                f'{outfolder}/{name}.pdf',
            )
        plt.close('all')


    def dump_table(df, name):
        if save:
            export.full_frame(
                user, 
                f'{outfolder}/{name}', 
                df, 
                index=False,
                date=False
            )


    # In[5]:


    data = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/05_join_clusters_umap/05_data_umap_clusters.csv.gz'), 
        index_col=0)


    # In[6]:


    # Simplify discharge
    data.Binary_outcome.replace({1: "Died", 0: 'Alive'}, inplace=True)

    # Set discharge disposition order
    data.Discharge_disposition = data.Discharge_disposition.astype("category")
    data.Discharge_disposition = data.Discharge_disposition.cat.reorder_categories([
        'Home', 
        'Rehab', 
        'SNF', 
        'LTACH',
        'Hospice', 
        'Died'
    ])

    data = data.rename(columns={
        'Patient_id': 'patient',
        'ICU_stay': 'stay',
        'ICU_day': 'day',
        'clusters': 'cluster'
    })

    # Ensure order
    data.sort_values(["patient", "stay", "day"], inplace=True)


    # In[7]:


    DISCHARGE_STACKS = [
        ('Home', 'Rehab', 'SNF', 'LTACH'),
        ('Hospice', 'Died')
    ]


    # In[8]:


    DISCH_PALETTE = [
        "tab:blue", #home
        "lightseagreen", #rehab
        "beige", #snf
        "gold",#ltach
        "orange",#hospice
        "crimson",#died 
    ]


    # # 1. Cohort description

    # In[9]:


    print(f"Total number of patients: {data.patient.nunique()}")
    print(f"Total number of ICU-days: {data.shape[0]}")


    # In[10]:


    data.groupby("COVID_status").agg({"patient": "nunique"})


    # In[11]:


    data.groupby(["Gender", "COVID_status"]).agg({"patient": "nunique"})


    # In[12]:


    data.groupby(["COVID_status", "Discharge_disposition"]).agg({"patient": "nunique"})


    # In[13]:


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


    # In[14]:


    df = data.groupby(["COVID_status", "Discharge_disposition"]).agg({"patient": "nunique"}).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    stacked_hue_barplot(
        df=df, 
        x="COVID_status", 
        y="patient", 
        hue="Discharge_disposition", 
        stacks=DISCHARGE_STACKS,
        ax=ax, 
        palette=mpl.colors.ListedColormap(DISCH_PALETTE),
    )
    ax.set_ylabel("Number of patients", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend_._loc = 2
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Discharge disposition by COVID", size=16);


    # In[15]:


    df.loc[df.COVID_status, "patient"] *= 100 / df.patient[df.COVID_status].sum()
    df.loc[~df.COVID_status, "patient"] *= 100 / df.patient[~df.COVID_status].sum()


    # In[16]:


    fig, ax = plt.subplots(figsize=(8, 4))
    df.pivot_table(values="patient", index=["Discharge_disposition"], columns="COVID_status").T.plot.barh(
        stacked=True,
        ax=ax,
        cmap=mpl.colors.ListedColormap(DISCH_PALETTE),
        ec="black"
    )
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.legend_.set_frame_on(False)
    ax.set_ylabel("")
    ax.set_yticklabels(["non-COVID", "COVID"], size=16)
    ax.set_xlabel("Percentage of patients", size=16)
    ax.set_title("COVID vs non-COVID split by discharge", size=16);


    # In[17]:


    df = data.groupby("patient").head(1)


    # In[18]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.Discharge_disposition.unique(), 2):
            days1 = df.Cumulative_ICU_days[(df.COVID_status == is_covid) & (df.Discharge_disposition == d1)]
            days2 = df.Cumulative_ICU_days[(df.COVID_status == is_covid) & (df.Discharge_disposition == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.Discharge_disposition.unique():
        days1 = df.Cumulative_ICU_days[~df.COVID_status & (df.Discharge_disposition == d)]
        days2 = df.Cumulative_ICU_days[df.COVID_status & (df.Discharge_disposition == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[19]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[20]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[21]:


    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="Cumulative_ICU_days", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1,
        showfliers=False
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Total days in ICU", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(
        loc="upper right", 
        title="Discharge disposition", 
        frameon=False, 
        fontsize=14,
        title_fontsize=12,
    )
    # ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Distribution of number of days in ICU per patient", size=16);
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="Cumulative_ICU_days", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()
    dump_figure('icu-days')


    # In[22]:


    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="Cumulative_ICU_days", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1,
        showfliers=False
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Total days in ICU", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(
        loc="upper right", 
        title="Discharge disposition", 
        frameon=False, 
        fontsize=14,
        title_fontsize=12,
    )
    # ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Distribution of number of days in ICU per patient", size=16);
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="Cumulative_ICU_days", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"$q=${x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()
    dump_figure('icu-days-no-outliers')


    # fig.savefig("13plots/13-01icu-days-no-outliers.pdf")


    # In[23]:


    fig, ax = plt.subplots()
    sns.violinplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="Cumulative_ICU_days", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1,
        cut=0,
        bw=0.1,
        scale="width"
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Total days in ICU", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Discharge disposition", frameon=False)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Distribution of number of days in ICU per patient", size=16);


    # # 2. Simple cluster and transition metrics

    # ## 2.1 Number of visited unique clusters per patient

    # In[24]:


    df = data.groupby(["COVID_status", "Discharge_disposition", "patient"]).agg(
        {"cluster": "nunique"}
    ).reset_index()


    # In[25]:


    df = df.loc[df.cluster > 0, :]


    # In[26]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.Discharge_disposition.unique(), 2):
            days1 = df.cluster[(df.COVID_status == is_covid) & (df.Discharge_disposition == d1)]
            days2 = df.cluster[(df.COVID_status == is_covid) & (df.Discharge_disposition == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.Discharge_disposition.unique():
        days1 = df.cluster[~df.COVID_status & (df.Discharge_disposition == d)]
        days2 = df.cluster[df.COVID_status & (df.Discharge_disposition == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[27]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[28]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[29]:


    fig, ax = plt.subplots()
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="cluster", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Number of unique clusters", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Discharge disposition", frameon=False)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Distribution of number of visited clusters per patient", size=16)
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="cluster", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();


    # ## 2.2 First cluster per patient

    # In[30]:


    def stacked_hatched_barplot(df, x, y, hue, hatch, ax, palette=None):
        if palette is None:
            palette = mpl.cm.tab10
        bar_gap = 1
        bar_width = 2
        col_pad = 3
        n_bars = df[hue].nunique()
        n_cols = df[x].nunique()
        n_hatch = df[hatch].nunique()
        hatch_cycle = ["", "////"]
        col_width = n_bars * bar_width + (n_bars - 1) * bar_gap + 2 * col_pad
        col_values = pd.Series(df[x].unique()).sort_values().reset_index(drop=True)
        bar_values = pd.Series(df[hue].unique()).sort_values().reset_index(drop=True)
        hatch_values = pd.Series(df[hatch].unique()).sort_values().reset_index(drop=True)
        to_display = pd.DataFrame(dict(
            col=np.repeat(col_values, n_bars), 
            bar=np.tile(bar_values, n_cols),
            bar_num=np.tile(bar_values.index, n_cols)
        )).reset_index().rename({"index": "col_num"}, axis=1)
        to_display["bar_pos"] = to_display.apply(
            lambda x: x.col_num * col_width + x.bar_num * bar_width + (x.bar_num - 1) * bar_gap + col_pad, 
            axis=1
        )
        bottom = np.zeros(to_display.shape[0])
        for i, h in enumerate(hatch_values):
            count = to_display.merge(
                df.loc[df[hatch] == h, :], 
                left_on=["col", "bar"], 
                right_on=[x, hue],
                how="left"
            )[y].fillna(0)
            this_h = hatch_cycle[h % len(hatch_cycle)]
            ax.bar(
                to_display.bar_pos, 
                count, 
                color=[palette(i) for i in to_display.bar_num],
                ec="#333333",
                width=bar_width,
                align="edge",
                hatch=this_h,
                bottom=bottom
            )
            bottom += count
        ax.set_xticks(col_values.index * col_width + col_width / 2)
        ax.set_xticklabels(col_values, size=16)

        bar_handles = []
        for i, b in enumerate(bar_values):
            bar_handles.append(mpl.patches.Patch(color=palette(i), label=b, ec="#333333"))
        bar_handles.append(mpl.patches.Patch(color="w", label="non-COVID", ec="#333333"))
        bar_handles.append(mpl.patches.Patch(color="w", label="COVID", ec="#333333", hatch="////"))
        ax.legend(handles=bar_handles, loc="upper right", title="Discharge disposition", frameon=False)
        return ax


    # In[31]:


    first_days = data.groupby("patient").head(1).index


    # In[32]:


    df = data.loc[first_days, :].groupby(["Discharge_disposition", "COVID_status"]).agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_2": "cluster"}, axis=1)


    # In[33]:


    fig, ax = plt.subplots(figsize=(15, 4))
    stacked_hatched_barplot(
        df, 
        x="cluster", 
        y="count", 
        hue="Discharge_disposition", 
        hatch="COVID_status", 
        ax=ax,
        palette=mpl.colors.ListedColormap(DISCH_PALETTE)
    )
    ax.legend_._loc = 1
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title("Distribution of first clusters for patients (absolute)", size=16);


    # In[34]:


    df["count_frac"] = df["count"] / df.groupby("Discharge_disposition").agg({
        "count": "sum"
    }).loc[df.Discharge_disposition, "count"].values


    # In[35]:


    fig, ax = plt.subplots(figsize=(16, 4))
    stacked_hatched_barplot(
        df, 
        x="cluster", 
        y="count_frac", 
        hue="Discharge_disposition", 
        hatch="COVID_status", 
        ax=ax,
        palette=mpl.colors.ListedColormap(DISCH_PALETTE)
    )
    ax.legend_._loc = 1
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Fraction of patients\nin discharge group", size=16)
    ax.set_title("Distribution of first clusters for patients (fractions)", size=16);


    # In[36]:


    df = data.loc[first_days, :].groupby("COVID_status").agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_2": "cluster"}, axis=1)


    # In[37]:


    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(
        data=df, 
        x="cluster", 
        y="count", 
        hue="COVID_status", 
        ax=ax,
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]]
    )
    ax.legend_._loc = 1
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title("Distribution of first clusters for patients (absolute)", size=16);


    # ## 2.3 First cluster for short-stay patients

    # In[38]:


    df = data.loc[
        data.index.isin(first_days) & (data.Cumulative_ICU_days < 6), 
        :
    ].groupby(["Discharge_disposition", "COVID_status"]).agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_2": "cluster"}, axis=1)


    # In[39]:


    fig, ax = plt.subplots(figsize=(16, 4))
    stacked_hatched_barplot(
        df, 
        x="cluster", 
        y="count", 
        hue="Discharge_disposition", 
        hatch="COVID_status", 
        ax=ax,
        palette=mpl.colors.ListedColormap(DISCH_PALETTE)
    )
    ax.legend_._loc = 1
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title("Distribution of first clusters for patients with <= 5 total ICU days (absolute)", size=16);


    # ## 2.4 Last cluster per patient

    # In[40]:


    last_days = data.groupby("patient").tail(1).index


    # In[41]:


    df = data.loc[last_days, :].groupby(["Discharge_disposition", "COVID_status"]).agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_2": "cluster"}, axis=1)


    # In[42]:


    fig, ax = plt.subplots(figsize=(16, 4))
    stacked_hatched_barplot(
        df, 
        x="cluster", 
        y="count", 
        hue="Discharge_disposition", 
        hatch="COVID_status", 
        ax=ax,
        palette=mpl.colors.ListedColormap(DISCH_PALETTE)
    )
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title("Distribution of last clusters for patients (absolute)", size=16);


    # In[43]:


    df["count_frac"] = df["count"] / df.groupby("Discharge_disposition").agg({
        "count": "sum"
    }).loc[df.Discharge_disposition, "count"].values


    # In[44]:


    fig, ax = plt.subplots(figsize=(16, 4))
    stacked_hatched_barplot(
        df, 
        x="cluster", 
        y="count_frac", 
        hue="Discharge_disposition", 
        hatch="COVID_status", 
        ax=ax,
        palette=mpl.colors.ListedColormap(DISCH_PALETTE)
    )
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Fraction of patients\nin discharge group", size=16)
    ax.set_title("Distribution of last clusters for patients (fractions)", size=16);


    # In[45]:


    df = data.loc[last_days, :].groupby("COVID_status").agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_2": "cluster"}, axis=1)


    # In[46]:


    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(
        data=df, 
        x="cluster", 
        y="count", 
        hue="COVID_status", 
        ax=ax,
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]]
    )
    ax.legend_._loc = 1
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title("Distribution of last clusters for patients (absolute)", size=16);


    # ## 2.5 Number of transitions per patient

    # In[47]:


    df = data.groupby(["patient", "Discharge_disposition", "COVID_status", "Cumulative_ICU_days"]).apply(
        lambda x: x.is_transition.sum()
    ).reset_index()
    df.rename({0: "is_transition"}, axis=1, inplace=True)


    # In[48]:


    df.is_transition.value_counts()


    # In[49]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.Discharge_disposition.unique(), 2):
            days1 = df.is_transition[(df.COVID_status == is_covid) & (df.Discharge_disposition == d1)]
            days2 = df.is_transition[(df.COVID_status == is_covid) & (df.Discharge_disposition == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.Discharge_disposition.unique():
        days1 = df.is_transition[~df.COVID_status & (df.Discharge_disposition == d)]
        days2 = df.is_transition[df.COVID_status & (df.Discharge_disposition == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[50]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[51]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[52]:


    fig, ax = plt.subplots()
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="is_transition", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Number of transitions", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Discharge disposition", frameon=False)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Distribution of number of transitions per patient", size=16)
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="is_transition", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();


    # In[53]:


    df["outcome"] = "Lived"
    df.loc[df.Discharge_disposition.isin(["Hospice", "Died"]), "outcome"] = "Died"


    # In[54]:


    df.is_transition.isna().sum()


    # In[55]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.outcome.unique(), 2):
            days1 = df.is_transition[(df.COVID_status == is_covid) & (df.outcome == d1)]
            days2 = df.is_transition[(df.COVID_status == is_covid) & (df.outcome == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.outcome.unique():
        days1 = df.is_transition[~df.COVID_status & (df.outcome == d)]
        days2 = df.is_transition[df.COVID_status & (df.outcome == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[56]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[57]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[58]:

    try: 
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        sns.boxplot(
            data=df, 
            x="COVID_status",
            hue="outcome", 
            y="is_transition", 
            ax=ax, 
            saturation=1, 
            palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
            linewidth=1,
            showfliers=True
        )
        for a in ax.collections:
            if isinstance(a, mpl.collections.PatchCollection):
                # remove line surround each box
                a.set_linewidth(0)
        ax.set_ylabel("Number of transitions", size=16)
        ax.set_xlabel("")
        ax.set_xticklabels(["non-COVID", "COVID"], size=16)
        ax.legend(loc="upper left", title="Outcome", frameon=False, fontsize=14, title_fontsize=12)
        ax.legend_.set_bbox_to_anchor((1, 0.8))
        ax.set_title("Number of transitions per patient", size=16)
        annotator = statannotations.Annotator.Annotator(
            ax, 
            pairs, 
            data=df, 
            x="COVID_status",
            hue="outcome", 
            y="is_transition", 
            verbose=False
        )
        annotator._verbose = False
        annotator.configure(line_width=1)
        annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
        annotator.annotate()

        dump_figure('transitions-binary')       #####  <------- of interest

    except:
        print('could not create fiture for transitions-binary')

    # fig.savefig("13plots/13-02n-transitions-binary.pdf")


    # ## 2.6 Number of normalized transitions

    # Let's normalize number of transitions by total number of ICU days per patient

    # In[59]:


    df.is_transition /= df.Cumulative_ICU_days


    # In[60]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.Discharge_disposition.unique(), 2):
            days1 = df.is_transition[(df.COVID_status == is_covid) & (df.Discharge_disposition == d1)]
            days2 = df.is_transition[(df.COVID_status == is_covid) & (df.Discharge_disposition == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.Discharge_disposition.unique():
        days1 = df.is_transition[~df.COVID_status & (df.Discharge_disposition == d)]
        days2 = df.is_transition[df.COVID_status & (df.Discharge_disposition == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[61]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[62]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[63]:


    # fig, ax = plt.subplots()
    # sns.boxplot(
    #     data=df, 
    #     x="COVID_status",
    #     hue="Discharge_disposition", 
    #     y="is_transition", 
    #     ax=ax, 
    #     saturation=1, 
    #     palette=DISCH_PALETTE,
    #     linewidth=1,
    #     showfliers=False
    # )
    # for a in ax.collections:
    #     if isinstance(a, mpl.collections.PatchCollection):
    #         # remove line surround each box
    #         a.set_linewidth(0)
    # ax.set_ylabel("Number of transitions / number of ICU-days", size=16)
    # ax.set_xlabel("")
    # ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    # ax.legend(loc="upper left", title="Discharge disposition", frameon=False)
    # ax.legend_.set_bbox_to_anchor((1, 0.8))
    # ax.set_title("Distribution of number of normalized transitions per patient", size=16)
    # annotator = statannotations.Annotator.Annotator(
    #     ax, 
    #     pairs, 
    #     data=df, 
    #     x="COVID_status",
    #     hue="Discharge_disposition", 
    #     y="is_transition", 
    #     verbose=False
    # )
    # annotator._verbose = False
    # annotator.configure(line_width=1)
    # annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    # annotator.annotate();


    # In[64]:

    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.outcome.unique(), 2):
            days1 = df.is_transition[(df.COVID_status == is_covid) & (df.outcome == d1)]
            days2 = df.is_transition[(df.COVID_status == is_covid) & (df.outcome == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval, days1.median(), days2.median()])
    for d in df.outcome.unique():
        days1 = df.is_transition[~df.COVID_status & (df.outcome == d)]
        days2 = df.is_transition[df.COVID_status & (df.outcome == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval, days1.median(), days2.median()])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval", "group1_median", "group2_median"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]

    dump_table(stat_results, 'norm-transitions-binary_stats.xlsx')



    # In[65]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[66]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[67]:


    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="outcome", 
        y="is_transition", 
        ax=ax, 
        saturation=1, 
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
        linewidth=1,
        showfliers=True
    )
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
    ax.set_ylabel("Number of transitions per ICU-day", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Outcome", frameon=False, fontsize=14, title_fontsize=12)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.set_title("Normalized number of transitions per patient", size=16)
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="outcome", 
        y="is_transition", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()
    dump_figure('norm-transitions-binary')        #### <<<<<<<<<<<<<<< Of interest


    # ## 2.7 Favorable and unfavorable transitions

    # In[68]:


    print(f"Total number of transitions: {(data.is_transition & (data.next_cluster != -1)).sum()}")


    # Let's compute cluster mortality by taking clusters for last days of the patients and computing number of expired patients divided by total number of patients in cluster.
    # 
    # Favourable transition is one where next cluster has less mortality.

    # In[69]:


    df = data.loc[data.is_transition, :].copy()


    # In[70]:


    cluster_mortality = data.copy()
    cluster_mortality = cluster_mortality.groupby("cluster").apply(
        lambda x: pd.Series([x.patient.nunique(), x.patient[x.Binary_outcome.eq("Died")].nunique()])
    )
    cluster_mortality.rename({0: "Total", 1: "Died"}, axis=1, inplace=True)
    cluster_mortality["mortality"] = cluster_mortality.Died / cluster_mortality.Total
    cluster_mortality = cluster_mortality.sort_values("mortality").reset_index()


    # In[71]:


    cluster_mortality.set_index("cluster", inplace=True)


    # In[72]:


    df["cluster_mortality"] = cluster_mortality.mortality[df.cluster].values


    # In[73]:


    df["next_cluster_mortality"] = cluster_mortality.mortality[df.next_cluster].values


    # In[74]:


    df["favorable_transition"] = df.next_cluster_mortality < df.cluster_mortality


    # Positive is bad: increasing mortality

    # In[75]:


    df["mortality_change"] = df.next_cluster_mortality - df.cluster_mortality


    # In[76]:


    df.groupby(["COVID_status", "favorable_transition"]).count()[["day"]]


    # In[77]:


    cluster_mortality.sort_values("mortality")


    # In[78]:


    data.loc[df.index, "favorable_transition"] = df.favorable_transition
    data.loc[df.index, "mortality_change"] = df.mortality_change


    # In[79]:


    # data.to_csv("../2021-12-09-v4-data/20data-v1.csv.gz")


    # In[80]:


    df = data.groupby(["patient", "Discharge_disposition", "COVID_status"]).apply(
        lambda x: pd.Series(
            [
                x.favorable_transition.dropna().astype(bool).sum(), 
                (~x.favorable_transition.dropna().astype(bool)).sum()
            ],
            index=[True, False]
        )
    ).reset_index().dropna().melt(
        id_vars=["patient", "Discharge_disposition", "COVID_status"],
        var_name="favorable_transition",
        value_name="n_transitions"
    )


    # In[81]:


    fg = sns.catplot(
        kind="box",
        col="COVID_status",
        data=df, 
        x="Discharge_disposition", 
        hue="favorable_transition",
        y="n_transitions", 
        saturation=1, 
        palette=[DISCH_PALETTE[-1], DISCH_PALETTE[0]],
        linewidth=1,
        sharey=False
    )
    fg.axes[0, 0].set_ylabel("Number of transitions", size=16)
    for ax in fg.axes.ravel():
        ax.set_xlabel("")
        ax.title.set_size(16)
        trans = mpl.transforms.Affine2D().translate(6, 0)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_horizontalalignment("right")
            t.set_transform(t.get_transform() + trans)
            t.set_size(14)


    # In[82]:


    df = data.groupby(["patient", "Discharge_disposition", "COVID_status"]).apply(
        lambda x: x.mortality_change.sum()
    ).reset_index()
    df.rename({0: "mortality_change"}, axis=1, inplace=True)


    # In[83]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.Discharge_disposition.unique(), 2):
            days1 = df.mortality_change[(df.COVID_status == is_covid) & (df.Discharge_disposition == d1)]
            days2 = df.mortality_change[(df.COVID_status == is_covid) & (df.Discharge_disposition == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.Discharge_disposition.unique():
        days1 = df.mortality_change[~df.COVID_status & (df.Discharge_disposition == d)]
        days2 = df.mortality_change[df.COVID_status & (df.Discharge_disposition == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[84]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[85]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[86]:


    fig, ax = plt.subplots()
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="mortality_change", 
        ax=ax, 
        saturation=1, 
        palette=DISCH_PALETTE,
        linewidth=1,
    )
    ax.set_ylabel("Sum of cluster mortality\nchange per patient", size=16)
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Discharge disposition", frameon=False)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.axhline(c="#999", ls="--", lw=1, zorder=-1)
    # ax.set_title("Distribution of number of transitions per patient", size=16)
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="Discharge_disposition", 
        y="mortality_change", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();


    # In[87]:


    df["outcome"] = "Positive"
    df.loc[df.Discharge_disposition.isin(["Hospice", "Died"]), "outcome"] = "Negative"


    # In[88]:


    stat_results = []
    for is_covid in [True, False]:
        for d1, d2 in itertools.combinations(df.outcome.unique(), 2):
            days1 = df.mortality_change[(df.COVID_status == is_covid) & (df.outcome == d1)]
            days2 = df.mortality_change[(df.COVID_status == is_covid) & (df.outcome == d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stat_results.append(["COVID_status", is_covid, d1, d2, days1.size, days2.size, pval])
    for d in df.outcome.unique():
        days1 = df.mortality_change[~df.COVID_status & (df.outcome == d)]
        days2 = df.mortality_change[df.COVID_status & (df.outcome == d)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["discharge", d, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[89]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[90]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        if r.variable == "COVID_status":
            pairs.append(((r.value, r.group1), (r.value, r.group2)))
        else:
            pairs.append(((r.group1, r.value), (r.group2, r.value)))


    # In[91]:


    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="COVID_status",
        hue="outcome", 
        y="mortality_change", 
        ax=ax, 
        saturation=1, 
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
        linewidth=1,
    )
    ax.set_ylabel("Sum of cluster mortality\nchange per patient", size=16)
    ax.set_xlabel("")
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)
    ax.legend(loc="upper left", title="Outcomes", frameon=False, fontsize=16, title_fontsize=14)
    ax.legend_.set_bbox_to_anchor((1, 0.8))
    ax.axhline(c="#999", ls="--", lw=1, zorder=-1)
    for t in ax.get_xticklabels():
        t.set_size(16)
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="COVID_status",
        hue="outcome", 
        y="mortality_change", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"$q=${x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()

    dump_figure('sum-of-transitions')
    # fig.savefig("13plots/13-04sum-of-transitions.pdf")


    # ## 2.8 Number of days spent in cluster

    # In[92]:


    pt_lengths = data.groupby(["patient"]).agg({"day": "count"})


    # In[93]:


    df = data.groupby(["patient", "Discharge_disposition", "COVID_status", "cluster"]).apply(
        lambda x: x.day.shape[0]
    ).reset_index()
    df.rename({0: "n_days"}, axis=1, inplace=True)


    # In[94]:


    fig, ax = plt.subplots(figsize=(16, 4))
    sns.boxplot(
        data=df, 
        x="cluster", 
        y="n_days", 
        ax=ax,
        palette="tab20",
        saturation=1,
        linewidth=1
    )
    ax.set_xlabel("Cluster", size=16)
    ax.set_ylabel("Number of ICU-days", size=16)
    ax.set_title("Distribution of number of ICU-days spent in clusters per patient", size=16);


    # In[95]:


    stat_results = []
    for c in df.cluster.unique():
        days1 = df.n_days[~df.COVID_status & (df.cluster == c)]
        days2 = df.n_days[df.COVID_status & (df.cluster == c)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["cluster", c, False, True, days1.size, days2.size, pval, days1.median(), days2.median()])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval", "group1_median", "group2_median"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]



    dump_table(stat_results, 'n-days-per-cluster_stats.xlsx')


    # In[96]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]
    stat_results_sign


    # In[97]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        pairs.append(((r.value, r.group1), (r.value, r.group2)))


    # In[98]:


    fig, ax = plt.subplots(figsize=(18, 4), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="cluster", 
        y="n_days", 
        hue="COVID_status", 
        ax=ax,
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
        saturation=1,
        linewidth=1
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        labels, 
        title="COVID status", 
        fontsize=14, 
        title_fontsize=14, 
        frameon=False,
        loc="upper left"
    )
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Clinical state", size=16)
    ax.set_ylabel("Number of ICU-days", size=16)
    ax.set_title("Distribution of number of ICU-days spent in clinical states per patient", size=16);
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="cluster", 
        y="n_days", 
        hue="COVID_status",
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()

    dump_figure('n-days-per-cluster')



    # fig.savefig("13plots/13-05n-days-per-cluster.pdf")


    # In[99]:


    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
    sns.boxplot(
        data=df, 
        x="cluster", 
        y="n_days", 
        hue="COVID_status", 
        ax=ax,
        palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
        saturation=1,
        linewidth=1,
        showfliers=False
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        labels, 
        title="COVID status", 
        fontsize=14, 
        title_fontsize=14, 
        frameon=False,
        loc="upper left"
    )
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Clinical state", size=16)
    ax.set_ylabel("Number of ICU-days", size=16)
    ax.set_title("Distribution of number of ICU-days spent in clinical states per patient", size=16);
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=df, 
        x="cluster", 
        y="n_days", 
        hue="COVID_status",
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
    annotator.annotate()

    dump_figure('n-days-per-cluster-no-outliers')


    # fig.savefig("13plots/13-05n-days-per-cluster-no-outliers.pdf")


    # ## 2.9 Relative number of days spent in cluster

    # In[100]:


    df.n_days /= pt_lengths.day[df.patient].values


    # In[101]:


    # fig, ax = plt.subplots(figsize=(16, 4))          ##### Deactivated to save memory in plotting
    # sns.violinplot(
    #     data=df, 
    #     x="cluster",
    #     hue="COVID_status", 
    #     y="n_days", 
    #     ax=ax,
    #     palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
    #     saturation=1,
    #     linewidth=1,
    #     cut=0,
    #     bw=0.05,
    #     scale="width"
    # )
    # ax.legend_._loc = 9
    # ax.legend_.set_bbox_to_anchor((0.685, 1))
    # ax.legend_.set_frame_on(False)
    # ax.set_xlabel("Cluster", size=16)
    # ax.set_ylabel("Fraction of ICU-days", size=16)
    # ax.set_title(
    #     "Distribution of fraction of ICU-days spent in clinical states per patient (out of total number of ICU days)", 
    #     size=16
    # );


    # ## 2.10 Relative timepoint per cluster

    # At which timepoint during ICU stay was a patient in a cluster? Closer to the beginning of the stay or to the end?

    # In[102]:


    icu_rank_lengths = data.groupby(["patient", "stay"]).agg({"day": "max"})


    # In[103]:


    df = data.copy()


    # In[104]:


    df.day /= icu_rank_lengths.day[pd.MultiIndex.from_arrays([data.patient, data.stay])].values


    # In[105]:


    stat_results = []
    for c in df.cluster.unique():
        days1 = df.day[~df.COVID_status & (df.cluster == c)]
        days2 = df.day[df.COVID_status & (df.cluster == c)]
        if days1.size == 0 or days2.size == 0:
            continue
        pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
        stat_results.append(["cluster", c, False, True, days1.size, days2.size, pval])
    stat_results = pd.DataFrame(stat_results, columns=["variable", "value", "group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]


    # In[106]:


    stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]


    # In[107]:


    pairs = []
    for _, r in stat_results_sign.iterrows():
        pairs.append(((r.value, r.group1), (r.value, r.group2)))


    # In[108]:


    # fig, ax = plt.subplots(figsize=(15, 4))          ### Deactivated to plot memory during plotting
    # sns.violinplot(
    #     data=df, 
    #     x="cluster", 
    #     hue="COVID_status",
    #     y="day", 
    #     ax=ax,
    #     palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
    #     saturation=1,
    #     linewidth=1,
    #     cut=0,
    #     bw=0.05,
    #     scale="width"
    # )
    # ax.legend_._loc = 2
    # ax.legend_.set_bbox_to_anchor((1, 1))
    # ax.set_xlabel("Cluster", size=16)
    # ax.set_ylabel("ICU day / length of ICU stay", size=16)
    # ax.set_title("Distribution of relative ICU-days per clinical state", size=16);
    # annotator = statannotations.Annotator.Annotator(
    #     ax, 
    #     pairs, 
    #     data=df, 
    #     x="cluster", 
    #     y="day", 
    #     hue="COVID_status",
    #     verbose=False
    # )
    # annotator._verbose = False
    # annotator.configure(line_width=1)
    # annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    # annotator.annotate();


    # ## 2.11 Number of BALs per cluster

    # In[109]:


    fig, ax = plt.subplots(figsize=(16, 4))
    ax = sns.countplot(
        data=data.loc[data.has_bal, :],
        x="cluster",
        hue="Discharge_disposition",
        ax=ax,
        palette=DISCH_PALETTE,
        saturation=1,
        ec="#333333"
    )
    ax.legend_.set_frame_on(False)
    ax.legend_._loc = 2
    ax.set_ylabel("Number of ICU-days with BAL", size=16)
    ax.set_xlabel("Cluster", size=16)
    ax.set_title("Count ICU-days with BAL per clinical state per discharge", size=16);


    # # 3. Cluster transitions analysis

    # ## 3.1 Barplots per cluster

    # In[110]:


    for c in sorted(data.cluster.unique()):
        idx = (data.cluster == c) & (data.next_cluster != c)
        fig, ax = plt.subplots(figsize=(11, 4))
        df = data.loc[idx, :].groupby(["next_cluster", "Discharge_disposition", "COVID_status"]).agg(
            {"day": "count"}
        ).reset_index()
        df = df.loc[df.next_cluster != c, :]
        # stacked_hatched_barplot(      ### deactivated to save memory in plotting
        #     df=df,
        #     x="next_cluster",
        #     y="day",
        #     hue="Discharge_disposition",
        #     hatch="COVID_status",
        #     ax=ax,
        #     palette=mpl.colors.ListedColormap(DISCH_PALETTE),
        # )
        # ax.legend_.set_frame_on(False)
        # ax.legend_._loc = 6
        # ax.legend_.set_bbox_to_anchor((1, 0.5))
        # ax.set_title(f"Transitions from cluster {c}", size=16)
        # ax.set_ylabel("Number of transitions", size=16)
        # ax.set_xlabel("Next cluster", size=16)
        # plt.show()


    # ## 3.2 Summary with ball-of-yarn

    # In[111]:


    def get_axes_pixel_transform(ax):
        t = ax.transAxes.transform([(0, 0), (1, 1)])
        return mpl.transforms.Affine2D().scale(1 / (t[1, 0] - t[0, 0]), 1 / (t[1, 1] - t[0, 1])) + ax.transAxes


    # In[112]:


    def get_arrow_path(ax, tx, p0, i0, i1, p1, base_w=10):
        def get_dist(p0, p1):
            return ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        base_h = 3
        head_w = 5
        head_h = 5
        points = np.array([p0, i0, i1, p1], dtype=float)
        (p0, i0, i1, p1) = tx.inverted().transform(ax.transData.transform(points))
        base_off = base_w / get_dist(p0, i0)
        base_hoff = base_h / get_dist(p0, i0)
        base_r = (p0[0] + base_off * (i0[1] - p0[1]), p0[1] - base_off * (i0[0] - p0[0]))
        base_m = (p0[0] + base_hoff * (i0[0] - p0[0]), p0[1] + base_hoff * (i0[1] - p0[1]))
        base_l = (p0[0] - base_off * (i0[1] - p0[1]), p0[1] + base_off * (i0[0] - p0[0]))
        head_off = head_w / get_dist(i1, p1)
        head_hoff = head_h / get_dist(i1, p1)
        head_r = (p1[0] + head_off * (p1[1] - i1[1]), p1[1] - head_off * (p1[0] - i1[0]))
        head_l = (p1[0] - head_off * (p1[1] - i1[1]), p1[1] + head_off * (p1[0] - i1[0]))
        head_r = (head_r[0] - head_hoff * (p1[0] - i1[0]), head_r[1] - head_hoff * (p1[1] - i1[1]))
        head_l = (head_l[0] - head_hoff * (p1[0] - i1[0]), head_l[1] - head_hoff * (p1[1] - i1[1]))
        verts = [base_r, i0, i1, p1, head_r, p1, head_l, p1, i1, i0, base_l, base_m, base_r]
        codes = [
            mpl.path.Path.MOVETO, mpl.path.Path.CURVE4, mpl.path.Path.CURVE4, mpl.path.Path.LINETO, 
            mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, 
            mpl.path.Path.CURVE4, mpl.path.Path.CURVE4, mpl.path.Path.LINETO, mpl.path.Path.LINETO,
            mpl.path.Path.CLOSEPOLY
        ]
        return mpl.path.Path(verts, codes)


    # In[113]:


    def reorder_df(df, curr_cl, idx):
        cur_idx = idx.get_loc(curr_cl)
        idx = pd.Series(np.roll(idx, 5 - cur_idx))
        idx[:5] = list(reversed(idx[:5]))
        idx[6:] = list(reversed(idx[6:].values))
        idx[5] = -1
        present = idx[idx.isin(df.next_cluster)]
        first_half = idx[:5].isin(df.next_cluster).sum()
        df.set_index(df.next_cluster, inplace=True)
        return df.loc[present, :], first_half


    # In[114]:


    def ball_of_yarn(df, cluster_sizes, names=None):
        if names is None:
            names = {}
        cluster_order = cluster_sizes.index
        gap = 5 * np.pi / 180
        data = df.copy()
        df = df.loc[df.cluster != df.next_cluster, :].groupby(
            ["cluster", "next_cluster"], dropna=False
        ).agg({"day": "count"}).reset_index()
        df = df.loc[df.day > 0, :]
        resolution =  (2 * np.pi - gap * cluster_sizes.size) / cluster_sizes.sum()
        cluster_angles = cluster_sizes * resolution
        df = df.reset_index()
        cdict = {
            'red':   [
                (0.0, 180/256, 180/256),
                (0.2, 256/256, 256/256),
                (1.0, 108/256, 108/256)
            ],

            'green': [
                (0.0, 180/256, 180/256),
                (0.2, 176/256, 176/256),
                (1.0, 0/256, 0/256)
            ],

            'blue':  [
                (0.0, 180/256, 180/256),
                (0.2, 27/256, 27/256),
                (1.0, 0/256, 0/256)
            ]
        }
        cmap = mpl.colors.LinearSegmentedColormap('exprCmap', segmentdata=cdict, N=256)
        lw = 1.5
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(12, 12))
        tx = get_axes_pixel_transform(ax)
        ax.spines['polar'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 2.2)
        start = np.pi / 2 + gap
        for i, cluster_size in enumerate(cluster_angles.tolist()):
            this_name = cluster_order[i]
            this_cluster = df.cluster[df.cluster == this_name].values[0]
            this_color = mpl.cm.tab20(i)
            start -= gap
            ax.bar(start - cluster_size, [0.1], bottom=1.7, width=cluster_size, align="edge", color=this_color)
            x_line = np.linspace(start, start - cluster_size)
            if -np.pi < start < 0:
                x_line = np.linspace(start - cluster_size, start)
            text = CurvedText(
                x=x_line,
                y=np.linspace(1.75, 1.75),
                text=str(names.get(this_name, this_name)),
                va="center",
                axes=ax,
                ha="center",
                weight="bold"
            )
    #         ax.text(start - cluster_size / 2, 1.75, this_name, ha="center", va="center", weight="bold")
            ax.scatter([start - cluster_size / 2], [1.4], s=400, color=this_color, zorder=1000)
            ax.text(start - cluster_size / 2, 1.4, str(this_cluster), ha="center", va="center", weight="bold", zorder=1100)
            n_text = start - gap / 1.5
            text_al = "left"
            text_val = "bottom"
            if n_text < -np.pi / 2:
                n_text = start - cluster_size + gap / 1.5
                text_al = "right"
            if -np.pi < n_text < 0:
                text_val = "top"
            ax.text(
                n_text, 
                1.8, 
                f"$n$={df.day[df.cluster == this_name].sum()}", 
                ha=text_al, 
                va=text_val, 
                zorder=11
            )
            start -= cluster_size
        ax.bar([0], [0.12], bottom=2, width=2 * np.pi, color="#aaa", zorder=5)
    #     ax.text(np.pi / 2, 2.05, "Leave ICU", ha="center", va="center", zorder=10, size=12)
        max_out = 0
        for i, cluster_size in enumerate(cluster_angles.tolist()):
            cluster_num = cluster_angles.index[i]
            df_slice = df.loc[df.cluster == cluster_num]
            df_slice, first_len = reorder_df(df_slice, cluster_num, cluster_angles.index)
    #         print(df_slice)
    #         df_slice.reset_index(inplace=True)
            has_exit = df_slice.next_cluster.isin([-1]).sum()
    #         break
    #         first_len = df_slice.shape[0] // 2
            arrow_space = cluster_size / (~df_slice.next_cluster.isin([-1])).sum()
            for j, (_, r) in enumerate(df_slice.iterrows()):
                r0 = 1.6
                pct_txt = str(r.day)
                color = cmap(r.day / df.day.max())
                lw = 0.0 + 12 * (r.day / df.day.max()) ** 0.8
                if r.next_cluster == -1:
                    t0 = np.pi / 2 - cluster_angles.cumsum().iat[i] - gap * i
                    t0 += cluster_angles.iat[i] / 2
                    r0 = 1.8
                    t1, r1 = (t0, 2)
                    text_pos = t0 - np.pi / 250
                    text_al = "left"
                    text_val = "top"
                    if text_pos < -np.pi or (-np.pi / 2 < text_pos < 0):
                        text_pos += 2 * np.pi / 250
                    if text_pos < -np.pi / 2:
                        text_al = "right"
                    if -np.pi < text_pos < 0:
                        text_val = "bottom"
                    ax.text(text_pos, r0 + 0.08, pct_txt, ha=text_al, va=text_val, zorder=10)
                    path = get_arrow_path(ax, tx, (t0, r0), (t0, r0 + (r1 - r0) / 2), 
                                          (t0, r0 + (r1 - r0) / 2), (t1, r1), base_w=lw)
                    patch = mpl.patches.PathPatch(
                        path, 
                        fc=color, 
                        ec=color, 
                        lw=1,
                        transform=tx
                    )
                    ax.add_patch(patch)
                    if r.day > max_out:
                        max_out = r.day
                else:
                    t0 = np.pi / 2
                    if i > 0:
                        t0 -= cluster_angles.cumsum().iat[i - 1] + gap * i
                    t0 -= arrow_space / 2
                    t0 -= j * arrow_space
                    if j >= first_len:
                        t0 += arrow_space
                    next_cl = r.next_cluster
                    next_cl_pos = cluster_angles.index.get_loc(next_cl)
                    next_cl = np.pi / 2 - cluster_angles.cumsum()[next_cl] - gap * next_cl_pos
                    next_cl += cluster_angles[r.next_cluster] / 2
                    t1, r1 = (next_cl, 1.4)
                    r1 -= 0.06
                    path = get_arrow_path(ax, tx, (t0, r0), (t0, 1), (t1, 0.85), (t1, r1), base_w=lw)
                    patch = mpl.patches.PathPatch(
                        path, 
                        fc=color, 
                        ec=color, 
                        lw=1,
                        transform=tx,
                        zorder=r.day
                    )
                    ax.add_patch(patch)
                    text_r = r0 + 0.05
                    path_effects = path_effects = [
                        mpl.patheffects.Stroke(linewidth=3, foreground="white"),
                        mpl.patheffects.Normal()
                    ]
                    k = j
                    if j > first_len:
                        k -= 1
                    if arrow_space < 0.075 and k % 2 == 1:
                        text_r -= 0.1
                    ax.text(t0, text_r, pct_txt, ha="center", va="center", zorder=1000, path_effects=path_effects)

        disc_order = ["Died+Hospice", "LTACH+SNF", "ICU readm.", "Home+Rehab"]
        hatch_map = {
            "Home+Rehab": "",
            "ICU readm.": "////",
            "LTACH+SNF": "xxxx",
            "Died+Hospice": ""
        }
        disc_cmap = {
            "Home+Rehab": "w",
            "ICU readm.": "w",
            "LTACH+SNF": "w",
            "Died+Hospice": "black"
        }
        for i, cluster_size in enumerate(cluster_angles.tolist()):
            this_name = cluster_angles.index[i]
            disc = data.discharge[
                (data.cluster == this_name) & (data.next_cluster == -1)
            ].value_counts()
            t0 = np.pi / 2 - cluster_angles.cumsum().iat[i] - gap * i
            t0 += cluster_angles.iat[i] / 2
            pt_size = cluster_angles.iat[i] / max_out * 1.7
    #         if disc.sum() < 10:
    #             pt_size *= 1.2
            rect_widths = disc * pt_size
            total_width = rect_widths.sum()
            t0 -= total_width / 2
            for d in disc_order:
                if d not in disc:
                    continue
                fill = disc_cmap[d]
                hatch = hatch_map[d]
                rect = mpl.patches.Rectangle(
                    (t0, 2.02), 
                    rect_widths[d], 
                    0.08, 
                    fc=fill,
                    hatch=hatch,
                    zorder=20,
                    ec="black",
                    lw=0.5
                )
                ax.add_patch(rect)
                t0 += rect_widths[d]
        disc_handles = []
        for i, d in enumerate(disc_order):
            t = d
            if d == "ICU readm.":
                t = "ICU readmission"
            disc_handles.append(mpl.patches.Patch(color=disc_cmap[d], label=t, ec="black", hatch=hatch_map[d]))
        ax.legend(handles=disc_handles, loc="upper right", title="Discharge disposition", frameon=False)
        return fig


    # In[115]:


    # from https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib
    class CurvedText(mpl.text.Text):
        """
        A text object that follows an arbitrary curve.
        """
        def __init__(self, x, y, text, axes, **kwargs):
            super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

            axes.add_artist(self)

            ##saving the curve:
            self.__x = x
            self.__y = y
            self.__zorder = self.get_zorder()

            ##creating the text objects
            self.__Characters = []
            for c in text:
                if c == ' ':
                    ##make this an invisible 'a':
                    t = mpl.text.Text(0,0,'a')
                    t.set_alpha(0.0)
                else:
                    t = mpl.text.Text(0,0,c, **kwargs)

                #resetting unnecessary arguments
                t.set_ha('center')
                t.set_rotation(0)
                t.set_zorder(self.__zorder +1)

                self.__Characters.append((c,t))
                axes.add_artist(t)


        ##overloading some member functions, to assure correct functionality
        ##on update
        def set_zorder(self, zorder):
            super(CurvedText, self).set_zorder(zorder)
            self.__zorder = self.get_zorder()
            for c,t in self.__Characters:
                t.set_zorder(self.__zorder+1)

        def draw(self, renderer, *args, **kwargs):
            """
            Overload of the Text.draw() function. Do not do
            do any drawing, but update the positions and rotation
            angles of self.__Characters.
            """
            self.update_positions(renderer)

        def update_positions(self,renderer):
            """
            Update positions and rotations of the individual text elements.
            """

            #preparations

            ##determining the aspect ratio:
            ##from https://stackoverflow.com/a/42014041/2454357

            ##data limits
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            ## Axis size on figure
            figW, figH = self.axes.get_figure().get_size_inches()
            ## Ratio of display units
            _, _, w, h = self.axes.get_position().bounds
            ##final aspect ratio
            aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

            #points of the curve in figure coordinates:
            x_fig,y_fig = (
                np.array(l) for l in zip(*self.axes.transData.transform([
                (i,j) for i,j in zip(self.__x,self.__y)
                ]))
            )

            #point distances in figure coordinates
            x_fig_dist = (x_fig[1:]-x_fig[:-1])
            y_fig_dist = (y_fig[1:]-y_fig[:-1])
            r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

            #arc length in figure coordinates
            l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

            #angles in figure coordinates
            rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
            degs = np.rad2deg(rads)


            rel_pos = 10
            for c,t in self.__Characters:
                #finding the width of c:
                t.set_rotation(0)
                t.set_va('center')
                bbox1  = t.get_window_extent(renderer=renderer)
                w = bbox1.width
                h = bbox1.height
    #             print(w, h)

                #ignore all letters that don't fit:
                if rel_pos+w/2 > l_fig[-1]:
                    t.set_alpha(0.0)
                    rel_pos += w
                    continue

                elif c != ' ':
                    t.set_alpha(1.0)

                #finding the two data points between which the horizontal
                #center point of the character will be situated
                #left and right indices:
                il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
                ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

                #if we exactly hit a data point:
                if ir == il:
                    ir += 1

                #how much of the letter width was needed to find il:
                used = l_fig[il]-rel_pos
    #             rel_pos = l_fig[il]

                #relative distance between il and ir where the center
                #of the character will be
                fraction = (w/2-used)/r_fig_dist[il]
    #             fraction = (w/2)/r_fig_dist[il]

                ##setting the character position in data coordinates:
                ##interpolate between the two points:
                x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
                y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

                #getting the offset when setting correct vertical alignment
                #in data coordinates
                t.set_va(self.get_va())
                bbox2  = t.get_window_extent(renderer=renderer)

                bbox1d = self.axes.transData.inverted().transform(bbox1)
                bbox2d = self.axes.transData.inverted().transform(bbox2)
                dr = np.array(bbox2d[0]-bbox1d[0])

                #the rotation/stretch matrix
                rad = rads[il]
                rot_mat = np.array([
                    [math.cos(rad), math.sin(rad)*aspect],
                    [-math.sin(rad)/aspect, math.cos(rad)]
                ])

                ##computing the offset vector of the rotated character
                drp = np.dot(dr,rot_mat)

                #setting final position and rotation:
                t.set_position(np.array([x,y]))
    #             t.set_transform(mpl.transforms.Affine2D().rotate_deg_around(
    #                 0,
    #                 0,
    # #                 -(bbox2d[1][1] - bbox2d[0][1]) / 2,
    #                 degs[il]
    #             ) + self.axes.transData)
    #+drp)
                t.set_rotation_mode("anchor")
                t.set_rotation(degs[il])
                self.axes.add_artist(mpl.patches.Rectangle(
                    [x - (bbox2d[1][0] - bbox2d[0][0]) / 2, y],
                    width=bbox2d[1][0] - bbox2d[0][0],
                    height=bbox2d[1][1] - bbox2d[0][1],
                    fill=None,
                    ec="black",
                    alpha=0.0,
    #                 angle=degs[il]
                ))

    #             t.set_va('center')
                t.set_ha('center')

                #updating rel_pos to right edge of character
    #             rel_pos += w-used
                rel_pos += w

    # fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.spines['polar'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 5)
    # ax.bar(np.pi / 2 - 1, [1], bottom=1.7, width=1, align="edge", color="salmon")
    xline = np.linspace(np.pi / 2 - 1, np.pi / 2)
    yline1 = 4 - xline ** 0
    yline2 = 2 - xline ** 3
    ax.plot(xline, yline1)
    ax.plot(xline, yline2)
    ax.text(np.pi / 2 - 1, 4.5, "blah blah ARDS", weight="bold", size=20,
           transform=mpl.transforms.Affine2D().rotate_around(np.pi / 2 - 1, 4.5, 1) + ax.transData)
    CurvedText(
        xline,
        yline1,
        "blah blah ARDS",
        ax,
        va="bottom",
    #     ha="center",
        weight="bold",
        size=20
    );
    CurvedText(
        xline,
        yline2,
        "blah blah ARDS",
        ax,
        va="bottom",
    #     ha="center",
        weight="bold",
        size=20
    );


    # In[116]:


    df = data.copy()
    last_stay_per_patient = df.groupby("patient").agg({"stay": "max"})
    df["last_stay"] = last_stay_per_patient.stay[df.patient].values
    df["discharge"] = df.Discharge_disposition.replace({
        "Died": "Died+Hospice",
        "Hospice": "Died+Hospice",
        "Home": "Home+Rehab",
        "Rehab": "Home+Rehab",
        "SNF": "LTACH+SNF",
        "LTACH": "LTACH+SNF"
    })
    df.loc[(df.next_cluster == -1) & (df.stay != df.last_stay), "discharge"] = "ICU readm."


    # In[117]:


    # fig = ball_of_yarn(
    #     df.loc[~df.COVID_status, :], 
    #     pd.Series(1, index=sorted(df.cluster.unique()))
    # )
    # fig.axes[0].set_title("non-COVID transitions")
    # fig.tight_layout()

    # dump_figure('yarn_non_covid')


    # # In[118]:


    # fig = ball_of_yarn(
    #     df.loc[df.COVID_status, :], 
    #     pd.Series(1, index=sorted(df.cluster.unique()))
    # )
    # fig.axes[0].set_title("COVID transitions")
    # fig.tight_layout()

    # dump_figure('yarn_covid')


    # ## 3.3 Statistical tests for difference in transitions

    # For each cluster I compute number of transitions (including leaving ICU) into next clusters separately for COVID and non-COVID patients.
    # 
    # Next, I apply [chi-squared](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html#scipy.stats.chi2_contingency) test to see if the distribution of transitions is different between COVID and non-COVID patients. As per recommendations of the test, I exclude transitions to states that have <= 5 total transitions in either group. Finally, I perform FDR correction for acquired p-values.

    # In[119]:


    pvals = []
    for c in sorted(data.cluster.unique()):
        df_covid = data.loc[
            data.COVID_status & data.cluster.eq(c) & data.next_cluster.ne(data.cluster), 
            :
        ].groupby(
            ["cluster", "next_cluster"], dropna=False
        ).agg({"day": "count"}).reset_index()
        df_covid = df_covid.loc[df_covid.cluster.eq(c), :]
        df_non_covid = data.loc[
            ~data.COVID_status & data.cluster.eq(c) & data.next_cluster.ne(data.cluster), 
            :
        ].groupby(
            ["cluster", "next_cluster"], dropna=False
        ).agg({"day": "count"}).reset_index()
        df_non_covid = df_non_covid.loc[df_non_covid.cluster.eq(c), :]
        contingency = pd.DataFrame(dict(covid=df_covid.day, non_covid=df_non_covid.day)).T
        contingency = contingency.loc[:, contingency.gt(5).sum() == 2]
        if contingency.size == 0:
            print(f"No common transitions with > 5 counts for {c}")
            continue
        res = scipy.stats.chi2_contingency(contingency, correction=False)
        pvals.append((c, res[1]))
    pvals = pd.DataFrame(pvals, columns=["clinical_state", "p_value"])


    # In[120]:


    pvals["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(pvals.p_value, alpha=0.05)[1]


    # In[121]:


    pvals.loc[pvals.pval_adj < 0.05, :]


    # For 6 out of 14 clinical states we observe statistically significant difference

    # In[122]:


    for c in pvals.clinical_state[pvals.pval_adj < 0.05]:
        idx = data.cluster.eq(c) & data.next_cluster.ne(c)
        fig, ax = plt.subplots(figsize=(11, 4))
        df = data.loc[idx, :].groupby(["next_cluster", "COVID_status"]).agg(
            {"day": "count"}
        ).reset_index()
        df = df.loc[df.next_cluster != c, :]
        df.loc[df.COVID_status, "day"] /= df.day[df.COVID_status].sum()
        df.loc[~df.COVID_status, "day"] /= df.day[~df.COVID_status].sum()
        # sns.barplot(             # Deactivated to save memory from plotting 
        #     data=df,
        #     x="next_cluster",
        #     y="day",
        #     hue="COVID_status",
        #     ax=ax,
        #     palette=[DISCH_PALETTE[0], DISCH_PALETTE[-1]],
        #     saturation=1,
        #     linewidth=1,
        # )
        # ax.legend_.set_frame_on(False)
        # ax.legend_._loc = 6
        # ax.legend_.set_bbox_to_anchor((1, 0.5))
        # ax.set_title(f"Transitions from cluster {c}", size=16)
        # ax.set_ylabel("Proportions of transitions", size=16)
        # ax.set_xlabel("Next cluster", size=16)
        # plt.show()


    # In[123]:


    def roll_of_yarn(df, cluster_sizes, names=None, only_show=None, threshold=0):
        if only_show is None:
            only_show = cluster_sizes.index.tolist()
        if names is None:
            names = {}
        cluster_order = cluster_sizes.index
        gap = 1
        blue_to_red = {
            "red": [
                (0.0, 62/256, 62/256),
                (1.0, 216/256, 216/256),
            ],
            "green": [
                (0.0, 109/256, 109/256),
                (1.0, 62/256, 62/256),
            ],
            "blue": [
                (0.0, 216/256, 216/256),
                (1.0, 62/256, 62/256),
            ]
        }
        mortality_cmap = mpl.colors.LinearSegmentedColormap("mort_cmaap", segmentdata=blue_to_red, N=256)
        cdict = {
            'red':   [
                (0.0, 180/256, 180/256),
                (0.2, 256/256, 256/256),
                (1.0, 108/256, 108/256)
            ],

            'green': [
                (0.0, 180/256, 180/256),
                (0.2, 176/256, 176/256),
                (1.0, 0/256, 0/256)
            ],

            'blue':  [
                (0.0, 180/256, 180/256),
                (0.2, 27/256, 27/256),
                (1.0, 0/256, 0/256)
            ]
        }
        cmap = mpl.colors.LinearSegmentedColormap('exprCmap', segmentdata=cdict, N=256)
        lw = 1.5
        fig, ax = plt.subplots(figsize=(15, 8))
        tx = get_axes_pixel_transform(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([0.4, 9.6])
        ax.set_yticklabels(["non-COVID", "COVID"], size=16)
        ax.set_ylim(0, 10.5)
        ax.set_xlim(0, cluster_sizes.cumsum().iat[-1] + gap * (cluster_sizes.size - 1))
        data = df.copy()

        for covid in (False, True):
            df = data.loc[data.COVID_status == covid, :]
            df = df.loc[df.cluster != df.next_cluster, :].groupby(
                ["cluster", "next_cluster"], dropna=False
            ).agg({"day": "count"}).reset_index()
            df = df.loc[df.next_cluster != -1, :]
            df = df.loc[df.day > 0, :]
            df = df.reset_index()

            start = 0
            bottom = 9.2 if covid else 0
            xticks = []
            for i, cluster_size in enumerate(cluster_sizes.tolist()):
                this_name = cluster_order[i]
                this_color = mortality_cmap(cluster_mortality.mortality[this_name])
                ax.bar(start, 0.8, bottom=bottom, width=cluster_size, align="edge", color=this_color)
                xticks.append(start + cluster_size / 2)
                if covid:
                    ax.scatter([start + cluster_size / 2], [5], s=1000, color=this_color, zorder=1000)
                    ax.text(start + cluster_size / 2, 5, cluster_order[i], ha="center", va="center", weight="bold", zorder=1100)
                n_text = start + cluster_size / 2
                ax.text(
                    n_text, 
                    9.6 if covid else 0.4, 
                    f"$n$={df.day[df.cluster == this_name].sum()}", 
                    ha="center", 
                    va="center", 
                    zorder=11
                )
                start += cluster_size + gap
            ax.set_xticks(xticks)
            ax.set_xticklabels([names.get(x, x) for x in cluster_order.tolist()])
            max_out = 0
            for i, cluster_size in enumerate(cluster_sizes.tolist()):
                this_name = cluster_order[i]
                if this_name not in only_show:
                    continue
                df_slice = df.loc[df.cluster == this_name]
                df_slice = df_slice.loc[df_slice.day > threshold, :]
                if df_slice.size == 0:
                    continue
                arrow_space = cluster_size / df_slice.shape[0]
                for j, (_, r) in enumerate(df_slice.iterrows()):
                    r0 = 8.7 if covid else 1.25
                    pct_txt = str(r.day)
                    color = cmap(r.day / df.day.max())
                    lw = 0.0 + 8 * (r.day / df.day.max()) ** 0.8
                    t0 = 0
                    if i > 0:
                        t0 += cluster_sizes.cumsum().iat[i - 1] + gap * i
                    t0 += arrow_space / 2
                    t0 += j * arrow_space
                    next_cl = r.next_cluster
                    next_cl_idx = cluster_sizes.index.get_loc(next_cl)
                    next_cl_pos = 0
                    if next_cl_idx > 0:
                        next_cl_pos += cluster_sizes.cumsum()[next_cl - 1] + gap * next_cl_idx
                    next_cl_pos += cluster_sizes[next_cl] / 2
                    t1, r1 = (next_cl_pos, 5)
                    r1 -= -0.5 if covid else 0.5
                    r2 = 8 if covid else 2
                    r3 = 7 if covid else 3
                    path = get_arrow_path(ax, tx, (t0, r0), (t0, r2), (t1, r3), (t1, r1), base_w=lw)
                    patch = mpl.patches.PathPatch(
                        path, 
                        fc=color, 
                        ec=color, 
                        lw=1,
                        transform=tx,
                        zorder=r.day
                    )
                    ax.add_patch(patch)
                    if covid:
                        text_r = r0 + 0.3
                    else:
                        text_r = r0 - 0.3
                    path_effects = path_effects = [
                        mpl.patheffects.Stroke(linewidth=3, foreground="white"),
                        mpl.patheffects.Normal()
                    ]
                    if arrow_space < 2 and j % 2 == 1:
                        text_r += -0.25 if covid else 0.25
                    ax.text(t0, text_r, pct_txt, ha="center", va="center", zorder=1000, path_effects=path_effects)
        cbar_ax = fig.add_axes((0.05, 0.4, 0.02, 0.3))
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap=mortality_cmap), 
            cax=cbar_ax,
        )
        cbar.set_label("Cluster mortality", size=12)
        return fig


    # In[124]:


    fig = roll_of_yarn(
        data, 
        pd.Series(8, index=sorted(data.cluster.unique())),
        threshold=30
    )
    fig.axes[0].set_title("COVID vs. non-COVID transitions, transition threshold=30", size=16)
    fig.tight_layout()
    # fig.savefig("13covid-vs-non-covid-transitions.pdf")

    dump_figure('state_boxes_covid-vs-non-covid-transition')


    # In[125]:


    def roll_of_yarn2(df, cluster_sizes, names=None, only_show=None, threshold=0):
        if only_show is None:
            only_show = cluster_sizes.index.tolist()
        if names is None:
            names = {}
        cluster_order = cluster_sizes.index
        gap = 1
        blue_to_red = {
            "red": [
                (0.0, 62/256, 62/256),
                (1.0, 216/256, 216/256),
            ],
            "green": [
                (0.0, 109/256, 109/256),
                (1.0, 62/256, 62/256),
            ],
            "blue": [
                (0.0, 216/256, 216/256),
                (1.0, 62/256, 62/256),
            ]
        }
        mortality_cmap = mpl.colors.LinearSegmentedColormap("mort_cmaap", segmentdata=blue_to_red, N=256)
    #     cdict = {
    #         'red':   [
    #             (0.0, 180/256, 180/256),
    #             (0.2, 256/256, 256/256),
    #             (1.0, 108/256, 108/256)
    #         ],

    #         'green': [
    #             (0.0, 180/256, 180/256),
    #             (0.2, 176/256, 176/256),
    #             (1.0, 0/256, 0/256)
    #         ],

    #         'blue':  [
    #             (0.0, 180/256, 180/256),
    #             (0.2, 27/256, 27/256),
    #             (1.0, 0/256, 0/256)
    #         ]
    #     }
        min_mortality_diff = cluster_mortality.mortality.min() - cluster_mortality.mortality.max()
        max_mortality_diff = cluster_mortality.mortality.max() - cluster_mortality.mortality.min()
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'exprCmap', 
            [DISCH_PALETTE[0], DISCH_PALETTE[-1]]
        )
        cmap = sns.cm.vlag
        lw = 1.5
        fig, ax = plt.subplots(figsize=(15, 8))
        tx = get_axes_pixel_transform(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([0.4, 9.6])
        ax.set_yticklabels(["non-COVID", "COVID"], size=16)
        ax.set_ylim(-0.2, 10.5)
        ax.set_xlabel("Median number of days per Clinical State", size=16)
        # ax.set_xlim(0, cluster_sizes.cumsum().iat[-1] + gap * (cluster_sizes.size - 1))
        data = df.copy()

        for covid in (True, False):
            df = data.loc[data.COVID_status == covid, :]
            cl_size = df.groupby(["patient", "cluster"]).agg({"day": "count"}).reset_index().groupby(
                "cluster"
            ).day.median()

            df = df.loc[df.is_transition, :].groupby(
                ["cluster", "next_cluster"], dropna=False
            ).agg({"day": "count"}).reset_index()
            df = df.loc[df.day > 0, :]
            df = df.reset_index()

            start = 0
            bottom = 9.2 if covid else 0
            # if not covid:
                # gap = (rightmost - cl_size.sum()) / (cl_size.size - 1)
                # start = (rightmost - cl_size.sum() - (cl_size.size - 1) * gap) / 2
                # non_covid_start = start
            # xticks = []
            for i, cluster_size in enumerate(cl_size.tolist()):
                this_name = cluster_order[i]
                this_color = mortality_cmap(cluster_mortality.mortality[this_name])
                ax.bar(start, 0.8, bottom=bottom, width=cluster_size, align="edge", color=this_color)
                # xticks.append(start + cluster_size / 2)
                # if covid:
                #     ax.scatter([start + cluster_size / 2], [5], s=1000, color=this_color, zorder=1000)
                #     ax.text(start + cluster_size / 2, 5, cluster_order[i], ha="center", va="center", weight="bold", zorder=1100)
                n_text = start + cluster_size / 2
                ax.text(
                    n_text, 
                    9.6 if covid else 0.4, 
                    this_name, 
                    ha="center", 
                    va="center", 
                    zorder=11
                )
                start += cluster_size + gap

            if covid:
                pad = 2
                start = 0
                rightmost = cl_size.sum() + gap * (cl_size.size - 1)
                ax.set_xlim(0, rightmost)
                sink_gap = (rightmost - start) / cl_size.size
                sink_pos = {}
                # ax.axvline(start, color="red", lw=1)
                # ax.axvline(rightmost, color="red", lw=1)
                for i, _ in enumerate(cl_size.tolist()):
                    this_name = cluster_order[i]
                    this_color = mortality_cmap(cluster_mortality.mortality[this_name])
                    ax.scatter([start + sink_gap / 2], [5], s=1000, color=this_color, zorder=1000)
                    ax.text(start + sink_gap / 2, 5, cluster_order[i], ha="center", va="center", weight="bold", zorder=1100)
                    # ax.axvline(start + sink_gap / 2, color="red", lw=1)
                    sink_pos[cluster_order[i]] = start + sink_gap / 2
                    start += sink_gap
            # ax.set_xticks(xticks)
            # ax.set_xticklabels([names.get(x, x) for x in cluster_order.tolist()])

            max_out = 0
            for i, cluster_size in enumerate(cl_size.tolist()):
                this_name = cluster_order[i]
                if this_name not in only_show:
                    continue
                df_slice = df.loc[df.cluster == this_name]
                df_slice = df_slice.loc[df_slice.day > threshold, :]
                if df_slice.size == 0:
                    continue
                arrow_space = cluster_size / df_slice.shape[0]
                for j, (_, r) in enumerate(df_slice.iterrows()):
                    r0 = 8.7 if covid else 1.25
                    pct_txt = str(r.day)
                    mort_diff = cluster_mortality.mortality[r.next_cluster] - cluster_mortality.mortality[r.cluster]
                    # color = cmap(r.day / df.day.max())
                    # color = cmap((mort_diff - min_mortality_diff) / (max_mortality_diff - min_mortality_diff))
                    if mort_diff > 0:
                        color = "orange"
                    else:
                        color = "green"
                    lw = 0.0 + 8 * (r.day / df.day.max()) ** 0.8
                    t0 = 0
                    # if not covid:
                    #     t0 += non_covid_start
                    if i > 0:
                        t0 += cl_size.cumsum().iat[i - 1] + gap * i
                    t0 += arrow_space / 2
                    t0 += j * arrow_space
                    next_cl = r.next_cluster
                    next_cl_idx = cl_size.index.get_loc(next_cl)
                    next_cl_pos = sink_pos[next_cl]
                    # if next_cl_idx > 0:
                    #     next_cl_pos += cluster_sizes.cumsum()[next_cl - 1] + gap * next_cl_idx
                    # next_cl_pos += cluster_sizes[next_cl] / 2
                    t1, r1 = (next_cl_pos, 5)
                    r1 -= -0.5 if covid else 0.5
                    r2 = 8 if covid else 2
                    r3 = 7 if covid else 3
                    path = get_arrow_path(ax, tx, (t0, r0), (t0, r2), (t1, r3), (t1, r1), base_w=lw)
                    patch = mpl.patches.PathPatch(
                        path, 
                        fc=color, 
                        ec=color, 
                        lw=1,
                        transform=tx,
                        zorder=r.day
                    )
                    ax.add_patch(patch)
                    if covid:
                        text_r = r0 + 0.3
                    else:
                        text_r = r0 - 0.3
                    path_effects = path_effects = [
                        mpl.patheffects.Stroke(linewidth=3, foreground="white"),
                        mpl.patheffects.Normal()
                    ]
                    if arrow_space < 2 and j % 3 == 1:
                        text_r += -0.25 if covid else 0.25
                    if arrow_space < 2 and j % 3 == 2:
                        text_r += -0.5 if covid else 0.5
                    ax.text(t0, text_r, pct_txt, ha="center", va="center", zorder=1000, path_effects=path_effects)
        ax.set_xticks(range(0, int(rightmost), 5))
        cbar_ax = fig.add_axes((0.05, 0.4, 0.02, 0.3))
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap=mortality_cmap), 
            cax=cbar_ax,
        )
        cbar.set_label("Cluster mortality", size=12)
        return fig
    fig = roll_of_yarn2(
        data, 
        pd.Series(8, index=sorted(data.cluster.unique())),
        threshold=30
    )

    dump_figure('state_boxes')

    plt.close('all')



    return


def over_07_cluster_descriptions(user, outstem):


        #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


    # In[2]:


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    sys.path.append('./../src/')
    from manuscript import sankey_side_by_side as sankey
    from manuscript import clustering, datasets, inout, export


    import itertools
    import scipy.stats
    import statsmodels.stats.multitest
    import statannotations.Annotator
    from statsmodels.stats.multitest import multipletests

    from scipy.stats import fisher_exact



    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")
            

    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    # In[3]:


    save = True     # whether or not to save elment to disk
    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/07_cluster_descriptions'    # name of notebook

    data_to_load = f'{user}/{outstem}/05_join_clusters_umap/05_data_umap_clusters.csv.gz'

    def dump_table(df, name):
        if save:
            export.full_frame(
                user, 
                f'{outfolder}/{name}', 
                df, 
                index=True,
                date=False
            )

    def dump_figure(name):
        if save: 
            export.image(
                user,
                f'{outfolder}/{name}',
            )


    # In[4]:


    data = pd.read_csv(
        inout.get_material_path(data_to_load), 
        index_col=0)


    # In[5]:


    print(f"Total number of patients: {data.Patient_id.nunique()}")
    print(f"Total number of ICU-days: {data.shape[0]}")


    # In[ ]:





    # In[6]:



    data.Discharge_disposition = data.Discharge_disposition.astype("category")

    data.Discharge_disposition = data.Discharge_disposition.cat.reorder_categories([
        'Home',
        'Rehab', 
        'SNF',
        'LTACH',
        'Hospice',
        'Died'
    ])

    DISCHARGE_STACKS = [
        ('Home', 'Rehab', 'SNF', 'LTACH'),
        ('Hospice', 'Died')
    ]
    DISCH_PALETTE = [
        "tab:blue", #home
        "lightseagreen", #rehab
        "beige", #snf
        "gold",#ltach
        "orange",#hospice
        "crimson",#died 
    ]


    # In[7]:


    #rename to match old code 

    data = data.rename(columns={
        'Patient_id': 'patient',
        'ICU_stay': 'stay',
        'ICU_day': 'day',
        'clusters': 'cluster'
    })

    # Ensure order
    data.sort_values(["patient", "stay", "day"], inplace=True)


    # In[ ]:





    # # BALs per cluster

    # In[8]:


    df = data.loc[data.has_bal, :].groupby(["cluster"]).count().day.reset_index()


    # In[9]:


    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        data=df, 
        x="cluster", 
        y="day", 
        color='lightblue'
    )


    ax.set_ylabel("Number of ICU-days with BAL", size=16)
    ax.set_xlabel("Cluster", size=16)
    ax.set_title("BALs per cluster", size=16)

    dump_figure('BALs_per_cluster.pdf')


    # In[ ]:





    # In[ ]:





    # # First cluster per patient

    # In[10]:


    first_days = data.groupby("patient").head(1).index


    # In[11]:


    df = data.loc[first_days, :].groupby(["Discharge_disposition"]).agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_1": "cluster"}, axis=1)


    # In[12]:


    def plot(df):

        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        clustering.stacked_hue_barplot(
            df=df, 
            x="cluster", 
            y="count", 
            hue="Discharge_disposition", 
            stacks=DISCHARGE_STACKS,
            ax=ax,
            palette=mpl.colors.ListedColormap(DISCH_PALETTE),
        )
        ax.set_xlabel("Cluster", size=16)
        ax.set_ylabel("Number of patients", size=16)
        # ax.set_title("Distribution of first clusters for patients", size=16);
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-1, 14 * 9 - 1)
        
        
        

    def get_p_values(df):


        toy = df.copy()
        toy.loc[:, 'passed'] = toy['Discharge_disposition'].isin(['Hospice', 'Died'])
        total = toy.groupby('passed')['count'].sum()

        clusters = sorted(toy['cluster'].unique())
        res = pd.Series(index=clusters, data=False)
        for c in toy['cluster'].unique():
            d = toy[toy['cluster']==c]
            d = d.groupby(['passed'])['count'].sum()
            d = d.reindex([True, False]).fillna(0)

            a = d[True]
            b = d[False]
            a2 = total[True] - a
            b2 = total[False] - b
            _, pval = fisher_exact(
                ((a, b), (a2, b2))
            )
            res[c] = pval
        res = res.to_frame('fishers').rename_axis('cluster')
        res['benjamini_hochberg'] = multipletests(res['fishers'], method='fdr_bh')[1]

        return res    


    # In[13]:


    plot(df)  
    dump_figure('first_cluster.pdf')

    pvals = get_p_values(df)
    dump_table(pvals, 'first.xlsx')
    display(pvals)


    # In[ ]:





    # In[ ]:





    # # Last cluster for patient

    # In[14]:


    last_days = data.groupby("patient").tail(1).index


    # In[15]:


    df = data.loc[last_days, :].groupby(["Discharge_disposition"]).agg(
        {"cluster": "value_counts"}
    ).rename({"cluster": "count"}, axis=1).reset_index().rename({"level_1": "cluster"}, axis=1)


    # In[16]:


    plot(df)  
    dump_figure('last_cluster.pdf')

    pvals = get_p_values(df)
    dump_table(pvals, 'last_cluster.xlsx')
    display(pvals)


    # In[17]:


    for j in pvals.index:
        print(j, pvals.loc[j, 'benjamini_hochberg'])


    # In[ ]:


    return




def over_08_VAP_flows(user, outstem):

    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    sys.path.append('./../src/')
    from manuscript import flow_sankey as sankey
    from manuscript import datasets, inout, export


    import itertools
    import scipy.stats
    # import statsmodels.stats.multitest
    import statannotations.Annotator

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    # fonts = inout.get_resource_path('fonts')
    # for f in os.listdir(fonts):
    #     if f.endswith(".ttf"):
    #         mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")
            
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    def dump_table(df, name):
        export.full_frame(
            user, 
            f'{outfolder}/{name}', 
            df, 
            index=True,
            date=False
        )

    def dump_figure(name):
        export.image(
            user,
            f'{outfolder}/{name}',
        )
        


    # In[6]:



    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/08_VAP_flows'    # name of notebook


    # In[9]:



    data = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/05_join_clusters_umap/05_data_umap_clusters.csv.gz'), 
        index_col=0)


    # In[10]:


    # match old naming

    data.Binary_outcome.replace({1: "Died", 0: 'Alive'}, inplace=True)

    data.Discharge_disposition = data.Discharge_disposition.astype("category")
    data.Discharge_disposition = data.Discharge_disposition.cat.reorder_categories([
        'Home', 
        'Rehab', 
        'SNF', 
        'LTACH',
        'Hospice', 
        'Died'
    ])

    data = data.rename(columns={
        'Patient_id': 'patient',
        'ICU_stay': 'stay',
        'ICU_day': 'day',
        'clusters': 'cluster'
    })

    # Ensure order
    data.sort_values(["patient", "stay", "day"], inplace=True)


    # In[11]:


    df=data.copy()


    # In[12]:


    EPISODE_WINDOW = 2  # days


    # In[24]:


    episode_dfs = []
    for episode_type in ["CAP", "HAP", "VAP"]:
        dfs = []
        for day in range(-EPISODE_WINDOW, EPISODE_WINDOW + 1 + 5):
            days = data.day[data.Episode_category.eq(episode_type)] + day
            days = days[days > 0]
            idx = (
                data.patient[days.index].astype(str) 
                + "/" + data.stay[days.index].astype(str) 
                + "/" + days.astype(str)
            )
            idx = idx[idx.isin(data.index)]
            df = pd.DataFrame({
                "cluster_name": data.cluster[idx].values,
                "episode_type": data.Episode_category[idx.index].values,
                "day": day
            }, index=idx.index)
            full_idx = data.index[data.Episode_category.eq(episode_type) & data.Episode_etiology.ne("Viral")] #nonviral
            df = df.reindex(full_idx)
            df.loc[full_idx[~full_idx.isin(idx.index)], "day"] = day
            df.loc[full_idx[~full_idx.isin(idx.index)], "episode_type"] = episode_type
            df.loc[full_idx, "cured"] = data.Episode_is_cured[full_idx]
            df.loc[full_idx, "patient"] = data.patient[full_idx]
            df.loc[full_idx, "stay"] = data.stay[full_idx]
            dfs.append(df)
            
        df = pd.concat(dfs)
        episode_dfs.append(df)
        
    df = pd.concat(episode_dfs)


    # In[25]:


    df.cluster_name = df.cluster_name.astype(str)
    df.cluster_name.replace({"nan": "-1"}, inplace=True)
    df.cluster_name = df.cluster_name.astype("category")


    # In[26]:


    df_ = data.copy()
    df_["outcome"] = "Alive"
    df_.loc[df_.Binary_outcome.eq("Died"), "outcome"] = "Dead"
    last_stay_per_patient = df_.groupby("patient").agg({"stay": "max"})
    df_["last_stay"] = last_stay_per_patient.stay[df_.patient].values
    df_.loc[df_.stay != df_.last_stay, "outcome"] = "ICU readm."
    patient_outcome = df_.set_index(["patient", "stay"]).groupby(["patient", "stay"]).head(1).outcome


    # In[31]:


    def episodes_sankey(df, episode_type=None, cured=None):
        def flow_color(left, right):
            if left == right:
                return "correct"
            if left in ("Other", "Floor"):
                return "decrease"
            if right == "Alive":
                return "increase"
            if right == "Dead":
                return "mistake"
            if right == "ICU readm.":
                return "decrease"
            if left > right:
                return "increase"
            return "mistake"
        
        title = "All episodes"
        idx = pd.Series(True, index=df.index)
        if episode_type is not None:
            title = f"{episode_type} episodes only"
            idx = idx & df.episode_type.eq(episode_type)
        if cured is not None:
            title = f"{cured} episodes only"
            idx = idx & df.cured.eq(cured)
        if episode_type is not None and cured is not None:
            title = f"{cured} {episode_type} episodes only"
            
        sankey_df = df.reset_index().loc[
            idx.values, 
            ["index", "patient", "stay", "day", "cluster_name"]
        ].pivot(
            index=["index", "patient", "stay"],
            columns="day",
            values="cluster_name"
        ).reset_index()
        title += f" (n={sankey_df.shape[0]})"

        for c in sankey_df.columns[3:]:
            sankey_df[c] = sankey_df[c].astype(float).astype(int)

        for c in sankey_df.columns[3:]:
            idx = sankey_df[c].eq(-1)
            if c < 0:
                sankey_df.loc[idx & sankey_df.stay[idx].eq(1), c] = "Other"
                sankey_df.loc[idx & sankey_df.stay[idx].gt(1), c] = "Floor"
            else:
                outcome_idx = pd.MultiIndex.from_arrays([sankey_df.patient[idx], sankey_df.stay[idx]])
                sankey_df.loc[idx, c] = patient_outcome[outcome_idx].values

        colors = {
            i: mpl.cm.tab20(i - 1) 
            for i in range(1, 15)
        }
        colors["Other"] = "#999"
        colors["Floor"] = "#444"
        colors["ICU readm."] = "#444"
        colors["Alive"] = "#62b7d1"
        colors["Dead"] = "#ccc"
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        diag = sankey.Sankey(
            sankey_df.iloc[:, 3:], 
            ax=ax, 
            order=["Other", "Floor", "Alive", "ICU readm."] + list(range(1, 15)) + ["Dead"],
            block_width=0.2,
            # flow_color_func=flow_color,
            colors=colors,
            alpha=0.5
        )
        diag.draw()
        ax.set_title(f"{title}\n–2 days, episode start, +7 days", size=16)
        ax.set_xticks(
            [diag.block_width / 2 + diag.flow_width * x + diag.block_width * x for x in range(sankey_df.shape[1] - 3)]
        )
        ax.set_xticklabels(sankey_df.columns[3:].astype(int))
        ax.set_xlabel("Day relative to episode onset", size=14)
        ax.get_xaxis().set_visible(True)
        ax.tick_params(axis="x", pad=5, labelsize=16)
        return ax


    # In[32]:


    episodes_sankey(df);
    dump_figure('sankey_all_episodes.pdf')


    # In[33]:


    episodes_sankey(df, cured="Cured");
    dump_figure('sankey_all_episodes_cured.pdf')


    # In[34]:


    episodes_sankey(df, cured="Not cured");
    dump_figure('sankey_all_episodes_not_cured.pdf')


    # In[35]:


    episodes_sankey(df, episode_type="VAP");
    dump_figure('sankey_all_vap.pdf')


    # In[36]:


    episodes_sankey(df, episode_type="VAP", cured="Cured");
    dump_figure('sankey_cured_vap.pdf')


    # In[37]:


    episodes_sankey(df, episode_type="VAP", cured="Not cured");
    dump_figure('sankey_notcured_vap.pdf')


    # In[38]:


    episodes_sankey(df, episode_type="VAP", cured="Indeterminate");
    dump_figure('sankey_indeterminate_vap.pdf')


    # In[ ]:


def over_09_VAP_plots(user, outstem):


    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt


    import matplotlib.patches as mpatches
    from scipy.stats import fisher_exact

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    sys.path.append('./../src/')
    from manuscript import flow_sankey as sankey
    from manuscript import clustering, datasets, inout, export


    import itertools
    import scipy.stats
    import statsmodels.stats.multitest
    import statannotations.Annotator

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")
            
    # get_ipython().run_line_magic('load_ext', 'autoreload')
    # get_ipython().run_line_magic('autoreload', '2')
    # get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    def dump_table(df, name):
        export.full_frame(
            user, 
            f'{outfolder}/{name}', 
            df, 
            index=True,
            date=False
        )

    def dump_figure(name):
        export.image(
            user,
            f'{outfolder}/{name}',
        )
        


    # In[2]:


    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/09_VAP_plots'    # name of notebook


    # In[3]:


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


    # # Take only adjudicated patient-days

    # In[4]:


    data = pd.read_csv(
        inout.get_material_path("general/02_recode_transplants/02data-internal_220901_1009.csv.gz"), index_col=0)
    data['day_bucket_stars']=pd.to_datetime(data['day_bucket_starts'])

    #take episode data
    episodes = data.loc[~data.Episode_category.isna(), :]

    #first episode date 
    first_ep = episodes.sort_values(by=['patient','day_bucket_starts']).drop_duplicates(subset=['patient'])
    first_ep_day = first_ep[['patient','day_bucket_starts']]
    first_ep_day = first_ep_day.rename(columns={'day_bucket_starts':'enrollment_bal_date'})

    #merge back
    data_enroll_bal = pd.merge(data, first_ep_day, how='left', on='patient')

    #filter by enrollment bal 
    data_enroll_bal['after_enrollment'] = np.where((data_enroll_bal['day_bucket_starts']>=data_enroll_bal['enrollment_bal_date']),1,0)
    data_enroll_bal=data_enroll_bal[data_enroll_bal['after_enrollment']==1]

    data = data_enroll_bal.copy()


    # In[5]:


    data.shape


    # In[6]:


    data.patient.nunique()


    # In[7]:


    DISCH_PALETTE = [
        "tab:blue", #home
        "lightseagreen", #rehab
        "beige", #snf
        "gold",#ltach
        "orange",#hospice
        "crimson",#died 
    ]

    DISCHARGE_STACKS = [
        ('Home', 'Rehab', 'SNF', 'LTACH'),
        ('Hospice', 'Died')
    ]

    data.Discharge_disposition = data.Discharge_disposition.astype("category")
    data.Discharge_disposition = data.Discharge_disposition.cat.reorder_categories([
            'Home', 
            'Rehab', 
            'SNF', 
            'LTACH',
            'Hospice', 
            'Died'
        ])


    # # Select non-viral VAPs

    # In[8]:


    vap_idx = data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")
    vap_idx.sum()


    # In[9]:


    vap_days_idx = pd.DataFrame(False, index=data.index, columns=pd.MultiIndex.from_tuples([
        (False, "Cured"),
        (False, "Not cured/Indeterminate"),
        (True, "Cured"),
        (True, "Not cured/Indeterminate"),
    ]))
    vap_cured = pd.Series(index=data.index, dtype=pd.StringDtype())


    # # What % of COVID vs non-COVID patients had non-viral VAP in this adjudicated group?

    # In[10]:


    non_viral_vap = data[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")]
    unique = non_viral_vap.drop_duplicates(subset='patient')


    # In[11]:


    had_vap = unique.groupby('COVID_status').agg({"patient": "count"}).reset_index().rename({"patient": "had_nonviral_vap"}, axis=1)
    covid_split = data.drop_duplicates(subset='patient').groupby('COVID_status').agg({"patient": "count"}).reset_index().rename({"patient": "total_patients"}, axis=1)
    plot_df = pd.merge(had_vap, covid_split, how='left', on='COVID_status')
    plot_df['percent_had_vap']=plot_df['had_nonviral_vap']/plot_df['total_patients']
    plot_df['did_not']=plot_df['total_patients']-plot_df['had_nonviral_vap']
    plot_df['total']=1
    plot_df


    # In[12]:


    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.COVID_status.unique(), 2):
            had_nonviral_vap1 = plot_df['had_nonviral_vap'][plot_df.COVID_status==d1].dropna()
            had_nonviral_vap2 = plot_df['had_nonviral_vap'][plot_df.COVID_status==d2].dropna()
            did_not1 = plot_df['did_not'][plot_df.COVID_status==d1].dropna()
            did_not2 = plot_df['did_not'][plot_df.COVID_status==d2].dropna()

            odds, pval = fisher_exact([ [had_nonviral_vap1[0], had_nonviral_vap2[1]],
                                            [did_not1[0], did_not2[1]]])
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stat_results_sign


    # In[13]:


    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = sns.barplot(x="COVID_status",  y="total", data=plot_df, color='tab:blue')
    bar2 = sns.barplot(x="COVID_status", y="percent_had_vap", data=plot_df, color='crimson')

    top_bar = mpatches.Patch(color='tab:blue', label='No VAPs')
    bottom_bar = mpatches.Patch(color='crimson', label='Had at least one VAP')
    plt.legend(handles=[top_bar, bottom_bar], loc="lower right")
    # ax.legend_.set_bbox_to_anchor((0.5, 1))

    ax.set_ylabel("Proportion", size=16)

    ax.set_xlabel("")
    ax.set_xticklabels(["Non-COVID", "COVID"], size=16)

    ax.tick_params(axis='x', labelsize=12)
    trans = mpl.transforms.Affine2D().translate(6, 0)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")
        t.set_transform(t.get_transform() + trans)
        
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=plot_df, 
        x="COVID_status",
        y="percent_had_vap", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    annotator.annotate();
    dump_figure("had_nonviral_VAP.pdf")


    # # What % had more than one VAP?

    # In[14]:


    #count number of nonv-viral VAPs
    n_vaps = non_viral_vap.groupby(["patient", "COVID_status"]).apply(
        lambda x: x.Episode_category.eq("VAP").sum()
    ).reset_index().rename({0: "n_VAPs"}, axis="columns")
    n_vaps.groupby('COVID_status').agg({'patient':'count'}).reset_index().rename({'patient':'multiple_VAPs'})


    # In[15]:


    #multiple VAPs? 
    count_vaps = n_vaps[n_vaps.n_VAPs>1].COVID_status.value_counts().reset_index().rename({'COVID_status':'multiple_VAPs'})
    n_vaps[n_vaps.n_VAPs>1].COVID_status.value_counts().reset_index()


    # In[16]:


    plot_df2 = pd.merge(count_vaps, covid_split, how='left', left_on='index', right_on='COVID_status')
    plot_df2['percent_multiple_VAPs']=plot_df2['COVID_status_x']/plot_df2['total_patients']
    plot_df2['total']=1
    plot_df2['did_not']=plot_df2['total_patients']-plot_df2['COVID_status_x']
    plot_df2


    # In[17]:


    plot_df=plot_df2.copy()
    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.COVID_status_y.unique(), 2):
            had_multiple_vap1 = plot_df['COVID_status_x'][plot_df.COVID_status_y==d1].dropna()
            had_multiple_vap2 = plot_df['COVID_status_x'][plot_df.COVID_status_y==d2].dropna()
            did_not1 = plot_df['did_not'][plot_df.COVID_status_y==d1].dropna()
            did_not2 = plot_df['did_not'][plot_df.COVID_status_y==d2].dropna()

            stat, pval= fisher_exact([ [had_multiple_vap1[0], had_multiple_vap2[1]],
                                            [did_not1[0], did_not2[1]]])
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results

    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = sns.barplot(x="COVID_status_y",  y="total", data=plot_df2, color='tab:blue')
    bar2 = sns.barplot(x="COVID_status_y", y="percent_multiple_VAPs", data=plot_df2, color='crimson')

    top_bar = mpatches.Patch(color='tab:blue', label='Did not have multiple VAPs')
    bottom_bar = mpatches.Patch(color='crimson', label='Had multiple VAPs')
    plt.legend(handles=[top_bar, bottom_bar], loc="lower right")
    # ax.legend_.set_bbox_to_anchor((1, 0.8))

    ax.set_ylabel("Proportion", size=16)

    ax.set_xlabel("")
    ax.set_xticklabels(["Non-COVID", "COVID"], size=16)

    ax.tick_params(axis='x', labelsize=12)
    trans = mpl.transforms.Affine2D().translate(6, 0)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")
        t.set_transform(t.get_transform() + trans)
        
        
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=plot_df2, 
        x="COVID_status_y",
        y="percent_multiple_VAPs", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    annotator.annotate();

    dump_figure("had_multiple_nonviral_VAP.pdf")


    # # Single-episode patients with VAP split by cured

    # In[19]:


    #define cohort
    df = data.loc[data.Episode_category.isin(["CAP", "HAP", "VAP"]), :]
    one_episode_pts = df.groupby("patient").ICU_day.count().eq(1).replace({False: np.nan}).dropna().index
    plot_df = df.loc[vap_idx & df.patient.isin(one_episode_pts)]
    plot_df = plot_df.groupby(
        ["Discharge_disposition", "Episode_is_cured"]
    ).ICU_day.count().reset_index().rename({"ICU_day": "cnt"}, axis="columns")
    plot_df["Binary_outcome"] = "Positive"
    plot_df.loc[plot_df.Discharge_disposition.isin(["Hospice", "Died"]), "Binary_outcome"] = "Negative"
    simple_df = plot_df.groupby(["Episode_is_cured", "Binary_outcome"]).cnt.sum().reset_index()


    # Compute stats
    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])
    obs = helper.groupby(['Episode_is_cured', 'passed'])['cnt'].sum()

    stats_results = []

    states = ['Cured', 'Indeterminate', 'Not cured']
    # Changed `product` to `combinations`
    for cat_a, cat_b in itertools.combinations(states, r=2):
            pval = fisher_exact(
                (
                    (
                        obs.loc[(cat_a, False)], obs.loc[(cat_a, True)]
                    ),
                    (
                        obs.loc[(cat_b, False)], obs.loc[(cat_b, True)]
                    )
                )
            )[1]
            stats_results.append([cat_a, cat_b, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2", "pval"])
    pairs = []
    # for _, r in stats_results.iterrows():
    #         pairs.append((r.group1, r.group2))

    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]

    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    # pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stat_results_sign


    # In[20]:


    #define cohort
    df = data.loc[data.Episode_category.isin(["CAP", "HAP", "VAP"]), :]
    one_episode_pts = df.groupby("patient").ICU_day.count().eq(1).replace({False: np.nan}).dropna().index
    plot_df = df.loc[vap_idx & df.patient.isin(one_episode_pts)]
    plot_df = plot_df.groupby(
        ["Discharge_disposition", "Episode_is_cured"]
    ).ICU_day.count().reset_index().rename({"ICU_day": "cnt"}, axis="columns")
    plot_df["Binary_outcome"] = "Positive"
    plot_df.loc[plot_df.Discharge_disposition.isin(["Hospice", "Died"]), "Binary_outcome"] = "Negative"
    simple_df = plot_df.groupby(["Episode_is_cured", "Binary_outcome"]).cnt.sum().reset_index()

    #plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot Binary outcomes fully transparent
    sns.barplot(
        data=simple_df, 
        x="Episode_is_cured", 
        y="cnt",
        hue="Binary_outcome", 
        hue_order=["Positive", "Negative"],
        ax=ax, 
        alpha=0
    )

    ax.set_ylabel("Number of patients", size=16)
    ax.set_xlabel("Episode outcome", size=16)
    ax.set_title("Patients with a single episode of VAP", size=16)

    # Compute stats
    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])
    obs = helper.groupby(['Episode_is_cured', 'passed'])['cnt'].sum()

    stats_results = []

    states = ['Cured', 'Indeterminate', 'Not cured']
    # Changed `product` to `combinations`
    for cat_a, cat_b in itertools.combinations(states, r=2):
            pval = fisher_exact(
                (
                    (
                        obs.loc[(cat_a, False)], obs.loc[(cat_a, True)]
                    ),
                    (
                        obs.loc[(cat_b, False)], obs.loc[(cat_b, True)]
                    )
                )
            )[1]
            stats_results.append([cat_a, cat_b, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2", "pval"])
    pairs = []
    # for _, r in stats_results.iterrows():
    #         pairs.append((r.group1, r.group2))

    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]

    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    # pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
            
            
    # Annotate with stats the transparent plot
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();

    # Plot all Discharge dispositions
    annotator = statannotations.Annotator.Annotator(
        ax, 
        [(("Cured", "Positive"), ("Cured", "Negative"))], 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        hue="Binary_outcome",
        hue_order=["Positive", "Negative"],
        verbose=False,
    )
    max_stack = max([len(group) for group in DISCHARGE_STACKS])
    bottom = np.zeros(len(annotator._plotter.group_names))
    for i in range(max_stack):
        curr_stack = [group[-i - 1] for j, group in enumerate(DISCHARGE_STACKS) if i < len(group)]
        heights = plot_df.loc[plot_df.Discharge_disposition.isin(curr_stack)].set_index(
            ["Episode_is_cured", "Binary_outcome"]
        ).reindex(annotator._plotter.group_names).cnt.fillna(0)
        ax.bar(
            annotator._plotter.groups_positions._groups_positions_list, 
            heights, 
            color=[DISCH_PALETTE[data.Discharge_disposition.cat.categories.get_loc(x)] for x in curr_stack],
            ec="#333333",
            width=annotator._plotter.plotter.width / len(annotator._plotter.plotter.hue_names),
            align="center",
            bottom=bottom,
        )
        bottom += heights


    ax.legend(
        [
            plt.Rectangle(
                [0, 0], 0, 0,
                linewidth=0.5,
                edgecolor="#333",
                facecolor=color,
                label=label
            ) for color, label in zip(DISCH_PALETTE, data.Discharge_disposition.cat.categories)
        ],
        data.Discharge_disposition.cat.categories,loc=2, bbox_to_anchor=(1, 1),
    )
        
    dump_figure("single_episode_VAP_outcomes.pdf")


    # In[21]:


    #define cohort
    plot_df = df.loc[vap_idx & df.patient.isin(one_episode_pts)]
    plot_df = plot_df.loc[plot_df.COVID_status].groupby(
        ["Discharge_disposition", "Episode_is_cured"]
    ).ICU_day.count().reset_index().rename({"ICU_day": "cnt"}, axis="columns")
    plot_df["Binary_outcome"] = "Positive"
    plot_df.loc[plot_df.Discharge_disposition.isin(["Hospice", "Died"]), "Binary_outcome"] = "Negative"
    simple_df = plot_df.groupby(["Episode_is_cured", "Binary_outcome"]).cnt.sum().reset_index()

    #plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot Binary outcomes fully transparent
    sns.barplot(
        data=simple_df, 
        x="Episode_is_cured", 
        y="cnt",
        hue="Binary_outcome", 
        hue_order=["Positive", "Negative"],
        ax=ax, 
        alpha=0
    )

    ax.set_ylabel("Number of patients", size=16)
    ax.set_xlabel("Episode outcome", size=16)
    ax.set_title("Patients with a single episode of VAP (COVID only)", size=16)

    # Compute stats
    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])
    obs = helper.groupby(['Episode_is_cured', 'passed'])['cnt'].sum()

    stats_results = []

    states = ['Cured', 'Indeterminate', 'Not cured']
    # Changed `product` to `combinations`
    for cat_a, cat_b in itertools.combinations(states, r=2):
            pval = fisher_exact(
                (
                    (
                        obs.loc[(cat_a, False)], obs.loc[(cat_a, True)]
                    ),
                    (
                        obs.loc[(cat_b, False)], obs.loc[(cat_b, True)]
                    )
                )
            )[1]
            stats_results.append([cat_a, cat_b, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2", "pval"])
    # pairs = []
    # for _, r in stats_results.iterrows():
    #         pairs.append((r.group1, r.group2))

    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))



    # Annotate with stats the transparent plot
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();

    # Plot all Discharge dispositions
    annotator = statannotations.Annotator.Annotator(
        ax, 
        [(("Cured", "Positive"), ("Cured", "Negative"))], 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        hue="Binary_outcome",
        hue_order=["Positive", "Negative"],
        verbose=False,
    )
    max_stack = max([len(group) for group in DISCHARGE_STACKS])
    bottom = np.zeros(len(annotator._plotter.group_names))
    for i in range(max_stack):
        curr_stack = [group[-i - 1] for j, group in enumerate(DISCHARGE_STACKS) if i < len(group)]
        heights = plot_df.loc[plot_df.Discharge_disposition.isin(curr_stack)].set_index(
            ["Episode_is_cured", "Binary_outcome"]
        ).reindex(annotator._plotter.group_names).cnt.fillna(0)
        ax.bar(
            annotator._plotter.groups_positions._groups_positions_list, 
            heights, 
            color=[DISCH_PALETTE[data.Discharge_disposition.cat.categories.get_loc(x)] for x in curr_stack],
            ec="#333333",
            width=annotator._plotter.plotter.width / len(annotator._plotter.plotter.hue_names),
            align="center",
            bottom=bottom,
        )
        bottom += heights


    # ax.legend_.set_bbox_to_anchor((1, 1))
    # ax.legend_._loc = 2
    ax.legend(
        [
            plt.Rectangle(
                [0, 0], 0, 0,
                linewidth=0.5,
                edgecolor="#333",
                facecolor=color,
                label=label
            ) for color, label in zip(DISCH_PALETTE, data.Discharge_disposition.cat.categories)
        ],
        data.Discharge_disposition.cat.categories,loc=2, bbox_to_anchor=(1, 1),
    )
        
    dump_figure("single_episode_VAP_outcomes_COVID.pdf")


    # In[22]:



    # Compute stats
    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])
    obs = helper.groupby(['Episode_is_cured', 'passed'])['cnt'].sum()

    stats_results = []

    states = ['Cured', 'Indeterminate', 'Not cured']
    # Changed `product` to `combinations`
    for cat_a, cat_b in itertools.combinations(states, r=2):
            pval = fisher_exact(
                (
                    (
                        obs.loc[(cat_a, False)], obs.loc[(cat_a, True)]
                    ),
                    (
                        obs.loc[(cat_b, False)], obs.loc[(cat_b, True)]
                    )
                )
            )[1]
            stats_results.append([cat_a, cat_b, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2", "pval"])
    # pairs = []
    # for _, r in stats_results.iterrows():
    #         pairs.append((r.group1, r.group2))

    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stat_results_sign


    # In[23]:


    plot_df = df.groupby(["patient", "Discharge_disposition"]).apply(
        lambda x: x.Episode_category.eq("VAP").sum()
    ).reset_index().rename({0: "n_VAPs"}, axis="columns")
    plot_df = plot_df.groupby(["n_VAPs", "Discharge_disposition"]).agg(
        {"patient": "count"}
    ).reset_index().rename({"patient": "cnt"}, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    stacked_hue_barplot(
        df=plot_df, 
        x="n_VAPs", 
        y="cnt",
        hue="Discharge_disposition", 
        stacks=DISCHARGE_STACKS,
        ax=ax, 
        palette=mpl.colors.ListedColormap(DISCH_PALETTE),
    )
    ax.legend_._loc = 2
    ax.legend_.set_bbox_to_anchor((1, 1))
    ax.set_xlabel("Number of VAP episodes for patient", size=16)
    ax.set_ylabel("Number of patients", size=16)
    ax.set_title(
        "Discharge disposition by number of VAP episodes", 
        size=16
    );
    dump_figure('dispo_multiple_VAP_episodes.pdf')


    # In[24]:


    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])

    obs = helper.groupby(['n_VAPs', 'passed'])['cnt'].sum()

    for ep in [0, 1, 2, 3,4]:
        next_episodes = ep+1

        pval = fisher_exact(
            (
                (
                    obs.loc[(ep, False)], obs.loc[(ep, True)]
                ),
                (
                    obs.loc[(next_episodes, False)], obs.loc[(next_episodes, True)]
                )
            )
        )[1]

        print(f'{ep} to {next_episodes} has pvalue of {pval:.2f} to differ.')


    # In[25]:


    mortality_by_vaps = df.groupby(["patient","Binary_outcome"]).apply(
        lambda x: x.Episode_category.eq("VAP").sum()
    ).reset_index().rename({0: "n_VAPs"}, axis="columns")
    mortality_by_vaps = mortality_by_vaps.groupby(["n_VAPs", "Binary_outcome"]).agg(
        {"patient": "count"}
    ).reset_index().rename({"patient": "number"}, axis=1)
    mortality_by_vaps


    # In[26]:


    sum_vaps = mortality_by_vaps.groupby('n_VAPs').sum('number').reset_index()
    sum_vaps = sum_vaps[['n_VAPs','number']].rename(columns={'number':'total'})
    mortality_by_vaps = pd.merge(mortality_by_vaps, sum_vaps, how='left')
    mortality_by_vaps['percent'] = mortality_by_vaps['number']/mortality_by_vaps['total']
    mortality_by_vaps


    # # VAPs and mortality

    # In[27]:


    non_viral_vap = data[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")]
    unique = non_viral_vap.drop_duplicates(subset='patient')
    unique_list = unique['Patient_id']
    data['had_vap'] = np.where(data['Patient_id'].isin(unique_list),1,0)
    data_unique = data.drop_duplicates(subset='patient')


    # In[28]:


    plot_df = data_unique.copy()
    plot_df = data_unique.groupby(['Binary_outcome','had_vap']).agg({'patient':'count'}).reset_index().rename(columns={'patient':'count'})
    number_vaps = plot_df.groupby('had_vap').agg({'count':'sum'}).reset_index().rename(columns={'count':'total'})
    plot_df=pd.merge(plot_df,number_vaps)
    plot_df['whole']=1
    plot_df['percent']=plot_df['count']/plot_df['total']
    plot_df['Binary_outcome']=plot_df['Binary_outcome'].replace({0:'Died',1:'Lived'})
    plot_df['had_vap']=plot_df['had_vap'].replace({0:'No VAP',1:'Had at least one VAP'})
    plot_df2=plot_df[plot_df.Binary_outcome==('Lived')]
    plot_df2


    # In[29]:


    #Fisher exact p=0.1644


    # In[30]:


    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = sns.barplot(x="had_vap",  y="whole", data=plot_df2, color='tab:blue')
    bar2 = sns.barplot(x="had_vap", y="percent", data=plot_df2, color='crimson')

    top_bar = mpatches.Patch(color='tab:blue', label='Lived')
    bottom_bar = mpatches.Patch(color='crimson', label='Died')
    plt.legend(handles=[top_bar, bottom_bar], loc="lower right")
    # ax.legend_.set_bbox_to_anchor((1, 0.8))

    ax.set_ylabel("Proportion", size=16)
    ax.set_xlabel(" ", size=16)

    ax.tick_params(axis='x', labelsize=12)
    trans = mpl.transforms.Affine2D().translate(6, 0)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")
        t.set_transform(t.get_transform() + trans)
        
    # annotator = statannotations.Annotator.Annotator(
    #     ax, 
    #     pairs, 
    #     data=plot_df, 
    #     x="COVID_status",
    #     y="percent_had_vap", 
    #     verbose=False
    # )
    # annotator._verbose = False
    # annotator.configure(line_width=1)
    # annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    # annotator.annotate();
    dump_figure("VAP_status_lived.pdf")


    # In[31]:


    # Broken down by Category


    # In[32]:


    categories_vaps=data_unique.groupby(['Patient_category','Binary_outcome','had_vap']).agg({'patient':'count'}).reset_index()
    categories_vaps


    # In[33]:


    #differences not stat sig by Fisher's exact 

    fisher_exact( [
                ( categories_vaps.loc[((categories_vaps.Patient_category==('Non-Pneumonia Control')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Non-Pneumonia Control')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0]),
               ( categories_vaps.loc[((categories_vaps.Patient_category==('Non-Pneumonia Control')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Non-Pneumonia Control')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0] 
                )
                ]) 


    # In[34]:


    #differences not stat sig by Fisher's exact 

    fisher_exact( [
                ( categories_vaps.loc[((categories_vaps.Patient_category==('Other Viral Pneumonia')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Other Viral Pneumonia')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0]),
               ( categories_vaps.loc[((categories_vaps.Patient_category==('Other Viral Pneumonia')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Other Viral Pneumonia')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0] 
                )
                ]) 


    # In[35]:


    #differences not stat sig by Fisher's exact 

    fisher_exact( [
                ( categories_vaps.loc[((categories_vaps.Patient_category==('Other Pneumonia')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Other Pneumonia')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0]),
               ( categories_vaps.loc[((categories_vaps.Patient_category==('Other Pneumonia')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('Other Pneumonia')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0] 
                )
                ]) 


    # In[36]:


    #differences not stat sig by Fisher's exact 

    fisher_exact( [
                ( categories_vaps.loc[((categories_vaps.Patient_category==('COVID-19')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('COVID-19')) 
                         & (categories_vaps.Binary_outcome==0)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0]),
               ( categories_vaps.loc[((categories_vaps.Patient_category==('COVID-19')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==0)), 'patient'].values[0], 
                 categories_vaps.loc[((categories_vaps.Patient_category==('COVID-19')) 
                         & (categories_vaps.Binary_outcome==1)
                         & (categories_vaps.had_vap==1)), 'patient'].values[0] 
                )
                ]) 


    # In[37]:


    plot_df = data_unique.copy()
    plot_df = data_unique[data_unique.COVID_status].groupby(['Binary_outcome','had_vap']).agg({'patient':'count'}).reset_index().rename(columns={'patient':'count'})
    number_vaps = plot_df.groupby('had_vap').agg({'count':'sum'}).reset_index().rename(columns={'count':'total'})
    plot_df=pd.merge(plot_df,number_vaps)
    plot_df['whole']=1
    plot_df['percent']=plot_df['count']/plot_df['total']
    plot_df['Binary_outcome']=plot_df['Binary_outcome'].replace({0:'Died',1:'Lived'})
    plot_df['had_vap']=plot_df['had_vap'].replace({0:'No VAP',1:'Had at least one VAP'})
    plot_df2=plot_df[plot_df.Binary_outcome==('Lived')]
    plot_df2


    # In[38]:


    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = sns.barplot(x="had_vap",  y="whole", data=plot_df2, color='tab:blue')
    bar2 = sns.barplot(x="had_vap", y="percent", data=plot_df2, color='crimson')

    top_bar = mpatches.Patch(color='tab:blue', label='Lived')
    bottom_bar = mpatches.Patch(color='crimson', label='Died')
    plt.legend(handles=[top_bar, bottom_bar], loc="lower right")
    # ax.legend_.set_bbox_to_anchor((1, 0.8))

    ax.set_ylabel("Proportion", size=16)
    ax.set_xlabel("VAP and outcomes \n (COVID patients only)", size=16)

    ax.tick_params(axis='x', labelsize=12)
    # trans = mpl.transforms.Affine2D().translate(6, 0)
    # for t in ax.get_xticklabels():
    #     t.set_rotation(30)
    #     t.set_horizontalalignment("right")
    #     t.set_transform(t.get_transform() + trans)
        
    # annotator = statannotations.Annotator.Annotator(
    #     ax, 
    #     pairs, 
    #     data=plot_df, 
    #     x="COVID_status",
    #     y="percent_had_vap", 
    #     verbose=False
    # )
    # annotator._verbose = False
    # annotator.configure(line_width=1)
    # annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    # annotator.annotate();
    dump_figure("VAP_status_lived_COVID.pdf")


    # In[39]:


    categories_vaps['Binary_outcome'] = categories_vaps['Binary_outcome'].replace({0:'Died',1:'Lived'})


    # In[40]:


    categories_vaps.dtypes


    # In[41]:


    sns.color_palette("Paired")
    # sns.set(font_scale=1.5)
    # fig, ax = plt.subplots(figsize=(6, 6))

    sns.catplot(
        data=categories_vaps,
        x='had_vap',
        y='patient',
        hue='Binary_outcome',
        palette=['crimson','tab:blue'],
        col='Patient_category',
        kind='bar',
        height=4,
        aspect=0.7,
        ci=None)

    ax.set_xlabel("")
    ax.set_ylabel("Patients", size=16)
    # ax.set_xticklabels(["no VAP", "had VAP"], size=16)

    dump_figure('category_vap_outcome.pdf')


    # # Analyse only VAPs that didn't end in patient death

    # In[74]:


    #define cohort

    vap_df = data[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")].groupby("patient").tail(1)
    vap_df.Death_date = pd.to_datetime(vap_df.Death_date)
    vap_df.day_bucket_starts = pd.to_datetime(vap_df.day_bucket_starts)

    vap_df = vap_df.loc[~((vap_df.Death_date - vap_df.day_bucket_starts).dt.days < 14), :]
    long_vap_df = vap_df.copy()
    plot_df = long_vap_df.groupby(
        ["Discharge_disposition", "Episode_is_cured"]
    ).agg({"patient": "count"}).reset_index().rename({"patient": "cnt"}, axis=1)
    plot_df["Binary_outcome"] = "Positive"
    plot_df.loc[plot_df.Discharge_disposition.isin(["Hospice", "Died"]), "Binary_outcome"] = "Negative"
    simple_df = plot_df.groupby(["Episode_is_cured", "Binary_outcome"]).cnt.sum().reset_index()
    simple_df


    # In[75]:



    # Compute stats
    helper = plot_df.copy()
    helper.loc[:, 'passed'] = helper['Discharge_disposition'].isin(['Hospice', 'Died'])
    obs = helper.groupby(['Episode_is_cured', 'passed'])['cnt'].sum()

    stats_results = []

    states = ['Cured', 'Indeterminate', 'Not cured']
    # Changed `product` to `combinations`
    for cat_a, cat_b in itertools.combinations(states, r=2):
            pval = fisher_exact(
                (
                    (
                        obs.loc[(cat_a, False)], obs.loc[(cat_a, True)]
                    ),
                    (
                        obs.loc[(cat_b, False)], obs.loc[(cat_b, True)]
                    )
                )
            )[1]
            stats_results.append([cat_a, cat_b, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2", "pval"])
    pairs = []
    # for _, r in stats_results.iterrows():
    #         pairs.append((r.group1, r.group2))

    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]

    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    # pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results        


    # In[77]:



    #plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot Binary outcomes fully transparent
    sns.barplot(
        data=simple_df, 
        x="Episode_is_cured", 
        y="cnt",
        hue="Binary_outcome", 
        hue_order=["Positive", "Negative"],
        ax=ax, 
        alpha=0
    )

    ax.set_ylabel("Number of patients", size=16)
    ax.set_xlabel("Episode outcome", size=16)
    ax.set_title("Outcomes for patients who had VAP who didn't die within 14 days", size=16)

            
    # Annotate with stats the transparent plot
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs, 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        verbose=False
    )
    annotator._verbose = False
    annotator.configure(line_width=1)
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();

    # Plot all Discharge dispositions
    annotator = statannotations.Annotator.Annotator(
        ax, 
        [(("Cured", "Positive"), ("Cured", "Negative"))], 
        data=simple_df, 
        x="Episode_is_cured",
        y="cnt", 
        hue="Binary_outcome",
        hue_order=["Positive", "Negative"],
        verbose=False,
    )
    max_stack = max([len(group) for group in DISCHARGE_STACKS])
    bottom = np.zeros(len(annotator._plotter.group_names))
    for i in range(max_stack):
        curr_stack = [group[-i - 1] for j, group in enumerate(DISCHARGE_STACKS) if i < len(group)]
        heights = plot_df.loc[plot_df.Discharge_disposition.isin(curr_stack)].set_index(
            ["Episode_is_cured", "Binary_outcome"]
        ).reindex(annotator._plotter.group_names).cnt.fillna(0)
        ax.bar(
            annotator._plotter.groups_positions._groups_positions_list, 
            heights, 
            color=[DISCH_PALETTE[data.Discharge_disposition.cat.categories.get_loc(x)] for x in curr_stack],
            ec="#333333",
            width=annotator._plotter.plotter.width / len(annotator._plotter.plotter.hue_names),
            align="center",
            bottom=bottom,
        )
        bottom += heights


    ax.legend(
        [
            plt.Rectangle(
                [0, 0], 0, 0,
                linewidth=0.5,
                edgecolor="#333",
                facecolor=color,
                label=label
            ) for color, label in zip(DISCH_PALETTE, data.Discharge_disposition.cat.categories)
        ],
        data.Discharge_disposition.cat.categories,loc=2, bbox_to_anchor=(1, 1),
    )
        
    dump_figure("14days_VAP_outcomes.pdf")


    # # Are VAP episodes for COVID patient longer?

    # In[78]:


    data.Episode_duration[vap_idx].sum()


    # In[79]:


    plot_df = data.loc[vap_idx].copy()
    plot_df.Episode_is_cured.replace({
        "Not cured": "Not cured/Indeterminate",
        "Indeterminate": "Not cured/Indeterminate"
    }, inplace=True)


    # In[80]:


    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.COVID_status.unique(), 2):
            covid_d1_duration = plot_df.Episode_duration[plot_df.COVID_status==d1]
            covid_d2_duration = plot_df.Episode_duration[plot_df.COVID_status==d2]

            stat, pval= scipy.stats.mannwhitneyu(covid_d1_duration,covid_d2_duration)
                
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])
    stats_results


    # In[81]:


    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.COVID_status.unique(), 2):
            covid_d1_duration = plot_df.Episode_duration[plot_df.COVID_status==d1]
            covid_d2_duration = plot_df.Episode_duration[plot_df.COVID_status==d2]

            stat, pval= scipy.stats.mannwhitneyu(covid_d1_duration,covid_d2_duration)
                
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])

    stat_results_sign = stats_results.loc[stats_results.pval < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(
        data=plot_df,
        x="COVID_status",
        y="Episode_duration",
        palette=['tab:blue','crimson'],
        ax=ax,
        showfliers=True,
        saturation=1,
        linewidth=1
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average VAP episode\nduration per patient, days", size=16)
    ax.set_xticklabels(["non-COVID", "COVID"], size=16)

    # ax.legend_._loc = 2
    # ax.legend_.set_bbox_to_anchor((1, 1))
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs = pairs,
        data=plot_df, 
        x="COVID_status", 
        y="Episode_duration", 
        verbose=False
    )
    annotator._verbose = False
    annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    annotator.annotate();
    dump_figure('covid_vs_not_duration.pdf')


    # In[82]:


    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.Episode_is_cured.unique(), 2):
            cured_yes_duration = plot_df.Episode_duration[plot_df.Episode_is_cured=="Cured"]
            cured_no_duration = plot_df.Episode_duration[plot_df.Episode_is_cured=="Not cured/Indeterminate"]

            stat, pval= scipy.stats.mannwhitneyu(cured_yes_duration,cured_no_duration)
                
            stats_results.append([d1, d2, pval])  

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2","pval"])

    stat_results_sign = stats_results.loc[stats_results.pval < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(
        data=plot_df,
        x="Episode_is_cured",
        y="Episode_duration",
        palette=['tab:blue','crimson'],
        ax=ax,
        showfliers=True,
        saturation=1,
        linewidth=1
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average VAP episode\nduration per patient, days", size=16)
    ax.set_xticklabels(["Cured episodes", "Not cured/indeterminate"], size=16)

    # ax.legend_._loc = 2
    # ax.legend_.set_bbox_to_anchor((1, 1))
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs = pairs,
        data=plot_df, 
        x="Episode_is_cured", 
        y="Episode_duration", 
        verbose=False
    )
    annotator._verbose = False
    annotator.set_custom_annotations([f"p={x:.2e}" for x in stat_results_sign.pval])
    annotator.annotate();
    dump_figure('cured_vs_not_duration.pdf')


    # In[83]:


    plot_df = data.loc[vap_idx].copy()


    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.Episode_is_cured.unique(), 2):
            days1 = plot_df.Episode_duration[plot_df.Episode_is_cured==d1].dropna()
            days2 = plot_df.Episode_duration[plot_df.Episode_is_cured==d2].dropna()
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stats_results.append([d1, d2, days1.size, days2.size, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results


    # In[84]:




    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.Episode_is_cured.unique(), 2):
            days1 = plot_df.Episode_duration[plot_df.Episode_is_cured==d1].dropna()
            days2 = plot_df.Episode_duration[plot_df.Episode_is_cured==d2].dropna()
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stats_results.append([d1, d2, days1.size, days2.size, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results


    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(
        data=plot_df,
        x="Episode_is_cured",
        y="Episode_duration",
        palette=['tab:blue','orange','crimson'],
        ax=ax,
        showfliers=True,
        saturation=1,
        linewidth=1
    )

    ax.set_xlabel("Episode cure status", size=14)
    ax.set_ylabel("VAP episode duration (days)", size=14)

    # ax.legend_._loc = 2
    # ax.legend_.set_bbox_to_anchor((1, 1))
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs = pairs,
        data=plot_df, 
        x="Episode_is_cured", 
        y="Episode_duration", 
        verbose=False
    )
    annotator._verbose = False
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();
    dump_figure('cured_vs_indeterm_vs_not_duration.pdf')


    # In[85]:


    # check in group who doesn't die
     


    # In[87]:


    vap_df = data[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")].groupby("patient").tail(1)
    vap_df.Death_date = pd.to_datetime(vap_df.Death_date)
    vap_df.day_bucket_starts = pd.to_datetime(vap_df.day_bucket_starts)
    vap_df = vap_df.loc[~((vap_df.Death_date - vap_df.day_bucket_starts).dt.days < 14), :]
    long_vap_df = vap_df.copy()
    long_vap_df.shape


    # In[88]:


    long_vap_df.Episode_is_cured = long_vap_df.Episode_is_cured.astype("category")
    long_vap_df.Episode_is_cured = long_vap_df.Episode_is_cured.cat.reorder_categories([
            'Cured', 
            'Indeterminate', 
            'Not cured'
        ])


    # In[89]:


    plot_df = long_vap_df.copy()

    stats_results = []

    for d1, d2 in itertools.combinations(plot_df.Episode_is_cured.unique(), 2):
            days1 = plot_df.Episode_duration[plot_df.Episode_is_cured==d1].dropna()
            days2 = plot_df.Episode_duration[plot_df.Episode_is_cured==d2].dropna()
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            stats_results.append([d1, d2, days1.size, days2.size, pval])

    stats_results = pd.DataFrame(stats_results, columns=["group1", "group2",
                                                       "group1_size", "group2_size", "pval"])
    stats_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stats_results.pval, alpha=0.05)[1]
    stat_results_sign = stats_results.loc[stats_results.pval_adj < 0.05, :]
    pairs = []
    for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
    stats_results


    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(
        data=plot_df,
        x="Episode_is_cured",
        y="Episode_duration",
        palette=['tab:blue','orange','crimson'],
        ax=ax,
        showfliers=True,
        saturation=1,
        linewidth=1
    )

    ax.set_title("Episode duration for patients who had VAP who didn't die within 14 days", size=16)
    ax.set_xlabel("Episode cure status", size=14)
    ax.set_ylabel("VAP episode duration (days)", size=14)

    # ax.legend_._loc = 2
    # ax.legend_.set_bbox_to_anchor((1, 1))
    annotator = statannotations.Annotator.Annotator(
        ax, 
        pairs = pairs,
        data=plot_df, 
        x="Episode_is_cured", 
        y="Episode_duration", 
        verbose=False
    )
    annotator._verbose = False
    annotator.set_custom_annotations([f"q={x:.2e}" for x in stat_results_sign.pval_adj])
    annotator.annotate();
    dump_figure('cured_vs_indeterm_vs_not_duration_aliveat14.pdf')


    # # VAP episodes flow

    # In[90]:


    data = data.rename(columns={
        #'Patient_id': 'patient',
        'ICU_stay': 'stay',
        'ICU_day': 'day',
        'clusters': 'cluster'
    })
    data.Binary_outcome.replace({1: "Died", 0: 'Alive'}, inplace=True)


    # In[91]:


    dfs = []
    df_slice = data.loc[data.Episode_category.isin(["CAP", "HAP", "VAP", "Non-PNA-ctrl"]), :]
    max_episode = df_slice.groupby("patient").day.count().max()
    for episode_num in range(max_episode):
        episode_slice = df_slice.groupby("patient").nth(episode_num).reset_index()
        df = episode_slice.loc[:, ["Episode_category", "Episode_is_cured", "patient"]].copy()
        df["episode_num"] = episode_num + 1
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)


    # In[92]:


    sankey_df = df.reset_index().loc[
        :, 
        ["patient", "episode_num", "Episode_category"]
    ].pivot(
        index=["patient"],
        columns="episode_num",
        values="Episode_category"
    ).reset_index().fillna(-1)


    # In[93]:


    df_ = data.copy()
    df_["outcome"] = "Alive"
    df_.loc[df_.Binary_outcome.eq("Died"), "outcome"] = "Dead"
    patient_outcome = df_.set_index("patient").groupby("patient").head(1).outcome


    # In[94]:


    for c in sankey_df.columns[1:]:
        idx = sankey_df[c].eq(-1)
        sankey_df.loc[idx, c] = patient_outcome[sankey_df.patient[idx]].values


    # In[95]:


    colors = {
        "Non-PNA-ctrl": DISCH_PALETTE[0],
        "CAP": DISCH_PALETTE[1],
        "HAP": DISCH_PALETTE[3],
        "VAP": DISCH_PALETTE[5],
        "Alive": "#62b7d1",
        "Dead": "#ccc"
    }
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    diag = sankey.Sankey(
        sankey_df.iloc[:, 1:], 
        ax=ax, 
        order=["Alive", "Non-PNA-ctrl", "CAP", "HAP", "VAP", "Dead"],
        block_width=0.2,
        colors=colors,
        alpha=0.5
    )
    diag.draw()
    ax.set_title("", size=16)
    ax.set_xticks(
        [diag.block_width / 2 + diag.flow_width * x + diag.block_width * x for x in range(sankey_df.shape[1] - 1)]
    )
    ax.set_xticklabels(sankey_df.columns[1:].astype(int))
    ax.set_xlabel("Episode number", size=14)
    ax.get_xaxis().set_visible(True)
    ax.tick_params(axis="x", pad=5, labelsize=16)
    dump_figure('pneumonia_episodes_flow.pdf')


    # In[96]:


    df_slice = data.loc[data.Episode_category.isin(["VAP"]), :]
    dfs = []
    max_episode = df_slice.groupby("patient").day.count().max()
    for episode_num in range(max_episode):
        episode_slice = df_slice.groupby("patient").nth(episode_num).reset_index()
        df = episode_slice.loc[:, ["Episode_category", "Episode_is_cured", "patient"]].copy()
        df["episode_num"] = episode_num + 1
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    sankey_df = df.reset_index().loc[
        :, 
        ["patient", "episode_num", "Episode_category"]
    ].pivot(
        index=["patient"],
        columns="episode_num",
        values="Episode_category"
    ).reset_index().fillna(-1)

    df_ = data.copy()
    df_["outcome"] = "Alive"
    df_.loc[df_.Binary_outcome.eq("Died"), "outcome"] = "Dead"
    patient_outcome = df_.set_index("patient").groupby("patient").head(1).outcome

    for c in sankey_df.columns[1:]:
        idx = sankey_df[c].eq(-1)
        sankey_df.loc[idx, c] = patient_outcome[sankey_df.patient[idx]].values
        
    sankey_df['6']=patient_outcome[sankey_df.patient].values

    colors = {
        "Non-PNA-ctrl": DISCH_PALETTE[0],
        "CAP": DISCH_PALETTE[1],
        "HAP": DISCH_PALETTE[3],
        "VAP": DISCH_PALETTE[5],
        "Alive": "#62b7d1",
        "Dead": "#ccc"
    }
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    diag = sankey.Sankey(
        sankey_df.iloc[:, 1:], 
        ax=ax, 
        order=["Alive", "Non-PNA-ctrl", "CAP", "HAP", "VAP", "Dead"],
        block_width=0.2,
        colors=colors,
        alpha=0.5
    )
    diag.draw()
    ax.set_title("", size=16)
    ax.set_xticks(
        [diag.block_width / 2 + diag.flow_width * x + diag.block_width * x for x in range(sankey_df.shape[1] - 1)]
    )
    ax.set_xticklabels(sankey_df.columns[1:].astype(int))
    ax.set_xlabel("VAP episode number", size=14)
    ax.get_xaxis().set_visible(True)
    ax.tick_params(axis="x", pad=5, labelsize=16)
    dump_figure('vap_episodes_flow.pdf')


    # In[ ]:

    return



    # In[ ]:


def over_10_VAP_transitions(user, outstem):

    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import os
    import sys
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import IPython.display
    IPython.display.display(IPython.display.HTML("<style>.container { width:90% !important; }</style>"))

    sys.path.append('./../src/')
    from manuscript import flow_sankey as sankey
    from manuscript import datasets, inout, export


    import itertools
    import scipy.stats
    import statsmodels.stats.multitest
    import statannotations.Annotator

    pd.options.display.max_columns = 200
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    fonts = inout.get_resource_path('fonts')
    for f in os.listdir(fonts):
        if f.endswith(".ttf"):
            mpl.font_manager.fontManager.addfont(f"{fonts}/{f}")
            
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


    def dump_table(df, name, index=True):
        export.full_frame(
            user, 
            f'{outfolder}/{name}', 
            df, 
            index=index,
            date=False
        )

    def dump_figure(name):
        export.image(
            user,
            f'{outfolder}/{name}',
        )


    # In[2]:


    DISCH_PALETTE = [
        "tab:blue", #home
        "lightseagreen", #rehab
        "beige", #snf
        "gold",#ltach
        "orange",#hospice
        "crimson",#died 
    ]


    # In[3]:


    # user = 'tstoeger'     # defines top hierarchy of output folder
    outfolder = f'{outstem}/10_VAP_transitions'    # name of notebook


    # In[4]:


    data = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/05_join_clusters_umap/05_data_umap_clusters.csv.gz'), 
        index_col=0
    )


    # In[5]:


    # match old naming

    data.Binary_outcome.replace({1: "Died", 0: 'Alive'}, inplace=True)

    data.Discharge_disposition = data.Discharge_disposition.astype("category")
    data.Discharge_disposition = data.Discharge_disposition.cat.reorder_categories([
        'Home', 
        'Rehab', 
        'SNF', 
        'LTACH',
        'Hospice', 
        'Died'
    ])

    data = data.rename(columns={
        'Patient_id': 'patient',
        'ICU_stay': 'stay',
        'ICU_day': 'day',
        'clusters': 'cluster'
    })

    # Ensure order
    data.sort_values(["patient", "stay", "day"], inplace=True)


    # ## Create intermediate data

    # A dataframe for each day around each VAP episode: in which clusters were the patients.
    # 
    # At the same time, check if any other pneumonia episode is overlapping

    # In[6]:


    dfs = []
    for day in range(-2, 8):
        days = data.day[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")] + day
        days = days[days > 0]
        idx = (
            data.patient[days.index].astype(str) 
            + "/" + data.stay[days.index].astype(str) 
            + "/" + days.astype(str)
        )
        idx = idx[idx.isin(data.index)]
        df = pd.DataFrame({
            "cluster": data.cluster[idx].values,
            "next_cluster": data.next_cluster[idx].values,
            "episode_type": data.Episode_category[idx.index].values,
            "day": day
        }, index=idx.index)
        if day != 0:
            overlapping = data.loc[idx, ["Episode_category"]]
            overlapping["current_ep"] = idx.index.values
            overlapping = overlapping.loc[overlapping.Episode_category.isin(["CAP", "HAP", "VAP"]), :]
            if overlapping.shape[0] > 0:
                n_overlapping = (~overlapping.Episode_category.isna()).sum()
                print(f"{n_overlapping} overlapping episodes on day {day}")
                for i, row in overlapping.iterrows():
                    print(f"On day {day} of {row.current_ep} VAP episode there was {i} episode")
                print()
        full_idx = data.index[data.Episode_category.eq("VAP") & data.Episode_etiology.ne("Viral")]
        df = df.reindex(full_idx)
        df.loc[full_idx[~full_idx.isin(idx.index)], "day"] = day
        df.loc[full_idx[~full_idx.isin(idx.index)], "episode_type"] = "VAP"
        df.loc[full_idx, "cured"] = data.Episode_is_cured[full_idx]
        df.loc[full_idx, "patient"] = data.patient[full_idx]
        df.loc[full_idx, "stay"] = data.stay[full_idx]
        df.loc[full_idx, "outcome"] = data.Binary_outcome[full_idx]
        dfs.append(df)
    df = pd.concat(dfs)


    # In[7]:


    df.shape


    # # Does Cured VAP lead to significantly favorable transitions?

    # We will take all VAP episodes, and summarize transition deltas from day 0 to day X (7 or before).
    # 
    # Do it for 3 cured categories, and test statistical differences

    # #### Add mortality data

    # In[8]:


    df.loc[df.next_cluster.eq(-1), "next_cluster"] = np.nan


    # In[9]:


    cluster_mortality = pd.read_csv(
        inout.get_material_path(f'{user}/{outstem}/04_clustering/Similarity-cluster_mortality.csv.gz'), 
        index_col=0
    )


    # In[10]:


    cluster_mortality['clusters'] = cluster_mortality['index'] + 1


    # In[11]:


    cluster_mortality = cluster_mortality[['clusters', 'mortality']]


    # In[12]:


    cluster_mortality = cluster_mortality.set_index("clusters")


    # In[13]:


    df.loc[~df.cluster.isna(), "mortality"] = cluster_mortality.mortality[df.cluster[~df.cluster.isna()]].values


    # In[14]:


    df.loc[~df.next_cluster.isna(), "next_mortality"] = cluster_mortality.mortality[
        df.next_cluster[~df.next_cluster.isna()]
    ].values


    # In[15]:


    df.loc[
        df.next_cluster.isna() & df.outcome.eq("Alive"),
        "next_mortality"
    ] = 0


    # In[16]:


    df.loc[
        df.next_cluster.isna() & df.outcome.eq("Died"),
        "next_mortality"
    ] = 1


    # In[17]:


    df = df.loc[df.day >= 0].copy()


    # In[18]:


    df["mortality_delta"] = df.next_mortality - df.mortality


    # In[19]:


    df.mortality_delta.fillna(0, inplace=True)


    # In[20]:


    df["episode_start"] = df.index


    # In[21]:


    def vap_transitions(df, day):
        df = df.loc[df.day <= day]
        stat_df = df.groupby(["cured", "episode_start"]).mortality_delta.sum().reset_index()
        stat_results = []
        for d1, d2 in itertools.combinations(stat_df.cured.unique(), 2):
            days1 = stat_df.mortality_delta[stat_df.cured.eq(d1)]
            days2 = stat_df.mortality_delta[stat_df.cured.eq(d2)]
            if days1.size == 0 or days2.size == 0:
                continue
            pval = scipy.stats.mannwhitneyu(days1, days2).pvalue
            pval_two = scipy.stats.mannwhitneyu(days1, days2, alternative='two-sided').pvalue
            stat_results.append([
                d1, d2, days1.size, days2.size, pval, days1.median(), 
                days2.median(), days1.mean(), days2.mean(),
                pval_two
                ])
        stat_results = pd.DataFrame(stat_results, columns=["group1", "group2",
                                                           "group1_size", "group2_size", "pval", "group1_median", 
                                                           "group2_median", "group1_mean", "group2_mean",
                                                           "pval_two_sided"
                                                           ])
        stat_results["pval_adj"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval, alpha=0.05)[1]
        stat_results["pval_adj_two"] = statsmodels.stats.multitest.fdrcorrection(stat_results.pval_two_sided, alpha=0.05)[1]

        stat_results_sign = stat_results.loc[stat_results.pval_adj < 0.05, :]
        pairs = []
        for _, r in stat_results_sign.iterrows():
            pairs.append((r.group1, r.group2))
        
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.boxplot(
            data=stat_df, 
            x="cured",
            y="mortality_delta", 
            ax=ax, 
            saturation=1, 
            palette=[DISCH_PALETTE[0], DISCH_PALETTE[3], DISCH_PALETTE[-1]],
            linewidth=1
        )
        for a in ax.collections:
            if isinstance(a, mpl.collections.PatchCollection):
                # remove line surround each box
                a.set_linewidth(0)
        ax.set_ylabel("Sum of transitions", size=16)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=14)
        # ax.set_xticklabels(["non-COVID", "COVID"], size=16)
        # ax.legend(loc="upper left", title="Outcome", frameon=False, fontsize=14, title_fontsize=12)
        # ax.legend_.set_bbox_to_anchor((1, 0.8))
        ax.set_title(f"Distribution of sum of transitions for first {day} days after episode", size=16)
        annotator = statannotations.Annotator.Annotator(
            ax, 
            pairs, 
            data=stat_df, 
            x="cured",
            y="mortality_delta", 
            verbose=False
        )
        annotator._verbose = False
        annotator.configure(line_width=1)
        annotator.set_custom_annotations([f"q={x:.2e}".replace("-", "–") for x in stat_results_sign.pval_adj])
        annotator.annotate();
        return stat_results


    # In[22]:

    try:
        stat_results = vap_transitions(df, 7)
        dump_figure("vap_sum_of_trans_7days.pdf")
        dump_table(stat_results, "vap_sum_of_trans_7days_stats.xlsx", index=False)

    except:
        print(outstem, 'could not print vap_sum_of_trans_7days')


    # In[23]:


    try:
        stat_results = vap_transitions(df, 3)
        dump_figure("vap_sum_of_trans_3days.pdf")
        dump_table(stat_results, "vap_sum_of_trans_3days_stats.xlsx", index=False)

    except:
        print(outstem, 'could not print vap_sum_of_trans_3days')


    # # In[24]:


    # vap_transitions(df, 4)


    # # In[25]:


    # vap_transitions(df, 5)


    # # In[26]:


    # vap_transitions(df, 6)


    # # In[ ]:






