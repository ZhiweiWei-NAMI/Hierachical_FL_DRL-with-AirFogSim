import pandas as pd

# http://localhost:6008/data/plugin/scalars/scalars?tag={tag_name}_{rsu_id}&run={method_name}&experiment=&format=csv
# 从url下载csv文件，保存的命名格式为run_{method_name}-tag-{tag_name}_{rsu_id}.csv
method_name = ['MAPPO_AggFed_cluster3_local_max12nRBselffree','MAPPO_AggFed_cluster3_max12nRBselffree','Certain_FedAvg_max12nRBselffree','MAPPO_nFL_max12nRBselffree','MAPPO_Cen_max12nRBselffree']
method_name = ['MAPPO_AggFed_cluster2_local_max12nRBselffree','MAPPO_AggFed_cluster1_local_max12nRBselffree','MAPPO_Cen_nMA_max12nRBselffree','MAPPO_AggFed_cluster2_max12nRBselffree','greedy_notLearn_max12nRBselffree']
method_name = ['MAPPO_nFL_max12nRBselffree']
# tag_name = ['Loss%2FEntropy','Reward%2FTotal','Latency%2FTotal_Avg_Latency','Ratio%2FTotal_Failed_Ratio','Ratio%2FTotal_Offloaded_Ratio']
tag_name = ['Num%2FTask_Num']
rsu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for org_method in method_name:
    print(org_method)
    for post_fix in ['_0','_2']:
        method = org_method + post_fix
        for tag in tag_name:
            if 'greedy_notLearn' in method and tag == 'Loss%2FEntropy':
                continue
            if 'total' not in tag and 'Total' not in tag:
                for rsu_id in rsu_ids:
                    url = f'http://localhost:6008/data/plugin/scalars/scalars?tag={tag}_{rsu_id}&run={method}&experiment=&format=csv'
                    df = pd.read_csv(url)
                    # 把%2F替换为_
                    tmp_tag = tag.replace('%2F', '_')
                    df.to_csv(f'./run-{method}-tag-{tmp_tag}_{rsu_id}.csv', index=False)
            else:
                url = f'http://localhost:6008/data/plugin/scalars/scalars?tag={tag}&run={method}&experiment=&format=csv'
                df = pd.read_csv(url)
                # 把%2F替换为_
                tmp_tag = tag.replace('%2F', '_')
                df.to_csv(f'./run-{method}-tag-{tmp_tag}.csv', index=False)