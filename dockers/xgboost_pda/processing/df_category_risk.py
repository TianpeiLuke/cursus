import numpy as np
import pandas as pd


def df_category_risk(df, col_names, tags, default_risk=0.001, cnt_threshold=5, smoothing_factor=1, tag_range=1):
    '''
       for each given cols in df, for each category, compute the ratio of positive risk tags  vs. negative risk tags
       tag_range = 1 => default positive tag =1, negative tag=0
                = 2 => default positive tag =1, negative tag=-1
                
        Input:
	        df: pandas.DataFrame including col_names
	        col_names: list of `x`'s upon which the risk value mapping is on. 
	        tags: `y`, should be of the same number of rows as df
	        
	        
    '''
    
	# determine the binary set of label to be {0,1} or {1, -1}
    if tag_range == 1:
        POS_TAG = 1
        NEG_TAG = 0
    else:
        POS_TAG = 1
        NEG_TAG = -1
        
    ratio_dict_list = []
    count_dict_list = []
    
	# embed the label column into the dataframe 
    df['_'] = tags
    
	# compute the overall ratio of y=1, as the prior distribution P(y)
    global_pos_ratio = len(tags[tags == POS_TAG]) / len(tags)
    print(f"uniform prior {global_pos_ratio}")
	
	# for each `x` to compute the risk value
    for col in col_names:
        cat_risk_map = dict()
        cat_count_map = dict()
        
        # for each category in col compute the ratio of positive risk tags vs. negative risk tag
        # groupby col and tags
        # | col | tags  | counts |
	    # demoniator
        df_total = df.groupby([col]).agg({col: 'count'})
        # numerator
        df_col = df.groupby([col, '_']).agg({'_': 'count'})
        
        # groupby first level and take the ratio
        # | col | tags  | ratio  |
        # if the available tags for each category == 3 ? == 2? ==1 
        # smooth denomiator with a constant smoothing factor
        df_ratio = df_col.groupby(level=0, group_keys=False).apply(lambda x: (x + smoothing_factor*global_pos_ratio) / (float(x.sum())  + smoothing_factor) if x.shape[0] == 2 \
                                                      else (x + smoothing_factor*global_pos_ratio) / (float(x[1:].sum())  + smoothing_factor) if x.shape[0] > 2 \
                                                      else x/ (float(x.sum()) / smoothing_factor))
                                                      
        # only choose the row segment with condition `tags = POS_TAG`
        print()
        pos_count = df_col.xs(POS_TAG, level=1)
        risk_ratio = df_ratio.xs(POS_TAG, level=1)
        
        # output dict
        cat_risk_map = risk_ratio['_'].to_dict()
        cat_count_map = pos_count['_'].to_dict()
        cat_total_map = df_total[col].to_dict()
        
        # filter out small total count categorical values
        for key, val in cat_total_map.items():
            # for category that does not have postive tag 
            if key not in cat_risk_map:
                cat_risk_map[key] = global_pos_ratio #default using global_pos_ratio
                cat_count_map[key] = 0
            if val < cnt_threshold:
                cat_risk_map[key] = default_risk
        ratio_dict_list.append(cat_risk_map)
        count_dict_list.append(cat_count_map)
    
    df.drop(['_'], axis=1, inplace=True)
    return ratio_dict_list, count_dict_list