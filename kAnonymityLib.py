import pandas as pd

class kAnonymity:
    def __init__(self):
        self.headers = []
        self.categorical = []
        self.feature_columns = []
        self.finished_partitions = []
        self.sensitive_column = ""
        self.dataframe = None
        self.result_df = None
        self.full_spans = {}
        
    def __str__(self):
        return "k-Anonymity Class Library with k=3"
    
    def set_headers(self, headers):
        self.headers = headers
        
    def file_to_list(self, textfile):
        df = pd.read_csv(textfile, header=None, names=["fields"])
        list = [x for x in df.fields ]
        return list

    def read_datafile(self, csvfile):
        if self.headers == []:
            self.dataframe = pd.read_csv(csvfile, index_col=False)
        else:
            self.dataframe = pd.read_csv(csvfile, header=None, names=self.headers, index_col=False)
    
    def set_categorial(self, categorical):
        self.categorical = categorical
        for name in self.categorical:
            self.dataframe[name] = self.dataframe[name].astype('category')
  
    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns

    def set_sensitive_column(self, sensitive_column):
        self.sensitive_column = sensitive_column

    def get_spans(self, df, partition, scale=None):
        spans = {}
        for column in df.columns:
            if column in self.categorical:
                span = len(df[column][partition].unique())
            else:
                span = df[column][partition].max()-df[column][partition].min()
            if scale is not None:
                span = span/scale[column]
            spans[column] = span
        return spans

    def split(self, df, partition, column):
        dfp = df[column][partition]
        if column in self.categorical:
            values = dfp.unique()
            lv = set(values[:len(values)//2])
            rv = set(values[len(values)//2:])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:        
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)
    
    def is_k_anonymous(self, partition, k=3):
        if len(partition) < k:
            return False
        return True

    def partition_dataset(self):
        self.full_spans = self.get_spans(self.dataframe, self.dataframe.index)
        self.finished_partitions = []
        partitions = [self.dataframe.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.get_spans(self.dataframe[self.feature_columns], partition, self.full_spans)
            for column, span in sorted(spans.items(), key=lambda x:-x[1]):
                lp, rp = self.split(self.dataframe, partition, column)
                if not self.is_k_anonymous(lp) or not self.is_k_anonymous(rp):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                self.finished_partitions.append(partition)            
        return 
   
    def agg_categorical_column(self, series):
        return [','.join(set(series))]

    def agg_numerical_column(self, series):
        return [series.mean()]

    def build_anonymized_dataset(self, max_partitions=None, k=3):
        rows = []
        if self.sensitive_column=="":
            cols = self.feature_columns
        else:
            cols = self.feature_columns+[self.sensitive_column]
        if len(self.finished_partitions) == 0:
            dfg = self.dataframe[cols].groupby(cols).size().reset_index(name='Count')
            df1 = dfg[ dfg.Count >= k ]
            master_list=[ df1[cols][df1.index==x].iloc[0].to_list() for x in df1.index ]
            data_store=[ [x, self.dataframe[cols][self.dataframe.index==x].iloc[0].to_list()] for x in self.dataframe.index ]
            new_index = [ x[0] for x in data_store if x[1] in master_list]
            new_list=[ self.dataframe[self.dataframe.index==x].iloc[0].to_list() for x in self.dataframe.index if x in new_index]
            self.result_df = pd.DataFrame(new_list)
            self.result_df.columns = self.dataframe.columns 
            del dfg
        else:
            aggregations = {}
            for column in self.feature_columns:
                if column in self.categorical:
                    aggregations[column] = self.agg_categorical_column
                else:
                    aggregations[column] = self.agg_numerical_column
            for i, partition in enumerate(self.finished_partitions):
                if max_partitions is not None and i > max_partitions:
                    break
                grouped_columns = self.dataframe.loc[partition].agg(aggregations, squeeze=False)
                sensitive_counts = self.dataframe.loc[partition].groupby(self.sensitive_column).agg({self.sensitive_column : 'count'})
                values = { x:grouped_columns[x][0] for x in grouped_columns.to_dict() }
                for sensitive_value, count in sensitive_counts[self.sensitive_column].items():
                    if count < k:
                        continue
                    values.update({
                        self.sensitive_column : sensitive_value,
                        'count' : count,

                    })
                    rows.append(values.copy())
            df1 = pd.DataFrame(rows)
        master_list=[ df1[cols][df1.index==x].iloc[0].to_list() for x in df1.index ]
        data_store=[ [x, self.dataframe[cols][self.dataframe.index==x].iloc[0].to_list()] for x in self.dataframe.index ]
        new_index = [ x[0] for x in data_store if x[1] in master_list]
        new_list=[ self.dataframe[self.dataframe.index==x].iloc[0].to_list() for x in self.dataframe.index if x in new_index]
        self.result_df = pd.DataFrame(new_list)
        self.result_df.columns = self.dataframe.columns 
        del df1, master_list, data_store, new_index, new_list
        return 

