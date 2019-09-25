from numpy import nan, zeros
import re
import PPS as P
import numpy as np
import pandas as pd
from copy import deepcopy
from copy import deepcopy
from pandas import read_csv, get_dummies, Series
from sklearn.preprocessing import PowerTransformer
import _thread
import time
import warnings
warnings.filterwarnings("ignore")


class deprecated:
    """ Removed Functions """

    def load_csv_dep(self):
        """Read data as csv and return as pandas data frame."""
        
        filePath=self.filename
        missing_headers=self.missing_headers
        print("Reading Data From : "+ filePath.upper())
        if missing_headers:
            data = read_csv(filePath, header=None)
        else:
            data = read_csv(filePath, header=0)
        global rows, cols
        rows, cols = data.shape
        data=self.whitespace_removal(data)
        
        return data


class Initialization:
    """ Variable Initialization Class for various parameters which we pass """    
    
    def __init__(self,file_name,target_column,prediction_type,check_only_flag=False,missing_headers=False,RemvRedCol_StatsInvariant = 0.9,RemvRedCol_NullCount=0.1,
        RemvRedCol_MinColumnLimit=10,RemvRedCol_MinRows=10000,ExtrctNumCol_MinRowThreshold=0.01,RplcMisngData_RowDelThreshold=0.01,RplcMisngDta_MinRows=1000,
        CatgclDetct_MaxUnqValsThreshold=0.005,Outlier_StdDevMeanMultiFactor=3,Imbalncd_MaxMinDiffThreshold=0.05,ChkGausNorm_MinDataStdThreshold=0.66,
        ChkGausNorm_MaxDataStdThreshold=0.70,RemvRedCol_MaxLength=100,RemvRedCol_IgnoreUniqueFlag=False,NumrcDtaBins_MaxUnqValsThreshld=0.005,**kwargs): 
        
        options = {
            'categorical_columns_list' : [],
            'numeric_columns_list':[],
            'ordinal_col_list_order': {},
            'ForcedImpute':[],
            'user_imputation_col_name':[],
            'user_imputation_val':[],
        }
        options.update(kwargs)
        
        self.filename=file_name
        self.target=target_column
        self.check_only_flag=check_only_flag
        self.missing_headers=missing_headers
        self.categorical_columns_list = kwargs.get("categorical_columns_list")
        self.numeric_columns_list = kwargs.get("numeric_columns_list")
        self.user_imputation_col_name = kwargs.get("user_imputation_col_name")
        self.user_imputation_val = kwargs.get("user_imputation_val")
        self.ForcedImpute = kwargs.get("ForcedImpute")
        self.ordinal_col_list_order = kwargs.get("ordinal_col_list_order")
        self.prediction_type = prediction_type
        self.RemvRedCol_StatsInvariant = RemvRedCol_StatsInvariant
        self.RemvRedCol_MaxLength = RemvRedCol_MaxLength
        self.RemvRedCol_IgnoreUniqueFlag = RemvRedCol_IgnoreUniqueFlag
        self.RemvRedCol_NullCount = RemvRedCol_NullCount
        self.RemvRedCol_MinColumnLimit = RemvRedCol_MinColumnLimit
        self.RemvRedCol_MinRows = RemvRedCol_MinRows
        self.ExtrctNumCol_Regex_NumericColumns ="^\d+,{0,1}\d+\.{0,1}\d+$"
        self.ExtrctNumCol_MinRowThreshold = ExtrctNumCol_MinRowThreshold
        self.RplcMisngDta_RowDelThreshold = RplcMisngData_RowDelThreshold
        self.RplcMisngDta_MinRows = RplcMisngDta_MinRows
        self.CatgclDetct_MaxUnqValsThreshold = CatgclDetct_MaxUnqValsThreshold
        self.NumrcDtaBins_MaxUnqValsThreshld = NumrcDtaBins_MaxUnqValsThreshld
        self.Outlier_StdDevMeanMultiFactor = Outlier_StdDevMeanMultiFactor
        self.Imbalncd_MaxMinDiffThreshold = Imbalncd_MaxMinDiffThreshold
        self.ChkGausNorm_MinDataStdThreshold = ChkGausNorm_MinDataStdThreshold
        self.ChkGausNorm_MaxDataStdThreshold = ChkGausNorm_MaxDataStdThreshold


class PreProcessingInternalUtilites(Initialization):
    """ Internal Utilites Class for functions that are called by External Utilites """

    def casing_resolve(self,data,categorical_casing_list):
        """Moves text to lower casing if casing issues are present."""
        
        print("Resolve Casing Issue")
        for col in categorical_casing_list:
            data[col]=data[col].astype(str).str.lower()

        return data

    def OrdinalVariablesList(self,data):
        """ Ordinal Variables Conversion to Numerical Data. Using the List."""
        
        self.ordinal_col_list=[]
        print("Ordinal Variables conversion to Numerical Data.")
        for key,val in self.ordinal_col_list_order.items():
            self.ordinal_col_list.append(key)
            list_df_data=data[key]
            list_new=[]
            for i in list_df_data:
                list_new.append(val.get(i))
            data[key]=list_new
        print("List of Ordinal Columns : "+str(self.ordinal_col_list))

        return data

    def check_user_input(self,data,list_columns,column_name):
        """Check whether user input is right wrt to data frame input.
        We all know how meh! type most users are."""

        data_list=(data.columns)
        user_list=(list_columns)
        if(all(x in data_list for x in user_list)==False):
            for element in user_list:
                if element not in data_list:
                    print('KeyError: In '+column_name+'. The following columns was not present \''+str(format(element))+'\'')
                    _thread.interrupt_main()

    def check_ordinal_data(self,data):
        """Check whether user input is right wrt to data frame input for the column name"""

        for key,val in self.ordinal_col_list_order.items():
            if key not in data.columns:
                print('KeyError: In Ordinal Dictonary Column. The following columns was not present \''+str(format(key))+'\'')
                _thread.interrupt_main()

    def check_ordinal_data_val(self,data):
        """Check whether user input is right wrt to data frame input for the values in the columns"""

        for col_name,val in self.ordinal_col_list_order.items():
            for col_val,col_int in val.items():
                if(col_val not in data[col_name].unique()):
                    print('KeyError: In Ordinal Dictonary Column. The following value was not present : \''+str(format(col_val))+'\'')
                    _thread.interrupt_main()
            if(len(val.items())<len(data[col_name].unique())):
                print('KeyError: In Ordinal Dictonary Column. The # of vals which were given for the following column were not sufficient : \''+str(format(key))+'\'')
                _thread.interrupt_main()


    def whitespace_removal(self,data):
        """Removes Whitespace from data"""
        
        print("Removing Whitespace From Data")
        data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        return data

    def remove_outliers(self,data, list_numeric_cols):
        """Remove outliers from data and return as a pandas data frame."""
        
        print("Remove Outliers for Dataset")
        mean = data.describe().iloc[1, :]
        std = data.describe().iloc[2, :]
        
        for (list_numeric_cols, mean, std) in zip(list_numeric_cols, mean, std):
            data = data[abs(data[list_numeric_cols]) < self.Outlier_StdDevMeanMultiFactor*std + mean]
        return data

    def perform_gauss_normalization(self,data,numeric_cols):
        """Utility Function for check_gauss_normalization"""
        
        print("Performing Power Transformations")
        data.dropna(inplace=True)
        transformed_df=data[numeric_cols]
        pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
        pt.fit(transformed_df)
        transformed_df=pt.transform(transformed_df)
        transformed_df=pd.DataFrame(transformed_df)
        data[numeric_cols]=transformed_df
        return data
        #pt.inverse_transform(transformed_df)

    def imputation_function(self,data,user_imputation_col_name,user_imputation_val,prediction_type):
        """Missing Data function for user based input if user input ignored can
        override the drop statement if too large rows are dropped by using some statisitcs"""

        print("Performing Imputation for missing values")


    def one_hot_encoding(self,data,categorical_variables):
        """Perform a one-hot encoding and return as pandas data frame."""
        if(len(categorical_variables)==0):
            return data
        if(self.target in categorical_variables):
            categorical_variables.remove(self.target)
        df=data[categorical_variables]
        print(categorical_variables)
        data.drop(list(categorical_variables),axis=1)
        df=get_dummies(df)
        
        return data.join(df)


class PreProcessingExternalUtilites(PreProcessingInternalUtilites):
    """ Parent Utilites Class for functions that are called by PreProccesing Flow Class """      

    def load_csv(self):
        """Read data as csv and return as pandas data frame."""

        super()
        filePath=self.filename
        missing_headers=self.missing_headers
        print("Reading Data From : "+ filePath.upper())
        if missing_headers:
            data = read_csv(filePath, header=None)
        else:
            data = read_csv(filePath, header=0)
        _thread.start_new_thread( self.check_user_input, ( data,self.categorical_columns_list,"categorical_columns_list" ) )
        _thread.start_new_thread( self.check_user_input, ( data,self.numeric_columns_list,'numeric_columns_list' ) )
        _thread.start_new_thread( self.check_ordinal_data, (data,))
        _thread.start_new_thread( self.check_ordinal_data_val, (data,)) 
        
        global rows, cols
        rows, cols = data.shape

        data=self.whitespace_removal(data)

        return data
        
    
        
    def remove_redundant_columns(self,data):
        """Remove Redundant Columns Empty Columns and Grain Columns"""
        
        print("Removing Redundant Columns From Data")
        redundant_list=[]
        d_null_series=data.isna().sum()
        frame=pd.DataFrame([d_null_series])
        frame_unique=pd.DataFrame([data.nunique()])
        for i in frame_unique.columns:
            if(frame_unique[i][0]>(self.RemvRedCol_StatsInvariant)*data.shape[0] and len(str(data[i][0]))<self.RemvRedCol_MaxLength and self.RemvRedCol_IgnoreUniqueFlag):
                data.drop([i],axis=1,inplace=True)
                redundant_list.append(i)
        for i in frame.columns:
            if(frame[i][0]> self.RemvRedCol_NullCount*data.shape[0] and data.shape[1]>self.RemvRedCol_MinColumnLimit and data.shape[0]>self.RemvRedCol_MinRows):
                data.drop([i],axis=1,inplace=True)
                redundant_list.append(i)
        print("Columns Removed are:"+str(redundant_list))
        
        return data
    
    def drop_duplicated_data(self,data):
        """Removed duplicates from data"""
        
        print("Dropping Duplicated Data")
        print("# of Duplicate Data Records : "+str(data[data.duplicated()==True].shape[0]))
        data.drop_duplicates(inplace=True)
        
        return data
    
    def replace_missing_data(self,data):
        """Replace missing data values and return as pandas data frame."""
        
        print("Replace Missing Data")
        data = data.replace('?', nan)

        nan_vals = dict(data.count(axis=1))
        nan_vals = {key: value for (key, value) in nan_vals.items() if value < data.shape[1]-2}
        print("Rows affected are : "+ str(len(nan_vals)))
        if(len(nan_vals)> self.RplcMisngDta_RowDelThreshold*data.shape[0] or data.shape[0]<self.RplcMisngDta_MinRows):  ## ARgument to always impute
            data=self.imputation_function(data)
            return data
        data = data.drop(index=nan_vals.keys())
        
        return data
    
    def extract_numeric_columns(self,data,col_list=[]):
        """ Extracting Numeric Columns from string type.
        Also, generating list of all numeric type columns detected. """
        
        print("Extracting Numeric Columns")
        numeric_cols=[]
        numeric_string_cols=[]
        data=(self.OrdinalVariablesList(data))
        if(len(col_list)==0):
            col_list=data.columns
        for col in col_list:
            val=data[col][data[col].astype(str).str.contains(self.ExtrctNumCol_Regex_NumericColumns,na = False)]
            if(val.all() and val.size>(self.ExtrctNumCol_MinRowThreshold)*rows):
                numeric_string_cols.append(col)
        numeric_string_cols.extend(self.ordinal_col_list)
        print("List of Numeric columns : " + str(numeric_string_cols))
        for col in numeric_string_cols:
            data[col] = data[col].astype(str).str.replace(",","").astype(float)
        numeric_cols.extend(list(data.select_dtypes(exclude=['object']).columns))
        self.list_numeric=list(set(numeric_string_cols))
        return data

    def categorical_variables(self,data):   
        """ Identify Categorical Variables and Perform One Hot Encoding. """
        
        print("Identifying categorical variables")
        categorical_variables=self.categorical_columns_list
        text_columns=list(data.select_dtypes(exclude=[np.number]).columns)
        frame_unique=pd.DataFrame([data[text_columns].nunique()])
        for i in frame_unique.columns:
            if(frame_unique[i][0]<(self.CatgclDetct_MaxUnqValsThreshold)*data.shape[0]):
                categorical_variables.append(i)
        categorical_variables=list(set(categorical_variables))
        print("List of Categorical Variables : " + str(categorical_variables))
        self.categorical_columns_list = categorical_variables
        data=self.one_hot_encoding(data,self.categorical_columns_list)
        return data

    def outlier_list(self,data,list_numeric_cols):
        """Produces a list of numeric columns that contain any outlier 
        and the count of outliers for the same. It also removes the outliers."""
        
        print("Checking Dataset For Outliers")
        outlier_list_count=[]
        mean = data.describe().iloc[1, :]
        std = data.describe().iloc[2, :]
        for (col, mean, std) in zip(list_numeric_cols, mean, std):
            count=0
            for i in data[col]:
                if(i> float(self.Outlier_StdDevMeanMultiFactor*std + mean)):
                    count+=1
            outlier_list_count.append([col,count])
        print(outlier_list_count)
        data=self.remove_outliers(data,self.list_numeric)
    
    def categorical_columns_casing_detection(self,data):
        """Produces a list of text columns that contain casing issues in the text. Then it performs casing resolution."""
        
        print("Casing Check Please Stop!")
        text_columns=list(data.select_dtypes(exclude=[np.number]).columns)
        categorical_casing_list=[]
        data_lower=deepcopy(data)
        for col in text_columns:
            data_lower[col]=data[col].astype(str).str.lower()
            if(len(data[col].unique())==len(data_lower[col].unique())):
                pass
            else:
                categorical_casing_list.append(col)
        print("List which was revised to lower case due to casing issues : "+str(categorical_casing_list))
        data=self.casing_resolve(data,categorical_casing_list)
        return data

    def NumericalDataBinning(self,data,numeric_cols):    ### What sort of Normalization
        """ Automated Binning of Data based on  Numerical Categorization """
        
        print("Checking if Data Binning is required")
        test_df=data[numeric_cols]
        test_df.dropna(inplace=True)
        binning_columns=[]
        frame_unique=pd.DataFrame([test_df.nunique()])
        for (col) in frame_unique.columns:
                if(frame_unique[col][0]<(self.CatgclDetct_MaxUnqValsThreshold)*data.shape[0]):
                    binning_columns.append(col)
        print("Here is the List of Binning Eligible Columns : "+str(binning_columns))
    
        return True
    
    def imbalanced_targetdata_detection(self,data,target):                   ### Skewness before outlier removal
        """Imbalanced Data Detection based on threshold that is difference
        in max count and min count out of all classes is greater than 5 percent"""
        
        print("Checking Dataset For Balance in Data labels")
        col_name='# of occurances'    
        df=(data.groupby(target).size().reset_index(name=col_name))
        res=df[col_name].max()-df[col_name].min();
        if(res>(self.Imbalncd_MaxMinDiffThreshold)*df.shape[0]):
            print("Imbalanced Class Problem Found for Target Column : "+str(target))
        else:
            print("Balanced Class Found for Target Column : "+str(target))
    
    def check_gauss_normalization(self,data,numeric_cols):    ### What sort of Normalization
        """Checks if data is within the conditions for gauss normalization
        then if that is the case, run gauss normalization"""
        
        print("Checking if Gauss Normalization is required")
        test_df=data[numeric_cols]
        test_df.dropna(inplace=True)
        gauss_columns=[]
        mean = test_df.describe().iloc[1, :]
        std = test_df.describe().iloc[2, :]
        for (col, mean, std) in zip(numeric_cols, mean, std):
            val_df=deepcopy(test_df[test_df[col]>mean-std])
            if((self.ChkGausNorm_MinDataStdThreshold)*test_df[col].shape[0]<(val_df[val_df[col]<mean+std].shape[0]) and (self.ChkGausNorm_MaxDataStdThreshold)*test_df[col].shape[0]>(val_df[val_df[col]<mean+std].shape[0])):
                pass
            else:
                gauss_columns.append(col)
        print("Here is the list of Gauss normalized columns : "+str(gauss_columns))
        res=self.perform_gauss_normalization(data,gauss_columns)
    
        return res


class process_flow_preprocessing(PreProcessingExternalUtilites):
    """ Flow class for Pre Processing Class. This class's
    whole purpose is to easily tweak the flow of preprocessing
    class rather than hard coding it and i am loving writing this.
    Atleast, for the day even when i am sick thanks for reading it!."""

    def __init__(self,file_name,target_column,prediction_type,check_only_flag=False,missing_headers=False,RemvRedCol_StatsInvariant = 0.9,RemvRedCol_NullCount=0.1,
        RemvRedCol_MinColumnLimit=10,RemvRedCol_MinRows=10000,ExtrctNumCol_MinRowThreshold=0.01,RplcMisngData_RowDelThreshold=0.01,RplcMisngDta_MinRows=1000,
        CatgclDetct_MaxUnqValsThreshold=0.005,Outlier_StdDevMeanMultiFactor=3,Imbalncd_MaxMinDiffThreshold=0.05,ChkGausNorm_MinDataStdThreshold=0.66,
        ChkGausNorm_MaxDataStdThreshold=0.70,ForcedImpute=False,RemvRedCol_MaxLength=100,RemvRedCol_IgnoreUniqueFlag=False,NumrcDtaBins_MaxUnqValsThreshld=0.005,**kwargs):
        Initialization.__init__(self,file_name,target_column,prediction_type,check_only_flag=False,missing_headers=False,RemvRedCol_StatsInvariant = 0.9,RemvRedCol_NullCount=0.1,
        RemvRedCol_MinColumnLimit=10,RemvRedCol_MinRows=10000,ExtrctNumCol_MinRowThreshold=0.01,RplcMisngData_RowDelThreshold=0.01,RplcMisngDta_MinRows=1000,
        CatgclDetct_MaxUnqValsThreshold=0.005,Outlier_StdDevMeanMultiFactor=3,Imbalncd_MaxMinDiffThreshold=0.05,ChkGausNorm_MinDataStdThreshold=0.66,
        ChkGausNorm_MaxDataStdThreshold=0.70,ForcedImpute=False,RemvRedCol_MaxLength=100,RemvRedCol_IgnoreUniqueFlag=False,NumrcDtaBins_MaxUnqValsThreshld=0.005,**kwargs)

    def preprocessingflow(self):
        """Diferrent type of predictions governs the preprocessing it has to be done with"""
        if(self.prediction_type==1):
            data=self.load_csv()
            data=self.categorical_columns_casing_detection(data)
            data=self.remove_redundant_columns(data)
            if(len(self.numeric_columns_list)>0):
                data=self.extract_numeric_columns(data,self.numeric_columns_list)
            else:
                data=self.extract_numeric_columns(data)
            self.outlier_list(data,self.list_numeric)
            data=self.replace_missing_data(data)
            data=self.drop_duplicated_data(data)
            self.NumericalDataBinning(data,self.numeric_columns_list)
            self.categorical_columns_list=self.categorical_variables(data)
            data=self.check_gauss_normalization(data,self.list_numeric)
            self.imbalanced_targetdata_detection(data,self.target)
            return data
        elif(self.prediction_type==2):
            data=self.load_csv()
            self.NumericalDataBinning(data,self.numeric_columns_list)
        elif(self.prediction_type==3):
            pass
        return data

