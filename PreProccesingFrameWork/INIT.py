import PPS as P
import time


start_time = time.time()
ordinal_col_list_order={'online_order':{'Yes':1,'No':0},'book_table':{'Yes':1,'No':0}}
obj=P.process_flow_preprocessing('zomato.csv','votes',1,categorical_columns_list=['listed_in(city)'],numeric_columns_list=['votes','approx_cost(for two people)'],ordinal_col_list_order=ordinal_col_list_order)
(obj.preprocessingflow())
print("--- %s seconds ---" % (time.time() - start_time))