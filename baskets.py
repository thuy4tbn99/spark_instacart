from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
conf = SparkConf().set("spark.cores.max", "32") \
    .set("spark.driver.memory", "50g") \
    .set("spark.executor.memory", "50g") \
    .set("spark.executor.memory_overhead", "50g") \
    .set("spark.driver.maxResultsSize", "16g")\
    .set("spark.executor.heartbeatInterval", "30s")
sc = SparkContext(conf=conf).getOrCreate();
spark = SparkSession(sc)

# read baskets_prior
baskets = spark.read.csv('./data/baskets_prior.csv',header=True, inferSchema=True)
baskets.createOrReplaceTempView("baskets")
baskets.show(5)
print(baskets.count())

# transform string to list
import pyspark.sql.functions as F
df2 = baskets.withColumn(
    "new_items",
    F.from_json(F.col("items"), "array<string>")
)
df2 = df2.drop('items')
df2.show(5)

from pyspark.ml.fpm import FPGrowth
import time

start = time.time()
local_time = time.ctime(start)
print("Start time:", local_time)
fpGrowth = FPGrowth(itemsCol="new_items", minSupport=0.000015, minConfidence=0.7)
model = fpGrowth.fit(df2)
model.associationRules.show()
print(model.associationRules.count())

assoRules = model.associationRules
freqItems = model.freqItemsets
end = time.time()
print("run time: ", (end-start)/60)
local_time = time.ctime(end)
print("End time:", local_time)

# freq to pandas
freq_pd =freqItems.toPandas()
freq_pd = freq_pd.sort_values('freq', ascending=False)
print(freq_pd.head(5))
freq_pd.to_csv('./data/freqItems_baskets3M.csv', index=False)

# save rules
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'

array_to_string_udf = udf(array_to_string, StringType())

assoRules = assoRules.withColumn('antecedent', array_to_string_udf(assoRules["antecedent"]))
assoRules = assoRules.withColumn('consequent', array_to_string_udf(assoRules["consequent"]))
print('after convert string to save: ', assoRules.show(7))
assoRules.coalesce(1).write.csv('./data/assoRules_baskets3M_50_70%')




