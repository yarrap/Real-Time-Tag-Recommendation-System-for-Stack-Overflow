## importing libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import substring, udf, concat_ws, explode
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, OneHotEncoder, StringIndexer, Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SparkSession 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time


## calculating the start time 
start_time = time.time()

# Creating a Spark session with configurations
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[9]")\
    .config("spark.driver.memory","16G")\
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4")\
    .getOrCreate()

# Setting log levels and configurations for Spark
spark.sparkContext.setLogLevel("INFO")
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100m")  # Set the threshold to 100MB

# Reading the preprocessed data file
data = spark.read.csv('data.csv', header=True)

# Creating StringIndexer for 'Tag' column
indexer = StringIndexer(inputCol="Tag", outputCol="tags_indexed")

# Adding a row index to join with final_df
data = data.withColumn("index", F.monotonically_increasing_id())

# Truncating the 'Body' and 'Title' columns to 300 characters
data = data.withColumn('Body', substring(data['Body'], 1, 300))
data = data.withColumn('Title', substring(data['Title'], 1, 300))
print("shape of data: ",data.count())

# Splitting 'Tags' column into an array
split_data = data.withColumn("Tags_array", split("Tags", ","))

# Exploding the array of tags to separate rows
exploded = split_data.withColumn("exploded_labels", explode("Tags_array"))
exploded.show(truncate=False)

# Creating StringIndexer for 'exploded_labels' column
indexer = StringIndexer(inputCol="exploded_labels", outputCol="indexed_label")
indexed = indexer.fit(exploded).transform(exploded)
df  = indexed

# Concatenating 'Body' and 'Title' columns
delimiter = " "
df = df.withColumn("Body", concat_ws(delimiter, "Body")).withColumn("Title", concat_ws(delimiter, "Title"))

# Tokenizing 'Body' and 'Title' columns
tokenizer = Tokenizer(inputCol="Body", outputCol="words_body")
df = tokenizer.transform(df)
tokenizer = Tokenizer(inputCol="Title", outputCol="words_title")
df = tokenizer.transform(df)

# Applying Term Frequency (TF)
hashingTF_title = HashingTF(inputCol="words_title", outputCol="raw_title", numFeatures=1000)
hashingTF_body = HashingTF(inputCol="words_body", outputCol="raw_body", numFeatures=1000)
tf_title = hashingTF_title.transform(df)
tf_body = hashingTF_body.transform(tf_title)

# Applying Inverse Document Frequency (IDF)
idf_title = IDF(inputCol="raw_title", outputCol="tfidf_title")
idf_model_title = idf_title.fit(tf_body)
tfidf_title = idf_model_title.transform(tf_body)

idf_body = IDF(inputCol="raw_body", outputCol="tfidf_body")
idf_model_body = idf_body.fit(tfidf_title)
tfidf_body = idf_model_body.transform(tfidf_title)

# Concatenating TF-IDF vectors
assembler = VectorAssembler(
    inputCols=["tfidf_title", "tfidf_body"],
    outputCol="features")
final_df = assembler.transform(tfidf_body)


data = final_df
data = data.withColumnRenamed("indexed_label","label")

# Splitting the data into training and testing sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

# Training a Random Forest classifier
classifier = RandomForestClassifier(labelCol="label", featuresCol="features")
model = classifier.fit(train_data)

# Evaluating the model on training and testing data
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
predictions_train = model.transform(train_data)
predictions_test = model.transform(test_data)

# Calculating the accuracy of the model
accuracy_test = evaluator.evaluate(predictions_test)
accuracy_train = evaluator.evaluate(predictions_train)

# Calculating runtime
from itertools import combinations, col

# getting the list of tags and their label encoding
columns = ["label","Tags"]
all_column_combinations = []
for r in range(1, len(columns) + 1):
    column_combinations = combinations(columns, r)
    all_column_combinations.extend(list(column_combinations))

distinct_combinations = []
for comb in all_column_combinations:
    selected_cols = [col(column) for column in comb]
    selected_df = train_data.select(selected_cols)
    distinct_combinations.append(selected_df)

# selecting a part of the output to show in the api
# below are just steps to show a sample data
output = predictions_test.select("Title","Body","prediction","Tags")
output=output.withColumnRenamed("Tags","Tags_x")
output_df = output.join(distinct_combinations[2],distinct_combinations[2]["Tags"]==output["Tags_x"],"inner")
filtered_df = output_df.filter(col('Tags_x').contains('android'))
filtered_df = filtered_df.limit(10)
filtered_df2 = output_df.filter((col('Tags').contains('android')))
filtered_df2 = filtered_df2.limit(10)
appended_df = filtered_df.union(filtered_df2)
cols = ["Tags_x"]
appended_df = appended_df.drop(*cols)
# appended_df.show()  

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")



