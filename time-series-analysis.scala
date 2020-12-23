// Databricks notebook source
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types
import org.apache.spark.sql.Column
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import spark.implicits._
import scala.util.Random
import org.apache.spark.sql.expressions.Window
import java.util.Date
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.time.{ZoneId, ZonedDateTime}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import scala.collection.mutable.ListBuffer

// COMMAND ----------

// MAGIC %md # Import Data

// COMMAND ----------

// Import tables
val features_data = spark.table("features")
val sales_data = spark.table("sales")
val stores_data = spark.table("stores")

// Check table content
features_data.show(10)
sales_data.show(10)
stores_data.show(10)

// Check sizes
println((features_data.count(), features_data.columns.size))
println((sales_data.count(), sales_data.columns.size))
println((stores_data.count(), stores_data.columns.size))

// COMMAND ----------

// MAGIC %md # Data Preprocessing and Cleaning

// COMMAND ----------

//Merging three tables
val df_sales = sales_data.join(features_data, Seq("Store", "Date", "IsHoliday"), "left").
  join(stores_data, Seq("Store"), "left").
  withColumn("Date", to_date($"Date", "dd/MM/yyyy"))

val weekWindow = Window.orderBy("Date")

// aggreate all store sales, and calculate the average of other features
val df = df_sales.
  withColumn("IsHoliday", when($"IsHoliday" === "TRUE", 1).otherwise(0)).
  groupBy("Date").
  agg(round(sum("Weekly_Sales")) as "total_sales",
     avg("Temperature") as "temp",
     avg("Fuel_Price") as "fuel_price",
     avg("Unemployment") as "unemployment",
     avg("CPI") as "cpi",
     avg("IsHoliday") as "IsHoliday").
     withColumn("month", month(to_timestamp($"Date", "yyyy-MM-dd"))).
     withColumn("day", dayofmonth(to_timestamp($"Date", "yyyy-MM-dd"))).
     withColumn("label", lead("total_sales", 3, 0).over(weekWindow)).
  orderBy("Date")

display(df)

// COMMAND ----------

display(df)

// COMMAND ----------

// MAGIC %md The above plot shows total sales across seasonality during the holiday months (from Nov to Dec). We could detrend the total sales and only predict the difference between this year and last year, because it can capture the nuance of price changes better, and render a more robust prediction power.

// COMMAND ----------

// MAGIC %md # Feature Engineering

// COMMAND ----------

// MAGIC %md ## Time-Series Seasonality Differencing and Other Features for All Prediction Horizons

// COMMAND ----------

// day is the Friday of each week, and it can be treated as a week indicator
val w = Window.orderBy(col("Date"))
val weekly_win = Window.orderBy(col("Date")).rowsBetween(-1, 0)
val biweekly_win = Window.orderBy(col("Date")).rowsBetween(-2, 0)
val triweekly_win = Window.orderBy(col("Date")).rowsBetween(-3, 0)
val monthly_win = Window.orderBy(col("Date")).rowsBetween(-4, 0)
val week5_win = Window.orderBy(col("Date")).rowsBetween(-5, 0)
val week6_win = Window.orderBy(col("Date")).rowsBetween(-6, 0)
val week7_win = Window.orderBy(col("Date")).rowsBetween(-7, 0)

// the following code detrended the total_sales, and genreate more time-sereis related features
val df_1week1 = df
  .withColumn("lag_1year", lag("total_sales", 48, 0).over(w))
  .where($"lag_1year">0)
  .withColumn("diff_sales", abs($"total_sales" - $"lag_1year"))
  .withColumn("lag_1week", lag("diff_sales", 1, 0).over(w))
  .withColumn("lag_2weeks", lag("diff_sales", 2, 0).over(w))
  .withColumn("lag_3weeks", lag("diff_sales", 3, 0).over(w))
  .withColumn("lag_4weeks", lag("diff_sales", 4, 0).over(w))
  .where($"lag_4weeks">0)
  .withColumn("ma_1week", avg("lag_1week").over(weekly_win))
  .withColumn("ma_2weeks", avg("lag_1week").over(biweekly_win))
  .withColumn("ma_3weeks", avg("lag_1week").over(triweekly_win))
  .withColumn("ma_4weeks", avg("lag_1week").over(monthly_win))
  .withColumn("std_1week", stddev("lag_1week").over(weekly_win))
  .withColumn("std_2weeks", stddev("lag_1week").over(biweekly_win))
  .withColumn("std_3weeks", stddev("lag_1week").over(triweekly_win))
  .withColumn("std_4weeks", stddev("lag_1week").over(monthly_win))
  .where($"std_4weeks">0)
  .withColumn("label", lead("diff_sales", 1, 0).over(w))
  .na.drop
  .drop("total_sales")
  .orderBy("Date")

val season_diff_1week = df_1week1.select("lag_1year", "Date")

val df_1week = df_1week1.drop("lag_1year")


// Prediction horizon: 2 weeks
val df_2weeks1 = df
  .withColumn("lag_1year", lag("total_sales", 48, 0).over(w))
  .where($"lag_1year">0)
  .withColumn("diff_sales", abs($"total_sales" - $"lag_1year"))
  .withColumn("lag_2weeks", lag("diff_sales", 2, 0).over(w))
  .withColumn("lag_3weeks", lag("diff_sales", 3, 0).over(w))
  .withColumn("lag_4weeks", lag("diff_sales", 4, 0).over(w))
  .withColumn("lag_5weeks", lag("diff_sales", 5, 0).over(w))
  .where($"lag_5weeks">0)
  .withColumn("ma_2weeks", avg("lag_2weeks").over(biweekly_win))
  .withColumn("ma_3weeks", avg("lag_2weeks").over(triweekly_win))
  .withColumn("ma_4weeks", avg("lag_2weeks").over(monthly_win))
  .withColumn("ma_5weeks", avg("lag_2weeks").over(week5_win))
  .withColumn("std_2weeks", stddev("lag_2weeks").over(biweekly_win))
  .withColumn("std_3weeks", stddev("lag_2weeks").over(triweekly_win))
  .withColumn("std_4weeks", stddev("lag_2weeks").over(monthly_win))
  .withColumn("std_5weeks", stddev("lag_2weeks").over(week5_win))
  .where($"std_5weeks">0)
  .withColumn("label", lead("diff_sales", 2, 0).over(w))
  .na.drop
  .drop("total_sales")
  .orderBy("Date")

val season_diff_2weeks = df_2weeks1.select("lag_1year", "Date")

val df_2weeks = df_2weeks1.drop("lag_1year")


// Prediction horizon: 3 weeks
val df_3weeks1 = df
  .withColumn("lag_1year", lag("total_sales", 48, 0).over(w))
  .where($"lag_1year">0)
  .withColumn("diff_sales", abs($"total_sales" - $"lag_1year"))
  .withColumn("lag_3weeks", lag("diff_sales", 3, 0).over(w))
  .withColumn("lag_4weeks", lag("diff_sales", 4, 0).over(w))
  .withColumn("lag_5weeks", lag("diff_sales", 5, 0).over(w))
  .withColumn("lag_6weeks", lag("diff_sales", 6, 0).over(w))
  .where($"lag_6weeks">0)
  .withColumn("ma_3weeks", avg("lag_3weeks").over(triweekly_win))
  .withColumn("ma_4weeks", avg("lag_3weeks").over(monthly_win))
  .withColumn("ma_5weeks", avg("lag_3weeks").over(week5_win))
  .withColumn("ma_6weeks", avg("lag_3weeks").over(week6_win))
  .withColumn("std_3weeks", stddev("lag_3weeks").over(triweekly_win))
  .withColumn("std_4weeks", stddev("lag_3weeks").over(monthly_win))
  .withColumn("std_5weeks", stddev("lag_3weeks").over(week5_win))
  .withColumn("std_6weeks", stddev("lag_3weeks").over(week6_win))
  .where($"std_6weeks">0)
  .withColumn("label", lead("diff_sales", 3, 0).over(w))
  .na.drop
  .drop("total_sales")
  .orderBy("Date")

val season_diff_3weeks = df_3weeks1.select("lag_1year", "Date")

val df_3weeks = df_3weeks1.drop("lag_1year")

// COMMAND ----------

display(df_1week)

// COMMAND ----------

// MAGIC %md ## Time-Series Train-Test Split

// COMMAND ----------

// Function for timeseries train-test split
def ts_split(split_pct: Double, df: DataFrame) : (DataFrame, DataFrame) =  { 
  /**
    * Returns two dataframe based on the given split percentage
    * The two dataframe can be used as train and test dataset
    *
    */
  val breakpoint = math.ceil(df.count() * split_pct)
  
  val df_id = df.withColumn("id", monotonically_increasing_id)
  val first = df_id.where($"id" < breakpoint).drop("id")
  val second = df_id.where($"id" >= breakpoint).drop("id")
  
  return (first, second)
}

// COMMAND ----------

// Split train-test sets from dataframes (80/20)
val split_pct = 0.8

// COMMAND ----------

val (train_set_1week_a, test_set_1week) = ts_split(split_pct, df_1week)
val (train_set_2weeks_a, test_set_2weeks) = ts_split(split_pct, df_2weeks)
val (train_set_3weeks_a, test_set_3weeks) = ts_split(split_pct, df_3weeks)

println(s"train_set_1week_a.count: ${train_set_1week_a.count}")
println(s"train_set_2weeks_a.count: ${train_set_2weeks_a.count}")
println(s"train_set_3weeks_a.count: ${train_set_3weeks_a.count}")

// COMMAND ----------

// Create gap between train and test sets to ensure no data leakage
val train_set_1week = train_set_1week_a.limit((train_set_1week_a.count - 5).toInt)
val train_set_2weeks = train_set_2weeks_a.limit((train_set_2weeks_a.count - 7).toInt)
val train_set_3weeks = train_set_3weeks_a.limit((train_set_3weeks_a.count - 9).toInt)

// Split the seasonal differences using the same ratio
// Only use "test_diff" later for reconstruction
val (train_diff_1week, test_diff_1week) = ts_split(split_pct, season_diff_1week)
val (train_diff_2weeks, test_diff_2weeks) = ts_split(split_pct, season_diff_2weeks)
val (train_diff_3weeks, test_diff_3weeks) = ts_split(split_pct, season_diff_3weeks)

// COMMAND ----------

// MAGIC %md ## Feature Importance

// COMMAND ----------

def featureImportance(train: DataFrame): Array[String] = {
  
  val feature_cols = train.drop("Date").columns
  
  // train a new random forest model
  val rf = new RandomForestRegressor()
  
  val vectorAssembler = new VectorAssembler()
          .setInputCols(feature_cols)
          .setOutputCol("features")
  
  val train_assembled = vectorAssembler.transform(train)
  
  // Train model
  val model = rf.fit(train_assembled)
    
  // extract feature importance from rf model
  val importances = model
    .asInstanceOf[RandomForestRegressionModel]
    .featureImportances

  val features = train_assembled.columns.filterNot(Set("Date", "label", "features"))

  val printImportance = importances.toArray zip features
  
  println("------------------------------------------")
  printImportance.sortBy(-_._1).foreach(x => println(x._2 + " -> " + x._1))
  println("------------------------------------------")
  
  val result_collector = new ListBuffer[String]
  
  printImportance.foreach(x => {
    if(x._1.toDouble > 0.03) {
      result_collector += x._2
    }
  })

  return result_collector.toArray
  
}

// COMMAND ----------

// MAGIC %md ## Define features for all prediction horizon

// COMMAND ----------

val features_1week = featureImportance(train_set_1week)
val features_2weeks = featureImportance(train_set_2weeks)
val features_3weeks = featureImportance(train_set_3weeks)

// COMMAND ----------

// MAGIC %md # Prediction

// COMMAND ----------

// MAGIC %md ## Main Classes for Machine Learning & Stats Models Prediction

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

class TrainModel(train_set: DataFrame, test_set: DataFrame, feature_cols: Array[String], evaluation_metrics: String, params: Map[String, String]) {
  
  val vectorAssembler = new VectorAssembler()
          .setInputCols(feature_cols)
          .setOutputCol("features")
  
  val train_assembled = vectorAssembler.transform(train_set)
  val test_assembled = vectorAssembler.transform(test_set)
  
  def RandomForestPredict(): DataFrame = {
    val model = new RandomForestRegressor()
        .setNumTrees(params("NumTrees").toInt)
        .setMaxBins(params("MaxBins").toInt)
    return model.fit(train_assembled).transform(test_assembled).select("prediction", "label")
  }
  
  def LinearRegressionPredict(): DataFrame = {
    val model = new LinearRegression()
      .setRegParam(params("RegParam").toFloat)
      .setElasticNetParam(params("ElasticNetParam").toFloat)
    return model.fit(train_assembled).transform(test_assembled).select("prediction", "label")
  }
  
  def DecisionTreePredict(): DataFrame = {
    val model = new DecisionTreeRegressor()
      .setMaxDepth(params("MaxDepth").toInt)
      .setMaxBins(params("MaxBins").toInt)
    return model.fit(train_assembled).transform(test_assembled).select("prediction", "label")
  }
  
  def GBTPredict(): DataFrame = {
    val model = new GBTRegressor()
      .setMaxIter(params("MaxIter").toInt)
      .setMaxBins(params("MaxBins").toInt)
      .setMaxDepth(params("MaxDepth").toInt)
    return model.fit(train_assembled).transform(test_assembled).select("prediction", "label")
  }
  
  def evaluteModel(predictions: DataFrame): Double = {    
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setMetricName(evaluation_metrics)
    return evaluator.evaluate(predictions)
  }
  
  def retrend(seasonality: DataFrame, predictions: DataFrame): DataFrame = {
    val diff = seasonality.withColumn("id", monotonically_increasing_id())
    val pred = predictions.withColumn("id", monotonically_increasing_id())
    val merged = pred.join(diff, diff.col("id") === pred.col("id"), "left_outer").drop("id")
    return merged.withColumn("label", $"label"+$"lag_1year")
      .withColumn("prediction", $"prediction"+$"lag_1year")
  }
  
  def smape(predictions: DataFrame, trim_last_n_rows: Int = 0): String = {

    var set_limit = predictions.count().toInt - trim_last_n_rows

    // Compute SMAPE
    val df = predictions.
      limit(set_limit).
      withColumn("diff_abs", abs($"prediction" - $"label")).
      withColumn("demo", (abs($"prediction") + abs($"label")) / 2).
      withColumn("division", $"diff_abs" / $"demo").
      agg(round(sum($"division") / predictions.count() * 100, 4)).
      collect()
    
    return df(0)(0).toString
  }
}

// COMMAND ----------

// MAGIC %md ## K-fold Rolling Cross-Validation for Time-Series

// COMMAND ----------

class RollingCV(modelName: String, param_grid: Map[String, Seq[String]]) {

  // transform arrays into lists with values paired with map key
  val pairedWithKey = param_grid.map { case (k,v) => v.map(i => k -> i).toList }
  val accumulator = pairedWithKey.head.map(x => Vector(x))
  val param_comb = pairedWithKey.tail.foldLeft(accumulator)( (acc, elem) => 
    for { x <- acc; y <- elem } yield x :+ y 
  )
  
                                                                    
  def initiate(train_set: DataFrame, features: Array[String]): Map[String, String] = {
    
    var result_collector : List[(Double,Map[String, String])] = List()

    for (comb <- param_comb) {

      val params = comb.groupBy(_._1).map { case (k,v) => (k,v.map(_._2).head)}
      
      /*** rolling section ***/
      val total_rows = train_set.count().toInt
      val initial_train_obs = total_rows % 10
      val fold_num = 2
      val shift = (total_rows - initial_train_obs) / fold_num 

      var total_rmse = 0.0
      
      for (i <- 1 to fold_num) {
    
        // define rolling frame
        var total_select = initial_train_obs + shift * i
        var train_pct = initial_train_obs.toFloat / total_select

        // train test split  
        val (train, test) = ts_split(train_pct, train_set.limit(total_select.toInt))

        /*******************REPLACE TO OTHER MODEL START *********************/
        // Initiate model object
        val model = new TrainModel(train, test, features, "rmse", params)
        
        var predictions: DataFrame = spark.emptyDataFrame
        
        modelName match {
          case "lr" => predictions = model.LinearRegressionPredict()
          case "rf" => predictions = model.RandomForestPredict()
          case "dt" => predictions = model.DecisionTreePredict()
          case "gbt" => predictions = model.GBTPredict()
        }
        
        val rmse = model.evaluteModel(predictions)
        /*******************REPLACE TO OTHER MODEL END *********************/

        total_rmse += rmse

        println(s"Hyper params: $params; rolling times: $i ($total_select/$total_rows); rmse: $rmse")
      }
      /*** rolling section ***/
      
      val avg_rmse = total_rmse / fold_num
      
      result_collector = result_collector :+ (avg_rmse, params)

    }
    
    val result_ordered = result_collector.sortBy(_._1)
    return result_ordered(0)._2
  }

}

// COMMAND ----------

// MAGIC %md ## Simple Moving Average Prediction Fuction

// COMMAND ----------

def sma_predict(df: DataFrame, w: org.apache.spark.sql.expressions.WindowSpec): String = {
  
  // Splitting the original data to get test set before calculating SMA
  val (train, test) = ts_split(split_pct, df)
  
  val ma = test
    .withColumn("ma_week", avg("total_sales").over(w))
    .drop("Date")

  val ma_pred = ma.
    withColumn("diff_abs", abs($"ma_week" - $"total_sales")).
    withColumn("demo", (abs($"ma_week") + abs($"total_sales")) / 2).
    withColumn("division", $"diff_abs" / $"demo").
    agg(round(sum($"division") / ma.count() * 100, 4)).collect()

  return ma_pred(0)(0).toString
}

// COMMAND ----------

// MAGIC %md ## Define Hyper-Parameter Map for ML Models

// COMMAND ----------

val lr_param_grid = Map(
  "RegParam" -> Seq("0","0.1","0.5"),
  "ElasticNetParam" -> Seq("0","0.1","0.5")
)

val rf_param_grid = Map(
  "NumTrees" -> Seq("5","10","15"),
  "MaxBins" -> Seq("28","30","32")
)

val dt_param_grid = Map(
  "MaxDepth" -> Seq("5","7","10"),
  "MaxBins" -> Seq("20","25","30")
)

val gbt_param_grid = Map(
  "MaxIter" -> Seq("5","7","10"),
  "MaxBins" -> Seq("23","25","27"),
  "MaxDepth" -> Seq("6","8","10")
)

// COMMAND ----------

// MAGIC %md ## Define Windows for SMA

// COMMAND ----------

val biweekly_w = Window.orderBy(col("day")).rowsBetween(-2, -1)
val triweekly_w = Window.orderBy(col("day")).rowsBetween(-3, -1)
val monthly_w = Window.orderBy(col("day")).rowsBetween(-4, -1)

// COMMAND ----------

// MAGIC %md ## One Week Prediction

// COMMAND ----------

display(train_set_1week)

// COMMAND ----------

display(test_set_1week)

// COMMAND ----------

// MAGIC %md ### Random Forest

// COMMAND ----------

val rf_cv_1week = new RollingCV("rf", rf_param_grid)
val rf_best_params_1week = rf_cv_1week.initiate(train_set_1week, features_1week)  

println(rf_best_params_1week)

val rf_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", rf_best_params_1week)

val rf_best_preds_1week = rf_best_model_1week.RandomForestPredict()

val rf_final_preds_1week = rf_best_model_1week.retrend(test_diff_1week, rf_best_preds_1week)

// COMMAND ----------

// val rf_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", Map("NumTrees" -> "5", "MaxBins" -> "28"))

// val rf_best_preds_1week = rf_best_model_1week.RandomForestPredict()

// val rf_final_preds_1week = rf_best_model_1week.retrend(test_diff_1week, rf_best_preds_1week)

// COMMAND ----------

display(rf_final_preds_1week)

// COMMAND ----------

val rf_1week_smape = rf_best_model_1week.smape(rf_final_preds_1week, 1)
println(rf_1week_smape)

// COMMAND ----------

// MAGIC %md ### Linear Regression

// COMMAND ----------

val lr_cv_1week = new RollingCV("lr", lr_param_grid)
val lr_best_params_1week = lr_cv_1week.initiate(train_set_1week, features_1week)  

println(lr_best_params_1week)

val lr_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", lr_best_params_1week)

val lr_best_preds_1week = lr_best_model_1week.LinearRegressionPredict()

val lr_final_preds_1week = lr_best_model_1week.retrend(test_diff_1week, lr_best_preds_1week)

// COMMAND ----------

val lr_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", Map("RegParam" -> "0.5", "ElasticNetParam" -> "0"))

val lr_best_preds_1week = lr_best_model_1week.LinearRegressionPredict()

val lr_final_preds_1week = lr_best_model_1week.retrend(test_diff_1week, lr_best_preds_1week)

// COMMAND ----------

display(lr_final_preds_1week)

// COMMAND ----------

val lr_1week_smape = lr_best_model_1week.smape(lr_final_preds_1week, 1)
println(lr_1week_smape)

// COMMAND ----------

// MAGIC %md ### Decision Tree

// COMMAND ----------

val dt_cv_1week = new RollingCV("dt", dt_param_grid)
val dt_best_params_1week = dt_cv_1week.initiate(train_set_1week, features_1week)  

println(dt_best_params_1week)

val dt_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", dt_best_params_1week)

val dt_best_preds_1week = dt_best_model_1week.DecisionTreePredict()

val dt_final_preds_1week = dt_best_model_1week.retrend(test_diff_1week, dt_best_preds_1week)

// COMMAND ----------

val dt_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", Map("MaxDepth" -> "5", "MaxBins" -> "20"))

val dt_best_preds_1week = dt_best_model_1week.DecisionTreePredict()

val dt_final_preds_1week = dt_best_model_1week.retrend(test_diff_1week, dt_best_preds_1week)

// COMMAND ----------

display(dt_final_preds_1week)

// COMMAND ----------

val dt_1week_smape = dt_best_model_1week.smape(dt_final_preds_1week, 1)
println(dt_1week_smape)

// COMMAND ----------

// MAGIC %md ### Gradient Boosted Tree

// COMMAND ----------

val gbt_cv_1week = new RollingCV("gbt", gbt_param_grid)
val gbt_best_params_1week = gbt_cv_1week.initiate(train_set_1week, features_1week) 

println(gbt_best_params_1week)

val gbt_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", gbt_best_params_1week)

val gbt_best_preds_1week = gbt_best_model_1week.GBTPredict()

val gbt_final_preds_1week = gbt_best_model_1week.retrend(test_diff_1week, gbt_best_preds_1week)

// COMMAND ----------

val gbt_best_model_1week = new TrainModel(train_set_1week, test_set_1week, features_1week, "rmse", Map("MaxIter" -> "5", "MaxDepth" -> "6", "MaxBins" -> "23"))

val gbt_best_preds_1week = gbt_best_model_1week.GBTPredict()

val gbt_final_preds_1week = gbt_best_model_1week.retrend(test_diff_1week, gbt_best_preds_1week)

// COMMAND ----------

display(gbt_final_preds_1week)

// COMMAND ----------

val gbt_1week_smape = gbt_best_model_1week.smape(gbt_final_preds_1week, 1)
println(gbt_1week_smape)

// COMMAND ----------

// MAGIC %md ## Two Weeks Prediction

// COMMAND ----------

// MAGIC %md ### Simple Moving Average (SMA)

// COMMAND ----------

val sma_2weeks_smape = sma_predict(df, biweekly_w)
println(sma_2weeks_smape)

// COMMAND ----------

// MAGIC %md ### Random Forest

// COMMAND ----------

val rf_cv_2weeks = new RollingCV("rf", rf_param_grid)
val rf_best_params_2weeks = rf_cv_2weeks.initiate(train_set_2weeks, features_2weeks)  

println(rf_best_params_2weeks)

val rf_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", rf_best_params_2weeks)

val rf_best_preds_2weeks = rf_best_model_2weeks.RandomForestPredict()

val rf_final_preds_2weeks = rf_best_model_2weeks.retrend(test_diff_2weeks, rf_best_preds_2weeks)

// COMMAND ----------

val rf_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", Map("NumTrees" -> "15", "MaxBins" -> "28"))

val rf_best_preds_2weeks = rf_best_model_2weeks.RandomForestPredict()

val rf_final_preds_2weeks = rf_best_model_2weeks.retrend(test_diff_2weeks, rf_best_preds_2weeks)

// COMMAND ----------

display(rf_final_preds_2weeks)

// COMMAND ----------

val rf_2weeks_smape = rf_best_model_2weeks.smape(rf_final_preds_2weeks, 2)
println(rf_2weeks_smape)

// COMMAND ----------

// MAGIC %md ### Linear Regression

// COMMAND ----------

val lr_cv_2weeks = new RollingCV("lr", lr_param_grid)
val lr_best_params_2weeks = lr_cv_2weeks.initiate(train_set_2weeks, features_2weeks)  

println(lr_best_params_2weeks)

val lr_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", lr_best_params_2weeks)

val lr_best_preds_2weeks = lr_best_model_2weeks.LinearRegressionPredict()

val lr_final_preds_2weeks = lr_best_model_2weeks.retrend(test_diff_2weeks, lr_best_preds_2weeks)

// COMMAND ----------

val lr_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", Map("RegParam" -> "0.5", "ElasticNetParam" -> "0"))

val lr_best_preds_2weeks = lr_best_model_2weeks.LinearRegressionPredict()

val lr_final_preds_2weeks = lr_best_model_2weeks.retrend(test_diff_2weeks, lr_best_preds_2weeks)

// COMMAND ----------

display(lr_final_preds_2weeks)

// COMMAND ----------

val lr_2weeks_smape = lr_best_model_2weeks.smape(lr_final_preds_2weeks, 2)
println(lr_2weeks_smape)

// COMMAND ----------

// MAGIC %md ### Decision Tree

// COMMAND ----------

val dt_cv_2weeks = new RollingCV("dt", dt_param_grid)
val dt_best_params_2weeks = dt_cv_2weeks.initiate(train_set_2weeks, features_2weeks)  

println(dt_best_params_2weeks)

val dt_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", dt_best_params_2weeks)

val dt_best_preds_2weeks = dt_best_model_2weeks.DecisionTreePredict()

val dt_final_preds_2weeks = dt_best_model_2weeks.retrend(test_diff_2weeks, dt_best_preds_2weeks)

// COMMAND ----------

val dt_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", Map("MaxDepth" -> "5", "MaxBins" -> "20"))

val dt_best_preds_2weeks = dt_best_model_2weeks.DecisionTreePredict()

val dt_final_preds_2weeks = dt_best_model_2weeks.retrend(test_diff_2weeks, dt_best_preds_2weeks)

// COMMAND ----------

display(dt_final_preds_2weeks)

// COMMAND ----------

val dt_2weeks_smape = dt_best_model_2weeks.smape(dt_final_preds_2weeks, 2)
println(dt_2weeks_smape)

// COMMAND ----------

// MAGIC %md ### Gradient Boosted Tree

// COMMAND ----------

val gbt_cv_2weeks = new RollingCV("gbt", gbt_param_grid)
val gbt_best_params_2weeks = gbt_cv_2weeks.initiate(train_set_2weeks, features_2weeks) 

println(gbt_best_params_2weeks)

val gbt_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", gbt_best_params_2weeks)

val gbt_best_preds_2weeks = gbt_best_model_2weeks.GBTPredict()

val gbt_final_preds_2weeks = gbt_best_model_2weeks.retrend(test_diff_2weeks, gbt_best_preds_2weeks)

// COMMAND ----------

val gbt_best_model_2weeks = new TrainModel(train_set_2weeks, test_set_2weeks, features_2weeks, "rmse", Map("MaxIter" -> "5", "MaxDepth" -> "6", "MaxBins" -> "23"))

val gbt_best_preds_2weeks = gbt_best_model_2weeks.GBTPredict()

val gbt_final_preds_2weeks = gbt_best_model_2weeks.retrend(test_diff_2weeks, gbt_best_preds_2weeks)

// COMMAND ----------

display(gbt_final_preds_2weeks)

// COMMAND ----------

val gbt_2weeks_smape = gbt_best_model_2weeks.smape(gbt_final_preds_2weeks, 2)
println(gbt_2weeks_smape)

// COMMAND ----------

// MAGIC %md ## Three Weeks Prediction

// COMMAND ----------

// MAGIC %md ### Simple Moving Average (SMA)

// COMMAND ----------

val sma_3weeks_smape = sma_predict(df, triweekly_w)
println(sma_3weeks_smape)

// COMMAND ----------

// MAGIC %md ### Random Forest

// COMMAND ----------

val rf_cv_3weeks = new RollingCV("rf", rf_param_grid)
val rf_best_params_3weeks = rf_cv_3weeks.initiate(train_set_3weeks, features_3weeks)

println(rf_best_params_3weeks)

val rf_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", rf_best_params_3weeks)

val rf_best_preds_3weeks = rf_best_model_3weeks.RandomForestPredict()

val rf_final_preds_3weeks = rf_best_model_3weeks.retrend(test_diff_3weeks, rf_best_preds_3weeks)

// COMMAND ----------

val rf_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", Map("NumTrees" -> "15", "MaxBins" -> "28"))

val rf_best_preds_3weeks = rf_best_model_3weeks.RandomForestPredict()

val rf_final_preds_3weeks = rf_best_model_3weeks.retrend(test_diff_3weeks, rf_best_preds_3weeks)

// COMMAND ----------

display(rf_final_preds_3weeks)

// COMMAND ----------

val rf_3weeks_smape = rf_best_model_3weeks.smape(rf_final_preds_3weeks, 3)
println(rf_3weeks_smape)

// COMMAND ----------

// MAGIC %md ### Linear Regression

// COMMAND ----------

val lr_cv_3weeks = new RollingCV("lr", lr_param_grid)
val lr_best_params_3weeks = lr_cv_3weeks.initiate(train_set_3weeks, features_3weeks)  

println(lr_best_params_3weeks)

val lr_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", lr_best_params_3weeks)

val lr_best_preds_3weeks = lr_best_model_3weeks.LinearRegressionPredict()

val lr_final_preds_3weeks = lr_best_model_3weeks.retrend(test_diff_3weeks, lr_best_preds_3weeks)

// COMMAND ----------

val lr_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", Map("RegParam" -> "0.1", "ElasticNetParam" ->"0"))

val lr_best_preds_3weeks = lr_best_model_3weeks.LinearRegressionPredict()

val lr_final_preds_3weeks = lr_best_model_3weeks.retrend(test_diff_3weeks, lr_best_preds_3weeks)

// COMMAND ----------

display(lr_final_preds_3weeks)

// COMMAND ----------

val lr_3weeks_smape = lr_best_model_3weeks.smape(lr_final_preds_3weeks, 3)
println(lr_3weeks_smape)

// COMMAND ----------

// MAGIC %md ### Decision Tree

// COMMAND ----------

val dt_cv_3weeks = new RollingCV("dt", dt_param_grid)
val dt_best_params_3weeks = dt_cv_3weeks.initiate(train_set_3weeks, features_3weeks)  

println(dt_best_params_3weeks)

val dt_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", dt_best_params_3weeks)

val dt_best_preds_3weeks = dt_best_model_3weeks.DecisionTreePredict()

val dt_final_preds_3weeks = dt_best_model_3weeks.retrend(test_diff_3weeks, dt_best_preds_3weeks)

// COMMAND ----------

val dt_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", Map("MaxDepth" -> "5", "MaxBins" -> "20"))

val dt_best_preds_3weeks = dt_best_model_3weeks.DecisionTreePredict()

val dt_final_preds_3weeks = dt_best_model_3weeks.retrend(test_diff_3weeks, dt_best_preds_3weeks)

// COMMAND ----------

display(dt_final_preds_3weeks)

// COMMAND ----------

val dt_3weeks_smape = dt_best_model_3weeks.smape(dt_final_preds_3weeks, 3)
println(dt_3weeks_smape)

// COMMAND ----------

// MAGIC %md ### Gradient Boosted Tree

// COMMAND ----------

val gbt_cv_3weeks = new RollingCV("gbt", gbt_param_grid)
val gbt_best_params_3weeks = gbt_cv_3weeks.initiate(train_set_3weeks, features_3weeks) 

println(gbt_best_params_3weeks)

val gbt_best_model_3weeks = new TrainModel(train_set_3weeks, test_set_3weeks, features_3weeks, "rmse", gbt_best_params_3weeks)

val gbt_best_preds_3weeks = gbt_best_model_3weeks.GBTPredict()

val gbt_final_preds_3weeks = gbt_best_model_3weeks.retrend(test_diff_3weeks, gbt_best_preds_3weeks)

// COMMAND ----------

display(gbt_final_preds_3weeks)

// COMMAND ----------

val gbt_3weeks_smape = gbt_best_model_3weeks.smape(gbt_final_preds_3weeks, 3)
println(gbt_3weeks_smape)

// COMMAND ----------

// MAGIC %md # Model Accuracy Comparison

// COMMAND ----------

val accuracy_results = Seq(
  ("SMA", "NA", sma_2weeks_smape, sma_3weeks_smape),
  ("Random Forest", rf_1week_smape, rf_2weeks_smape, rf_3weeks_smape),
  ("Decision Tree", dt_1week_smape, dt_2weeks_smape, dt_3weeks_smape),
  ("Linear Regression", lr_1week_smape, lr_2weeks_smape, lr_3weeks_smape),
  ("Gradient Boosted Tree", gbt_1week_smape, gbt_2weeks_smape, gbt_3weeks_smape),
).toDF("Model", "1 week", "2 weeks", "3 weeks")

display(accuracy_results)
