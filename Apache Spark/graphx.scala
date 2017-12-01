// Databricks notebook source
val bikeStations = sqlContext.sql("SELECT * FROM stationdata_csv")
val tripData = sqlContext.sql("SELECT * FROM tripdata_csv")

// COMMAND ----------

display(bikeStations)

// COMMAND ----------

display(tripData)

// COMMAND ----------

bikeStations.printSchema()
tripData.printSchema()

// COMMAND ----------

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

// COMMAND ----------

val justStations = bikeStations
  .selectExpr("float(station_id) as station_id", "name")
  .distinct()

val completeTripData = tripData
  .join(justStations, tripData("Start Station") === bikeStations("name"))
  .withColumnRenamed("station_id", "start_station_id")
  .drop("name")
  .join(justStations, tripData("End Station") === bikeStations("name"))
  .withColumnRenamed("station_id", "end_station_id")
  .drop("name")

// COMMAND ----------

val stations = completeTripData
  .select("start_station_id", "end_station_id")
  .rdd
  .distinct() // helps filter out duplicate trips
  .flatMap(x => Iterable(x(0).asInstanceOf[Number].longValue, x(1).asInstanceOf[Number].longValue)) // helps us maintain types
  .distinct()
  .toDF() // return to a DF to make merging + joining easier

stations.take(1) // this is just a station_id at this point

// COMMAND ----------

val stationVertices: RDD[(VertexId, String)] = stations
  .join(justStations, stations("value") === justStations("station_id"))
  .select("station_id", "name")
  .rdd
  .map(row => (row(0).asInstanceOf[Number].longValue, row(1).asInstanceOf[String])) // maintain type information

stationVertices.take(1)

// COMMAND ----------

val stationEdges:RDD[Edge[Long]] = completeTripData
  .select("start_station_id", "end_station_id")
  .rdd
  .map(row => Edge(row(0).asInstanceOf[Number].longValue, row(1).asInstanceOf[Number].longValue, 1))

// COMMAND ----------

val defaultStation = ("Missing Station") 
val stationGraph = Graph(stationVertices, stationEdges, defaultStation)
stationGraph.cache()

// COMMAND ----------

println("Total Number of Stations: " + stationGraph.numVertices)
println("Total Number of Trips: " + stationGraph.numEdges)
// sanity check
println("Total Number of Trips in Original Data: " + tripData.count)

// COMMAND ----------


