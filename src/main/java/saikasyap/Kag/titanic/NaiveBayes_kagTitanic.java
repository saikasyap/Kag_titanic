package saikasyap.Kag.titanic;

/**
 * Created by saikasyap on 6/27/17.
 */import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.feature.Bucketizer;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.catalyst.expressions.IsNotNull;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import scala.collection.mutable.WrappedArray;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.Tokenizer;


public class NaiveBayes_kagTitanic {
    public static void main(String[] args) {
        Logger.getLogger( "org" ).setLevel( Level.ERROR );
        Logger.getLogger( "akka" ).setLevel( Level.ERROR );

        SparkConf sparkConf = new SparkConf()
                .setMaster( "local[*]" )
                .setAppName( "Titanic Spark" );
        JavaSparkContext javaSparkContext = new JavaSparkContext( sparkConf );
        SQLContext sqlContext = new SQLContext( javaSparkContext );

		/*--------------------------------------------------------------------------
		Loading  train Data
		--------------------------------------------------------------------------*/
        //Create the schema for the data to be loaded into Dataset.

        StructType dataSchema = DataTypes
                .createStructType(new StructField[] {
                        DataTypes.createStructField("PassengerId", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Survived", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Pclass", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Name", DataTypes.StringType, false),
                        DataTypes.createStructField("Sex", DataTypes.StringType, false),
                        DataTypes.createStructField("Age", DataTypes.DoubleType, false),
                        DataTypes.createStructField("SibSp", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Parch", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Ticket", DataTypes.StringType, false),
                        DataTypes.createStructField("Fare", DataTypes.DoubleType, false),
                        DataTypes.createStructField("Bin", DataTypes.StringType, false),
                        DataTypes.createStructField("Embarked", DataTypes.StringType, false)
                });


        Dataset<Row> trainDf = sqlContext.read().option( "header", "true" ).option( "inferSchema", "true" ).csv( "/home/saikasyap/Kag_titanic/train.csv" );
        trainDf.show( 5 );
        trainDf.printSchema();
		/*--------------------------------------------------------------------------
		Cleanse Data
		--------------------------------------------------------------------------*/


        System.out.println( "Number of passengers in training data: " + trainDf.count() );

        Dataset<Row> protrainDf = processData( trainDf ,sqlContext);
        protrainDf.show();


		/*--------------------------------------------------------------------------
		Analyze Data
		--------------------------------------------------------------------------*/

        //Perform correlation analysis
        for ( StructField field : dataSchema.fields() ) {
            if ( ! field.dataType().equals(DataTypes.StringType)) {
                System.out.println( "Correlation between Survived and " + field.name()
                        + " = " + protrainDf.stat().corr("Survived", field.name()) );
            }
        }

// Converting string labels into indices.
        StringIndexer embarkedIndexer = new StringIndexer().setInputCol( "Embarked" ).setOutputCol( "EmbarkedIndexed" ).setHandleInvalid( "skip" );
        StringIndexer sexIndexer = new StringIndexer().setInputCol( "Sex" ).setOutputCol( "SexIndexed" ).setHandleInvalid( "skip" );
        StringIndexer survivedIndexer = new StringIndexer().setInputCol( "Survived" ).setOutputCol( "SurvivedLabel" ).setHandleInvalid( "skip" );
        StringIndexer TitleIndexer = new StringIndexer().setInputCol( "title" ).setOutputCol( "TitleIndexed" ).setHandleInvalid( "skip" );


        double[] fareSplits = new double[]{0.0, 10.0, 20.0, 30.0, 40.0, Double.POSITIVE_INFINITY};
        Bucketizer fareBucketize = new Bucketizer()
                .setInputCol("Fare")
                .setOutputCol("FareBucketed")
                .setSplits( fareSplits );


        // Creating dummy columns
        OneHotEncoder embEncoder = new OneHotEncoder().setInputCol( "EmbarkedIndexed" ).setOutputCol( "EmbarkedVec" );
        OneHotEncoder sexEncoder = new OneHotEncoder().setInputCol( "SexIndexed" ).setOutputCol( "SexVec" );
        OneHotEncoder titleEncoder = new OneHotEncoder().setInputCol( "TitleIndexed" ).setOutputCol( "TitleVec" );
        // The vector assembler creates a feature column where it combines all the required features at one place.
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols( new String[]{"Pclass", "SexVec", "AgeCat", "SibSp", "Parch", "FareBucketed", "EmbarkedVec", "Family", "Child", "Mom","TitleIndexed"} )
                .setOutputCol( "features" );

        // Performing PCA

        PCA pca = new PCA().setK( 10 ).setInputCol( "features" ).setOutputCol( "pcaFeatures" );

        // Train a DecisionTree model.

        NaiveBayes nbClassifier = new NaiveBayes()
                .setLabelCol("Survived")
                .setFeaturesCol("pcaFeatures");



        Pipeline pipeline = new Pipeline()
                .setStages( new PipelineStage[]{embarkedIndexer, sexIndexer, survivedIndexer, TitleIndexer , fareBucketize , embEncoder, sexEncoder, assembler, pca, nbClassifier} );
        // Specifying grid search parameters

        //Split the data into training data set (20% held out for testing).
        Dataset<Row>[] splits = protrainDf.randomSplit(new double[]{0.85, 0.15});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        //Create the object

        PipelineModel model = pipeline.fit(trainingData);
        Dataset<Row> result = model.transform( testData );
        result.show(5);
        result = result.select("PassengerId","Survived");
        //View confusion matrix
        System.out.println("Confusion Matrix :");
        result.groupBy(col("Survived"), col("prediction")).count().show();









        // Split the data into training data set (20% held out for testing).
        //  Dataset<Row>[] splits = protrainDf.randomSplit(new double[]{0.8, 0.2});
        //    Dataset<Row> trainingData = splits[0];
        //      Dataset<Row> testData = splits[1];

        // Convert indexed labels back to original labels.


//        RandomForestClassificationModel Model = rfc.fit(trainingData);

// processing the test data...
        Dataset<Row> testDf = sqlContext.read().option( "header", "true" ).option( "inferSchema", "true" ).csv( "/home/saikasyap/Kag_titanic/test.csv" );
        testDf.show( 5 );
        testDf.printSchema();

        Dataset<Row> InptestDf = processData( testDf,sqlContext );
        // Prediction on test data.
    /*    Dataset<Row> predTest = cvModel.transform( InptestDf );
        predTest = predTest.withColumn( "prediction", predTest.col( "prediction" ).cast( "int" ) );
        predTest = predTest.withColumnRenamed( "prediction", "Survived" );
        predTest.select( "PassengerId", "Survived" ).show( 30 );
        // Saving to a File
        predTest.coalesce( 1 )
                .select( "PassengerId", "Survived" )
                .write().mode( "Overwrite" )
                .option( "header", "true" )
                .csv( "/home/saikasyap/Kag_titanic/output");
*/
    }

    //Feature Engineering....

    public static Dataset<Row> processData (Dataset < Row > data, SQLContext sqlContext)
    {
        data = data.drop( "Ticket" ).drop( "Cabin" );
        Double ageMean = data.select( mean( "Age" ) ).head().getDouble( 0 );
        Double fareMean = data.select( mean( "Fare" ) ).head().getDouble( 0 );
        data = data.na().fill( ageMean, new String[]{"Age"} );
        data = data.na().fill( fareMean, new String[]{"Fare"} );


        data = data.withColumn( "Family", data.col( "Parch" ).$plus( data.col( "SibSp" )).$plus( 1 ));


        sqlContext.udf().register( "childInd", new UDF1<Double, Integer>() {
            @Override
            public Integer call(Double val) {
                if (val < 16)
                    return 1;
                else
                    return 0;
            }
        }, DataTypes.IntegerType );

        sqlContext.udf().register( "FindTitle", new UDF1<String, String>() {
            @Override
            public String call(String Name) throws Exception {
                String title= Name;

                Pattern p = Pattern.compile( "(Dr|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Lady|Mme|Don|Mr.)" );
                Matcher m = p.matcher( Name);

                if (m.find()) {
                    title = m.group(1);
                }


                switch (title) {
                    case "Mrs?":
                    case "Dona":
                    case "Mme":
                    case "Lady":
                        title = "Mrs";
                        break;
                    case "Rev":
                    case "Col":
                    case "Major":
                    case "Capt":
                    case "Master":
                    case "Jonkheer":
                    case "Sir":
                    case "Don":
                    case "Dr" :
                        title = "Mr";
                        break;
                    case "Mlle":
                    case "Countess":
                    case "Ms":
                        title = "Miss";
                        break;
                }
                return title;
            }

        }, DataTypes.StringType );



        sqlContext.udf().register( "AgeCateg", new UDF1<Double, Integer>() {
            @Override
            public Integer call(Double age) {
                if (age > 0 && age < 20) {
                    return 1;
                } else if (age >= 20 && age < 30) {
                    return 2;
                } else if (age >= 30 && age < 50) {
                    return 3;
                } else if (age >= 50) {
                    return 4;
                }
                return 0;
            }
        }, DataTypes.IntegerType );

        sqlContext.udf().register( "momInd", new UDF4<Double, String, Integer, String, Integer>() {
            @Override
            public Integer call(Double age, String gender, Integer parch, String name) {
                if ((age > 17) && (gender.equals( "female" )) && (parch > 0) && (!name.contains( "Miss" )))
                    return 1;
                else
                    return 0;
            }
        }, DataTypes.IntegerType );


        data = data.withColumn( "Child", callUDF( "childInd", data.col( "Age" ) ) );
        data = data.withColumn( "Mom", callUDF( "momInd", data.col( "Age" ), data.col( "Sex" ), data.col( "Parch" ), data.col( "Name" ) ) );
        data = data.withColumn( "title", callUDF( "FindTitle", data.col( "Name" ) ) );
        data = data.withColumn( "AgeCat", callUDF( "AgeCateg", data.col( "Age" ) ) );


        return data;





    }
// Have a got a score of 0.79426/1 after submission for this code...




}


