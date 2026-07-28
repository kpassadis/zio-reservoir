package tests

import zio.*
import rezervoir.Utils.*
import rezervoir.BatchedDataset
import rezervoir.Dataset
import breeze.linalg.DenseMatrix
import rezervoir.Batch
import breeze.linalg.*
import zio.json.*
import rezervoir.{Utils, toBatch}


class DataTests extends munit.FunSuite:

    private val x = DenseMatrix((1.0, 2.0, 3.0),(4.0, 5.0, 6.0),(7.0, 8.0, 9.0),(10.0, 11.0, 12.0))


    test("Air Passengers Batch from DenseMatrix has correct number of rows") {
        val passengers = Dataset.airpassengers()
        val batch = passengers.toBatch(nAhead = 1)
        assertEquals(batch.rows(), passengers.rows - 1)
    }

    test("Batched Dataset from air passengers") {
        val passengers = Dataset.airpassengers()
        val fullBatch = passengers.toBatch(nAhead = 1)
        val program = for {
            dataset <- BatchedDataset.fromBatch(10, fullBatch)
            _ <- dataset.stream().foreach(batch => Console.printLine(s"Batch size ${batch.rows()}"))
            tuple <- dataset.split(0.8)
            (train, test) = tuple
            _ <- Console.printLine(s"Train rows: ${train.indexes}, Test rows: ${test.indexes}")

        } yield ()

        program.unsafeRun()
    }

    test("Batch enum pretty print") {
        val lorenz = Dataset.lorenz(steps = 100)
        val scaler = MinMaxScaler.fit(lorenz)
        val scaled = scaler.transform(lorenz)
        println(scaled)
    }

    test("Air passengers dataset to batch") {
        val mat = Dataset.airpassengers()
        val batch = mat.toBatch(4)
        batch match
            case Batch.Labeled(x, y) => 
                assertEquals(x.rows, y.rows)
                println(y)
            case Batch.Unlabeled(_) =>
                assertEquals(1, 2)
        
    }

    test("Horizontal concatenation") {
        val z = DenseMatrix.horzcat(x, x)
        assertEquals(z.cols, 6)
    }

    test("Batched dataset has expected number of rows") {
        
        val datasetLayer = BatchedDataset.live(2, x)
        val program = for {
            dataset <- ZIO.service[BatchedDataset]
            x <- dataset.next()
        } yield x

        val batchOpt = program.provideLayer(datasetLayer).unsafeRun()
        assert(batchOpt.isDefined)

        val batch = batchOpt.get
        batch match {
            case Batch.Unlabeled(x) => assert(x.rows == 2)
            case Batch.Labeled(x, y) => assert(x.rows == 2)
        }
    }

    test("Reseting a batched dataset returns the data at the beginning of the set") {
        val dataset = BatchedDataset.live(1, x)
        val program = for {
            dataset <- ZIO.service[BatchedDataset]
            batch <- dataset.next()
            _ <- dataset.reset()
            batch1 <- dataset.next()
        } yield (batch, batch1)

        val (aOpt, bOpt) = program.provide(dataset).unsafeRun()
        val (a, b) = (aOpt.get, bOpt.get)
        val res = (b, b) match
            case (Batch.Unlabeled(x), Batch.Unlabeled(y)) => sum(x - y) == 0.0
            case _ =>  false
        assert(res)
    }

    test("When a timeseries with 100 timesteps is split into batches of size 25 then 4 batches should be created") {
        val timeseries = Dataset.mackeyGlass(100)
        assertEquals(timeseries.rows, 100)
        val batchSize = 25
        val datasetLayer = BatchedDataset.live(batchSize, timeseries)
        val n = (for {
            dataset <- ZIO.service[BatchedDataset]
            c <- dataset.stream().runFold(0){case (acc, b) => acc + b.rows() / batchSize}
        } yield c).provide(datasetLayer)

        assertEquals(n.unsafeRun(), 4)

    }

    test("Lead should return a timeseries with the first n rows removed from the matrix") {
        val x = DenseMatrix.tabulate(10, 1){case (i, j) => i.toDouble}
        val y = Dataset.lead(x, 4)
        assertEquals(y.rows, 6)
    }

    test("Json Encoder/Decoder for Batch enum") {
        val x = DenseMatrix.tabulate(10, 1){case (i, j) => i.toDouble}
        val y = DenseMatrix.tabulate(10, 1){case (i, j) => i.toDouble + 2}
        val batch = Batch.Labeled(x, y)
        val encoded = batch.toJson
        
        val decodedE = encoded.fromJson[Batch]
        assert(decodedE.isRight)
        val decoded = decodedE.toOption.get
        assert(batch == decoded)
    }

    test("Batch read/write to/from file works both ways") {
        val x = DenseMatrix.tabulate(10, 1){case (i, j) => i.toDouble}
        val y = DenseMatrix.tabulate(10, 1){case (i, j) => i.toDouble + 2}
        val batch = Batch.Labeled(x, y)
        val encoded = batch.toJson
        
    }
