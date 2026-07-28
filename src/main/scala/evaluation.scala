package rezervoir

import zio.* 

import breeze.linalg.sum
import breeze.numerics.{abs, pow, signum}
import breeze.stats.mean

trait Metric:
    def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double]

//If the Dataset is empty return -1.0 which does not really make any sense since a negative number cannot be returned from these metrics
object NormalizedMeanSquareError extends Metric:

    override def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double] = (for {
        dataset <- ZIO.service[Dataset]
        model   <- ZIO.service[Model[?, ?, ?, ?]]
        
        // Unify both dataset types into a single functional stream of batches
        batches <- dataset match
            case full: FullDataset => 
                full.next().map(_.toList) // Wrap the Option into a List stream
            case batched: BatchedDataset => 
                batched.stream().runCollect.map(_.toList)

        // Extract predictions and targets
        pairs = batches.collect { case Batch.Labeled(x, y) => (model.predict(Batch.Labeled(x, y)), y) }
        
        score <- if pairs.isEmpty then ZIO.fail(None) else ZIO.succeed {
            // Combine all batches into giant evaluation matrices for calculation
            val yHat = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (Batch.Labeled(pred, _), _) => pred }*)
            val y    = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (_, target) => target }*)
            
            val mse = sum(pow(y - yHat, 2)) / y.size.toDouble
            val yMean = mean(y)
            val variance = sum(pow(y - yMean, 2)) / y.size.toDouble
            
            if variance == 0.0 then 0.0 else mse / variance
        }
    } yield score).orElse(ZIO.succeed(-1.0))

object MeanAbsolutePercentageError extends Metric:

    override def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double] = (for {
        dataset <- ZIO.service[Dataset]
        model   <- ZIO.service[Model[?, ?, ?, ?]]
        
        batches <- dataset match
            case full: FullDataset => 
                full.next().map(_.toList)
            case batched: BatchedDataset => 
                batched.stream().runCollect.map(_.toList)
                
        pairs = batches.collect { case Batch.Labeled(x, y) => (model.predict(Batch.Labeled(x, y)), y) }
        
        score <- if pairs.isEmpty then ZIO.fail(None) else ZIO.succeed {
            val yHat = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (Batch.Labeled(pred, _), _) => pred }*)
            val y    = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (_, target) => target }*)
            
            // Avoid division by zero if target values are exactly 0.0
            val eps = 1e-8
            val absolutePercentageError = abs((y - yHat) / (y + eps))
            
            (sum(absolutePercentageError) / y.size.toDouble) * 100.0
        }
    } yield score).orElse(ZIO.succeed(-1.0))

object RootMeanSquaredError extends Metric:
    override def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double] = (for {
        dataset <- ZIO.service[Dataset]
        model   <- ZIO.service[Model[?, ?, ?, ?]]
        batches <- dataset match
            case full: FullDataset => full.next().map(_.toList)
            case batched: BatchedDataset => batched.stream().runCollect.map(_.toList)
        pairs = batches.collect { case Batch.Labeled(x, y) => (model.predict(Batch.Labeled(x, y)), y) }
        score <- if pairs.isEmpty then ZIO.fail(None) else ZIO.succeed {
            val yHat = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (Batch.Labeled(pred, _), _) => pred }*)
            val y    = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (_, target) => target }*)
            
            math.sqrt(sum(pow(y - yHat, 2)) / y.size.toDouble)
        }
    } yield score).orElse(ZIO.succeed(-1.0))

object MeanAbsoluteError extends Metric:
    override def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double] = (for {
        dataset <- ZIO.service[Dataset]
        model   <- ZIO.service[Model[?, ?, ?, ?]]
        batches <- dataset match
            case full: FullDataset => full.next().map(_.toList)
            case batched: BatchedDataset => batched.stream().runCollect.map(_.toList)
        pairs = batches.collect { case Batch.Labeled(x, y) => (model.predict(Batch.Labeled(x, y)), y) }
        score <- if pairs.isEmpty then ZIO.fail(None) else ZIO.succeed {
            val yHat = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (Batch.Labeled(pred, _), _) => pred }*)
            val y    = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (_, target) => target }*)
            
            sum(abs(y - yHat)) / y.size.toDouble
        }
    } yield score).orElse(ZIO.succeed(-1.0))

object DirectionAccuracy extends Metric:
    override def apply(): ZIO[Dataset & Model[?, ?, ?, ?], Nothing, Double] = (for {
        dataset <- ZIO.service[Dataset]
        model   <- ZIO.service[Model[?, ?, ?, ?]]
        batches <- dataset match
            case full: FullDataset => full.next().map(_.toList)
            case batched: BatchedDataset => batched.stream().runCollect.map(_.toList)
        pairs = batches.collect { case Batch.Labeled(x, y) => (model.predict(Batch.Labeled(x, y)), y) }
        score <- if pairs.isEmpty then ZIO.fail(None) else ZIO.succeed {
            val yHat = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (Batch.Labeled(pred, _), _) => pred }*)
            val y    = breeze.linalg.DenseMatrix.vertcat(pairs.map { case (_, target) => target }*)
            
            // Slice matrices to compare step t+1 against step t
            val yDiff    = y(1 until y.rows, ::) - y(0 until y.rows - 1, ::)
            val yHatDiff = yHat(1 until yHat.rows, ::) - y(0 until y.rows - 1, ::)
            
            // Check where the signs match (both up or both down)
            val matches = signum(yDiff) :== signum(yHatDiff)
            
            // Count trues and divide by total elements evaluated
            val correctDirections = matches.activeValuesIterator.count(_ == true)
            correctDirections.toDouble / matches.size.toDouble * 100.0
        }
    } yield score).orElse(ZIO.succeed(-1.0))