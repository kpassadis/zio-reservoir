package rezervoir

import zio.* 
import zio.stream.*
import breeze.linalg.{DenseMatrix, min, max, csvread}
import scala.util.{Try, Using, Random}
import zio.json.*
import scala.reflect.ClassTag
import breeze.storage.Zero
import java.io.{BufferedReader, FileReader}
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import scala.util.Random


extension (batch: Batch)
    /**
     * Chronologically splits a Batch into two separate batches.
     * @param prop The proportion of data for the first batch (e.g., 0.8 for 80% Train, 20% Test)
     * @return A tuple of (FirstBatch, SecondBatch)
     */
    def split(prop: Double): (Batch, Batch) = 
        require(prop >= 0.0 && prop <= 1.0, "Split proportion must be between 0.0 and 1.0")
        
        batch match
            case Batch.Labeled(x, y) =>
                val splitIdx = (x.rows * prop).toInt
                
                // 1. Slice and copy the Features (X)
                val x1 = x(0 until splitIdx, ::).copy
                val x2 = x(splitIdx until x.rows, ::).copy
                
                // 2. Slice and copy the Targets (Y) at the exact same index
                val y1 = y(0 until splitIdx, ::).copy
                val y2 = y(splitIdx until y.rows, ::).copy
                
                (Batch.Labeled(x1, y1), Batch.Labeled(x2, y2))
                
            case Batch.Unlabeled(x) =>
                val splitIdx = (x.rows * prop).toInt
                
                val x1 = x(0 until splitIdx, ::).copy
                val x2 = x(splitIdx until x.rows, ::).copy
                
                (Batch.Unlabeled(x1), Batch.Unlabeled(x2))

extension (m:DenseMatrix[Double])

    /**
     * A function to create a Batch from a univariate time series represented as a DenseMatrix of shape [Nx1].
     * The function will return a Labeled or an Unlabeled Batch depending on the parameter nAhead. If nAhead is equal
     * to zero or larger than the number of rows of the matrix (in other words out of bounds) then 
       an Unlabeled batch is created and returned. If nAhead is specified within bounds then a Labeled Batch
       is returned where 
     * 
     */
    def toBatch(nAhead: Int): Batch = 
        if nAhead <= 0 || nAhead >= m.rows then Batch.Unlabeled(m)
        else
            val rows = m.rows - nAhead
            
            // Optimized Input: Directly slice and copy the submatrix in one shot
            val x = m(0 until rows, ::).copy
            
            // Optimized Output: Allocate the target matrix
            val y = DenseMatrix.zeros[Double](rows, nAhead)
            
            // Vectorized assignment: Populate column-by-column rather than cell-by-cell
            for c <- 0 until nAhead do
                y(::, c) := m(c + 1 until rows + c + 1, 0)

            Batch.Labeled(x, y)
            

    // If s is negative it will drop from the head of the matrix and if negative it will drop the tail of the matrix.
    def drop(s:Int):DenseMatrix[Double] =
        if s == 0 then m.copy
        else
            val m1 = DenseMatrix.zeros[Double](m.rows - math.abs(s), m.cols)
            if s > 0 && s < m.rows then
                m1 := m(0 until m.rows - s, ::)
            else if s < 0 && -s < m.rows then
                m1 := m(math.abs(s) until m.rows, ::)
            else m1
        
    def shift(s:Int):DenseMatrix[Double] =
        if s == 0 then m.copy
        else
            val shifted = DenseMatrix.zeros[Double](m.rows, m.cols)
            if s > 0 && s < m.rows then
                // Lag (Positive): Shift data DOWN.
                // Example s=1: rows 1 to N get data from 0 to N-1. Row 0 remains 0.0.
                shifted(s until m.rows, ::) := m(0 until m.rows - s, ::)
                
            else if s < 0 && -s < m.rows then
                // Lead (Negative): Shift data UP.
                // Example s=-1: rows 0 to N-1 get data from 1 to N. Row N remains 0.0.
                val absSteps = -s
                shifted(0 until m.rows - absSteps, ::) := m(absSteps until m.rows, ::)
                    
            // Note: If |s| >= m.rows, it correctly returns an all-zero matrix
            shifted


/**
 * Batch is an Scala enum to model labeled and unlabeled datasets. Since Echo State Networks are typically used in 
 * a supervised setting the categorizationof data into labeled and unlabeld is used to help distinguish the mode
 * of operation. When the model receives Labeled data it means that the target values are known, and this can only be 
 * true during the training process. On the other hand, if a model receives Unlabeled data it operates in inference mode.
 * 
*/
enum Batch:
    case Labeled(x:DenseMatrix[Double], y:DenseMatrix[Double])
    case Unlabeled(x:DenseMatrix[Double])

    /**
     * It is not trivial to randomly split a time series dataset as it is necessary to maintain the time order.
     * One methodology to perform a random split is called Moving Block Bootstrap. The process is the following:
     * 1. Define a block size L which captures the dynamics of the time series. These blocks are the objects you sample with replacement.
     * 2. Sample with replacement from these blocks. These blocks are 
     * 
     * 
     * @param blockSize an integer that represents the size of the block
     * @param n an integer used to specify the size of the bootstrap sample
    */
    def movingBlockBootstrap(blockSize: Int, n: Int, seed: Long = 42L): Seq[Batch] = 
        val totalRows = this.rows()
        val rng = new scala.util.Random(seed)
        
        val maxStartIdx = totalRows - blockSize

        val blocks = (0 to maxStartIdx).map { start =>
            val end = start + blockSize - 1
            this match {
                // copy prevents memory aliasing crashes during vertcat
                case Labeled(x, y) => Labeled(x(start to end, ::).copy, y(start to end, ::).copy)
                case Unlabeled(x)  => Unlabeled(x(start to end, ::).copy)
            }
        }.toIndexedSeq

        val blocksNeeded = math.ceil(totalRows.toDouble / blockSize).toInt

        val samples = (0 until n).map { _ =>
            
            val sampledBlocks = (0 until blocksNeeded).map { _ =>
                blocks(rng.nextInt(blocks.length))
            }

            val stitchedBatch = sampledBlocks.reduceLeft(_ vertcat _)

            // Because blocksNeeded * blockSize might be slightly larger than totalRows,
            // we must truncate the final batch back to the exact original dataset length.
            stitchedBatch match {
                case Labeled(x, y) => Labeled(x(0 until totalRows, ::).copy, y(0 until totalRows, ::).copy)
                case Unlabeled(x)  => Unlabeled(x(0 until totalRows, ::).copy)
            }
        }

        samples // Return the generated samples, not the blocks!
    

    /**
     * Utility functions to facilitate working with batches instead of working directly with 
     * dense matrices.
     * 
    */
    def vertcat(other:Batch):Batch = (this, other) match
        case (Labeled(x, y), Labeled(x1, y1) )=>
            Labeled(DenseMatrix.vertcat(x, x1), DenseMatrix.vertcat(y, y1))
        case (Unlabeled(x), Unlabeled(x1))=>
            Unlabeled(DenseMatrix.vertcat(x, x1))
        case _ => this 

    def horzcat(other:Batch):Batch = (this, other) match
        case (Labeled(x, y), Labeled(x1, y1) )=>
            Labeled(DenseMatrix.horzcat(x, x1), DenseMatrix.horzcat(y, y1))
        case (Unlabeled(x), Unlabeled(x1))=>
            Unlabeled(DenseMatrix.horzcat(x, x1))
        case _ => this 
      
    def exportToJson(filename:String): String = 
        Files.writeString(Paths.get(filename), this.toJson).toString

    /**
     * Returns the numer of rows of a batch. 
     * 
    */
    def rows():Int = this match
        case Labeled(x, _) => x.rows
        case Unlabeled(x) => x.rows  

    def cols():(Int, Int) = this match
        case Labeled(x, y) => (x.cols, y.cols)
        case Unlabeled(x) => (x.cols, x.cols)
  
    def shift(steps: Int): Batch = 
    
        /**
         * Helper to safely shift any matrix while preserving dimensions. In other words,
         * a matrix that contains the same number of time steps will be returned.
         * This is equivalent to zero-padding. 
        */
        def shiftMatrix(m: DenseMatrix[Double], s: Int): DenseMatrix[Double] =
            if s == 0 then m.copy
            else
                val shifted = DenseMatrix.zeros[Double](m.rows, m.cols)
                if s > 0 && s < m.rows then
                    // Lag (Positive): Shift data DOWN.
                    // Example s=1: rows 1 to N get data from 0 to N-1. Row 0 remains 0.0.
                    shifted(s until m.rows, ::) := m(0 until m.rows - s, ::)
                
                else if s < 0 && -s < m.rows then
                    // Lead (Negative): Shift data UP.
                    // Example s=-1: rows 0 to N-1 get data from 1 to N. Row N remains 0.0.
                    val absSteps = -s
                    shifted(0 until m.rows - absSteps, ::) := m(absSteps until m.rows, ::)
                    
                // Note: If |s| >= m.rows, it correctly returns an all-zero matrix
                shifted
        // Apply the transformation universally to whatever data the Batch contains
        this match
            case Labeled(x, y) => Batch.Labeled(shiftMatrix(x, steps), shiftMatrix(y, steps))
            case Unlabeled(x)  => Batch.Unlabeled(shiftMatrix(x, steps))
    

    override def toString(): String = this match
        case Labeled(x, y) => DenseMatrix.horzcat(x, y).toString
        case Unlabeled(x) => x.toString
    

object Batch:

    import Utils.{denseMatrixDecoder, denseMatrixEncoder}
    
    given JsonEncoder[Batch] = DeriveJsonEncoder.gen[Batch]
    
    given JsonDecoder[Batch] = DeriveJsonDecoder.gen[Batch]

    /**
     * A simple utility function to read a Json - encoded file into a batch. 
     * 
    */
    def fromJson(filename:String): ZIO[Any, Throwable, Either[String, Batch]]  = 
        ZIO.attempt(Files.readString(Paths.get(filename))).map(_.fromJson[Batch])
   

    def empty(rows:Int, inputCols:Int, outputCols:Option[Int] = None):Batch = 
        val x = DenseMatrix.zeros[Double](rows, inputCols)
        outputCols match
            case Some(value) =>
                val y = DenseMatrix.zeros[Double](rows, value)
                Labeled(x, y)
            case _ =>
                val x = DenseMatrix.zeros[Double](rows, inputCols)
                Unlabeled(x)

        
        

/**
 * The `Dataset` trait serves as an integration layer, wrapping a purely mathematical `Batch`. 
 * * A reasonable architectural question is: If `Batch` already wraps and safely manages 
 * our underlying matrix structures, why do we need another wrapper?
 * * The reason is Separation of Concerns. While `Batch` handles pure linear algebra and memory 
 * slicing, `Dataset` bridges that pure data into the highly concurrent ZIO ecosystem. 
 * * To support different learning paradigms, `Dataset` is implemented in two distinct flavors:
 * 1. `FullDataset`: Holds the entire batch in memory at once. It pairs perfectly with the 
 * `Ridge` optimizer, which utilizes closed-form LAPACK solvers to train the Readout layer instantly.
 * 2. `BatchedDataset`: Wraps the batch in a `ZStream`, providing lazy, chunked data evaluation. 
 * It is designed specifically for iterative, streaming solvers like `GradientDescent`.
 */
trait Dataset:

    /** The underlying complete batch representation of the data */
    val dataset:Batch

    /** Advances the internal cursor and yields the next chunk, or None if exhausted */
    def next():ZIO[Any, Nothing, Option[Batch]]

    /** * Chronologically splits the dataset into (Training, Testing).
     * @param prop The proportion of data to allocate to the first dataset (e.g., 0.8)
     */
    def split(prop:Double):ZIO[Any, Nothing, (Dataset, Dataset)]

    /** Returns a finite, purely functional stream of the dataset's batches */
    def stream(): ZStream[Any, Nothing, Batch]


object Dataset:

    /**
     * Loads a CSV file entirely into memory and constructs a FullDataset.
     * Assumes the CSV contains purely numerical data.
     * * @param path The filepath to the CSV.
     * @param hasHeader Boolean indicating if the first row (headers) should be skipped.
     * @param targetCol The index of the target variable. Default is -1 (the very last column).
     */
    def loadFull(path: String, hasHeader: Boolean = true, targetCol: Int = -1): ZIO[Any, Throwable, FullDataset] = 
        for {
            // ZIO.attemptBlocking prevents disk I/O from starving the compute thread pool
            matrix <- ZIO.attemptBlocking {
                csvread(new java.io.File(path), skipLines = if hasHeader then 1 else 0)
            }
            batch  <- ZIO.succeed(extractBatch(matrix, targetCol))
            dataset <- FullDataset.fromBatch(batch)
        } yield dataset


    /**
     * Loads a CSV file into memory and securely partitions it into a streaming BatchedDataset.
     * * @param path The filepath to the CSV.
     * @param batchSize The number of chronological rows to include in each batch.
     * @param hasHeader Boolean indicating if the first row should be skipped.
     * @param targetCol The index of the target variable. Default is -1 (the very last column).
     */
    def loadBatched(path: String, batchSize: Int, hasHeader: Boolean = true, targetCol: Int = -1): ZIO[Any, Throwable, BatchedDataset] = 
        for {
            matrix  <- ZIO.attemptBlocking {
                csvread(new java.io.File(path), skipLines = if hasHeader then 1 else 0)
            }
            batch   <- ZIO.succeed(extractBatch(matrix, targetCol))
            dataset <- BatchedDataset.fromBatch(batchSize, batch)
        } yield dataset


    /**
     * Internal helper to mathematically slice the raw DenseMatrix into X (features) and Y (labels).
     */
    private def extractBatch(matrix: DenseMatrix[Double], targetCol: Int): Batch = 
        val cols = matrix.cols
        
        // Resolve negative indices (e.g., -1 means the last column)
        val actualTargetCol = if (targetCol < 0) cols + targetCol else targetCol
        
        if (cols == 1) then
            // If the CSV only has 1 column, there are no features to map, it is purely unlabeled data
            Batch.Unlabeled(matrix)
        else if (actualTargetCol == cols - 1) then
            // Most common scenario: Target is the very last column
            val x = matrix(::, 0 until actualTargetCol).copy
            val y = matrix(::, actualTargetCol to actualTargetCol).copy
            Batch.Labeled(x, y)
        else
            // Complex scenario: Target is at the beginning or middle of the CSV
            val xLeft  = if actualTargetCol > 0 then matrix(::, 0 until actualTargetCol) else null
            val xRight = if actualTargetCol < cols - 1 then matrix(::, actualTargetCol + 1 until cols) else null
            
            // Re-concatenate the features, bridging the gap where the target column was removed
            val x = if xLeft == null then xRight.copy
                    else if xRight == null then xLeft.copy
                    else DenseMatrix.horzcat(xLeft, xRight)
                    
            val y = matrix(::, actualTargetCol to actualTargetCol).copy
            Batch.Labeled(x, y)

    def lorenz(steps: Int, dt: Double = 0.01, sigma: Double = 10.0, rho: Double = 28.0, beta: Double = 8.0/3.0): Batch.Labeled = 
        val data = DenseMatrix.zeros[Double](steps + 1, 3)
        
        // Initial state
        var x = 1.0; var y = 1.0; var z = 1.0
        data(0, ::) := breeze.linalg.DenseVector(x, y, z).t
        
        for t <- 1 to steps do
            // Simple Euler integration (for brevity, RK4 is better for extreme precision)
            val dx = sigma * (y - x) * dt
            val dy = (x * (rho - z) - y) * dt
            val dz = (x * y - beta * z) * dt
            
            x += dx; y += dy; z += dz
            data(t, ::) := breeze.linalg.DenseVector(x, y, z).t
            
        val features = data(0 until steps, ::).copy
        val targets  = data(1 to steps, ::).copy
        
        Batch.Labeled(features, targets)

    def airpassengers():DenseMatrix[Double] = 
        val x = Array[Double](
            112,118,132,129,121,135,148,148,136,119,104,118,115,126,141,135,125,
            149,170,170,158,133,114,140,145,150,178,163,172,178,199,199,184,162,
            146,166,171,180,193,181,183,218,230,242,209,191,172,194,196,196,236,
            235,229,243,264,272,237,211,180,201,204,188,235,227,234,264,302,293,
            259,229,203,229,242,233,267,269,270,315,364,347,312,274,237,278,284,
            277,317,313,318,374,413,405,355,306,271,306,315,301,356,348,355,422,
            465,467,404,347,305,336,340,318,362,348,363,435,491,505,404,359,310,
            337,360,342,406,396,420,472,548,559,463,407,362,405,417,391,419,461,
            472,535,622,606,508,461,390,432
        )
        val mat = new DenseMatrix(x.length, 1, x)
        val minVal = min(mat)
        val maxVal = max(mat)
    
        (mat - minVal) / (maxVal - minVal)
        

   /**
   * Generates the Mackey-Glass delayed differential equation timeseries.
   *
   * @param nTimesteps Number of timesteps to compute.
   * @param tau        Time delay. Default 17 (chaotic regime).
   * @param a          Equation parameter a. Default 0.2.
   * @param b          Equation parameter b. Default 0.1.
   * @param n          Equation parameter n. Default 10.
   * @param x0         Initial condition. Default 1.2.
   * @param h          Time delta between discrete steps. Default 1.0.
   * @param seed       Optional seed for reproducibility.
   * @param history    Optional array to warm up the process. Must be > tau/h.
   * @return           A DenseMatrix of shape [nTimesteps x 1].
   */
    def mackeyGlass(
        nTimesteps: Int,
        tau: Double = 17.0,
        a: Double = 0.2,
        b: Double = 0.1,
        n: Int = 10,
        x0: Double = 1.2,
        h: Double = 1.0,
        seed: Option[Long] = None,
        history: Option[Array[Double]] = None
    ): DenseMatrix[Double] = {

        val historyLength = math.floor(tau / h).toInt

        // 1. Initialize the history array
        val actualHistory = history match {
            case Some(hist) =>
                require(hist.length >= historyLength, s"Provided history length (${hist.length}) is < tau/h ($historyLength)")
                hist.takeRight(historyLength)
            case None =>
                val rng = seed.map(new Random(_)).getOrElse(new Random())
                Array.fill(historyLength)(x0 + 0.2 * (rng.nextDouble() - 0.5))
        }

        // 2. The Mackey-Glass Delayed Differential Equation
        def dxdt(x: Double, xtau: Double): Double = {
            (a * xtau) / (1.0 + math.pow(xtau, n.toDouble)) - b * x
        }

        // 3. 4th-Order Runge-Kutta numerical solver
        def rk4(x: Double, xtau: Double): Double = {
            val k1 = dxdt(x, xtau)
            val k2 = dxdt(x + k1 * h / 2.0, xtau)
            val k3 = dxdt(x + k2 * h / 2.0, xtau)
            val k4 = dxdt(x + k3 * h, xtau)
            x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        }

        // 4. Pre-allocate the full memory buffer
        val X = new Array[Double](historyLength + nTimesteps)
        Array.copy(actualHistory, 0, X, 0, historyLength)

        // 5. Compute the time series
        var xt = x0
        for (i <- historyLength until historyLength + nTimesteps) do {
            X(i) = xt
            val xtau = if (tau > 0) X(i - historyLength) else 0.0
            xt = rk4(xt, xtau)
        }

        // 6. Slice the results (drop the warmup history) and format as [N x 1] matrix
        val resultData = X.slice(historyLength, historyLength + nTimesteps)
        new DenseMatrix(nTimesteps, 1, resultData)
    }

    def lead(x:DenseMatrix[Double], lags:Int):DenseMatrix[Double] =
        assert(x.rows > lags + 1) 
        //preallocate the space for the lagged matrix
        val lagged = DenseMatrix.zeros[Double](x.rows - lags, x.cols)
        lagged := x(lags until x.rows, ::)
        lagged

    

    
        

/**
 * The BatchedDataset will return the data in batches. Three ingredients are required to construct a Batched dataset:
 * - A set of "ranges". Each scala Range contains the indexes of a single batch.
 * - A ZIO Ref which serves as a stateful counter in order to keep track of the current batch.
 * - The full dataset.
 * 
*/
class BatchedDataset private(val indexes:Seq[Range], val ref:Ref[Int], override val dataset:Batch) extends Dataset:

    /**
     * Splits a dataset but not randomly. Since Echo State Networks are designed to work with temporal data
     * the chronological order should be maintained. 
     * 
     * @param prop a Double value between 0 and 1 that represents the proportion of data into the training set
     * @returns a tuple of Batched datasets
     * 
    */
    /**
     * Chronologically splits a dataset. 
     * Physically duplicates the underlying memory to prevent data leakage during
     * test set evaluation, and ensures contiguous memory for LAPACK routines.
     * * @param prop A Double value between 0 and 1 representing the proportion of training data.
     * @returns A tuple of perfectly isolated (Train, Test) Batched datasets.
     */
    override def split(prop: Double): ZIO[Any, Nothing, (BatchedDataset, BatchedDataset)] = 
        require(prop > 0.0 && prop < 1.0, "Split proportion must be between 0.0 and 1.0")
        
        val splitRow = (dataset.rows() * prop).toInt
        
        val (trainBatch, testBatch) = dataset match {
            case Batch.Labeled(x, y) =>
                (
                    Batch.Labeled(x(0 until splitRow, ::).copy, y(0 until splitRow, ::).copy),
                    Batch.Labeled(x(splitRow until x.rows, ::).copy, y(splitRow until y.rows, ::).copy)
                )
            case Batch.Unlabeled(x) =>
                (
                    Batch.Unlabeled(x(0 until splitRow, ::).copy),
                    Batch.Unlabeled(x(splitRow until x.rows, ::).copy)
                )
        }
        
        val batchSize = if (indexes.nonEmpty) indexes.head.length else 1
        
        for {
            trainSplit <- BatchedDataset.fromBatch(batchSize, trainBatch)
            testSplit  <- BatchedDataset.fromBatch(batchSize, testBatch)
        } yield (trainSplit, testSplit)

    /**
     * When called it obtains the current index from the Ref counter, obtains the corresponding range
     * and returns the data batch. Notice that getAndUpdate is used, which means that once the value is
     * pulled out of the Ref it is also incremented by 1. What happens when the index increases out of range?
     * This is dealt by the lift function: if the index is out of bounds None is returned. 
    */
    override def next(): ZIO[Any, Nothing, Option[Batch]] = for {
        idx   <- ref.getAndUpdate(_ + 1) 
        batch <- ZIO.fromOption(indexes.lift(idx)).map { range =>
            dataset match
                case Batch.Unlabeled(x) => 
                    Batch.Unlabeled(x(range, ::).toDenseMatrix)
                case Batch.Labeled(x, y) => 
                    Batch.Labeled(x(range, ::).toDenseMatrix, y(range, ::).toDenseMatrix)
                }.option 
            } yield batch

    /**
     * Checks the counter and returns true is the bacthes have not been exhausted.
     * 
    */
    def hasNext():ZIO[Any, Nothing, Boolean] = ref.get.map(idx => idx < indexes.length)

    def reset():UIO[Unit] = ref.update(_ => 0)

   
    /**
     * Every time the function is called a new finite stream is created. When the stream
     * is completely consumed during the training process this means a single training epoch has 
     * been completetd, a full sweep through the dataset. 
     * 
     * */
    override def stream():ZStream[Any, Nothing, Batch] = 
        val indexStream = ZStream.fromIterable(indexes)
        indexStream.map { range =>
            dataset match
                case Batch.Unlabeled(x) => 
                    Batch.Unlabeled(x(range, ::).toDenseMatrix)
                case Batch.Labeled(x, y) => 
                    Batch.Labeled(x(range, ::).toDenseMatrix, y(range, ::).toDenseMatrix)
        }


class FullDataset private(override val dataset:Batch) extends Dataset:

    //Every time next is called the full dataset is provided
    override def next(): ZIO[Any, Nothing, Option[Batch]] = ZIO.succeed(Some(dataset))

    /**
     * It might seem strange that we copy but there is a subtle reason behind this. In Breeze,
     * the underlying matrix library, slicing returns a view to the original matrix.
     * This data will eventually be passed to the optimizer, which in turn will hand it 
     * to the native implementation (BLAS/LAPACK) to do the computations. These libraries
     * rely heavily on contiguous memory blocks, and a view might not necessarily constitute one.
     * This will incur a heavy performance penalty to the math computations and may even crash 
     * the computations. The copy operation ensures a contiguous block of memory will be used.
     * 
    */
    override def split(prop: Double): ZIO[Any, Nothing, (FullDataset, FullDataset)] = 

        dataset match
            case Batch.Labeled(x, y) => 
                val n = x.rows
                val splitIndex = (prop * n).toInt
                val (xTrain, yTrain) = (x(0 until splitIndex, ::).copy, y(0 until splitIndex, ::).copy)
                val (xTest, yTest)   = (x(splitIndex until n, ::).copy, y(splitIndex until n, ::).copy)
                ZIO.succeed((FullDataset(Batch.Labeled(xTrain, yTrain)), FullDataset(Batch.Labeled(xTest, yTest))))
            case Batch.Unlabeled(x) => 
                val n = x.rows
                val splitIndex = (prop * n).toInt
                val (xTrain, xTest) = (x(0 until splitIndex, ::).copy, x(splitIndex until n, ::).copy)
                ZIO.succeed((FullDataset(Batch.Unlabeled(xTrain)), FullDataset(Batch.Unlabeled(xTest))))

    override def stream():ZStream[Any, Nothing, Batch] = 
        ZStream.from(dataset)

    /**
     * A function to convert a full dataset to a batched dataset. 
     * 
    */
    def batched(batchSize:Int=1):ZIO[Any, Nothing, BatchedDataset] = 
        BatchedDataset.fromBatch(batchSize, this.dataset)

        
object FullDataset:

    def fromBatch(batch:Batch):ZIO[Any, Nothing, FullDataset] = ZIO.succeed(FullDataset(batch))

    def live(x:DenseMatrix[Double], yOpt:Option[DenseMatrix[Double]] = None):ZLayer[Any, Nothing, FullDataset] = 

        val dataset = yOpt match {
            case Some(y) => FullDataset(Batch.Labeled(x, y))
            case None => FullDataset(Batch.Unlabeled(x))
        }
        ZLayer.fromZIO(ZIO.succeed(dataset))

    def live(batch:Batch):ZLayer[Any, Nothing, FullDataset] = ZLayer.fromZIO(ZIO.succeed(FullDataset(batch)))


object BatchedDataset:

    def fromBatch(batchSize:Int, batch:Batch):ZIO[Any, Nothing, BatchedDataset] = for {
        ranges <- ZIO.succeed((0 until batch.rows()).grouped(batchSize).toSeq)
        ref <- Ref.make(0)
    } yield BatchedDataset(ranges, ref, batch)

    def live(batchSize:Int, x:DenseMatrix[Double], yOpt:Option[DenseMatrix[Double]] = None):ZLayer[Any, Nothing, BatchedDataset] = 
        
        val ranges:Seq[Range] = (0 until x.rows).grouped(batchSize).toSeq
        
        val dataset = yOpt match {
            case Some(y) => Ref.make(0).map(ref => BatchedDataset(ranges, ref, Batch.Labeled(x, y)))
            case None => Ref.make(0).map(ref => BatchedDataset(ranges, ref, Batch.Unlabeled(x)))
        }

        ZLayer.fromZIO(dataset)

    def stream(batchSize: Int, x: DenseMatrix[Double], yOpt: Option[DenseMatrix[Double]] = None): ZStream[Any, Nothing, Batch] = 

        val ranges = (0 until x.rows by batchSize).map(start => start until math.min(start + batchSize, x.rows))

        ZStream.fromIterable(ranges).map { range =>
            yOpt match {
                case Some(y) => 
                    Batch.Labeled(x(range, ::).toDenseMatrix, y(range, ::).toDenseMatrix)
                case None => 
                    Batch.Unlabeled(x(range, ::).toDenseMatrix)
            }
        }