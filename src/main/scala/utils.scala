package rezervoir

import zio.*
import zio.json.*
import breeze.linalg.DenseMatrix
import scala.reflect.ClassTag
import breeze.storage.Zero
import breeze.linalg.*
import breeze.linalg.min
import breeze.linalg.max

/**
 * A collection of utility functions and typeclass instances required to support
 * the broader rezervoir ecosystem. This includes ZIO runtime bridges, JSON serialization
 * for native C/Fortran arrays (Breeze), and data normalization tools.
 */
object Utils:
    
    /**
     * Extension method to provide a synchronous bridge for ZIO effects.
     * * In a pure functional application, effects should be evaluated at the very edge 
     * of the world (the main method). However, when interfacing with legacy code, 
     * REPL environments, or writing specific tests, it is often necessary to strictly 
     * evaluate a ZIO effect into its underlying value.
     */
    extension [E, A](zio: ZIO[Any, E, A])
        /**
         * Safely executes the ZIO effect on the default runtime and unwraps the result.
         * @return The strictly evaluated value of type A.
         * @throws FiberFailure if the underlying ZIO effect fails.
         */
        def unsafeRun(): A =
            Unsafe.unsafe { implicit unsafe => Runtime.default.unsafe.run(zio).getOrThrowFiberFailure()}


    /**
     * Generic JSON Encoder for Breeze DenseMatrix.
     * * Breeze matrices are backed by flattened, column-major 1D arrays for LAPACK performance.
     * This encoder safely unwraps them into human-readable 2D nested JSON arrays (Seq[Seq[A]])
     * to allow for easy inspection, debugging, and persistence of Reservoir states and Readout weights.
     */
    given denseMatrixEncoder[A](using JsonEncoder[A], ClassTag[A]): JsonEncoder[DenseMatrix[A]] =
        JsonEncoder[Seq[Seq[A]]].contramap { x =>
            (0 until x.rows).map { r =>
                x(r, ::).t.toArray.toIndexedSeq
            }
        }

    /**
     * Generic JSON Decoder for Breeze DenseMatrix.
     * * Reads a nested JSON array (Seq[Seq[A]]) and reconstructs the heavily optimized, 
     * column-major 1D flat array required by Breeze's DenseMatrix. The safest 
     * way to handle a matrix is to pre allocate it and then fill it. 
     */
    given denseMatrixDecoder[A](using JsonDecoder[A], ClassTag[A], Zero[A]): JsonDecoder[DenseMatrix[A]] = 
    JsonDecoder[Seq[Seq[A]]].map { seq => 
        val rows = seq.length
        val cols = if (rows > 0) seq.head.length else 0

        // Pre-allocate a safe, contiguous column-major matrix
        val mat = DenseMatrix.zeros[A](rows, cols)
        
        // Safely populate it exactly as it appeared in the JSON
        for r <- 0 until rows; c <- 0 until cols do
            mat(r, c) = seq(r)(c)
            
        mat
    }

    /**
     * MinMaxScaler transforms data features by scaling them to a given range, strictly [-1, 1].
     * * In the context of Echo State Networks (ESNs), inputs must typically be scaled between 
     * [-1, 1] so that they effectively interact with the `tanh` activation function of the 
     * reservoir neurons. Unscaled data with large bounds will instantly saturate the `tanh` 
     * nodes to exactly -1 or 1, destroying the network's non-linear dynamics and gradient flow.
     */
    case class MinMaxScaler(
        xMin: Option[DenseMatrix[Double]] = None, 
        xMax: Option[DenseMatrix[Double]] = None,
        yMin: Option[DenseMatrix[Double]] = None,
        yMax: Option[DenseMatrix[Double]] = None):
    
        /** Core scaling logic: maps a raw column vector into the [-1, 1] domain. */
        private def scale(mat: DenseMatrix[Double], minVals: DenseMatrix[Double], maxVals: DenseMatrix[Double]): DenseMatrix[Double] =
            val scaled = DenseMatrix.zeros[Double](mat.rows, mat.cols)
            for c <- 0 until mat.cols do
                val cMin = minVals(0, c)
                val cMax = maxVals(0, c)
                val range = if (cMax - cMin) == 0 then 1.0 else (cMax - cMin) // Prevent division by zero
                scaled(::, c) := ((mat(::, c) - cMin) / range) * 2.0 - 1.0
            scaled

        /** Core unscaling logic: maps a [-1, 1] column vector back into its original raw domain. */
        private def unscale(mat: DenseMatrix[Double], minVals: DenseMatrix[Double], maxVals: DenseMatrix[Double]): DenseMatrix[Double] =
            val unscaled = DenseMatrix.zeros[Double](mat.rows, mat.cols)
            for c <- 0 until mat.cols do
                val cMin = minVals(0, c)
                val cMax = maxVals(0, c)
                val range = cMax - cMin
                unscaled(::, c) := ((mat(::, c) + 1.0) / 2.0) * range + cMin
            unscaled


        /**
         * Transforms features (and targets if present in the batch) to the [-1, 1] range.
         * * @param batch The batch to scale. 
         * @return A safely scaled batch ready to be piped into the ESN's `forward` function.
         */
        def transform(batch: Batch): Batch = batch match
            case Batch.Labeled(x, y) =>
                val targetMin = yMin.get
                val targetMax = yMax.get
                Batch.Labeled(scale(x, xMin.get, xMax.get), scale(y, targetMin, targetMax))
                
            case Batch.Unlabeled(x) =>
                // In the forward pass, Unlabeled batches represent raw Input Features (X)
                // before they enter the model. Therefore, we scale using X statistics.
                Batch.Unlabeled(scale(x, xMin.get, xMax.get))


        /**
         * Reverts a scaled batch back to its original mathematical domain.
         * * @param batch The batch containing scaled predictions.
         * @return A batch with absolute real-world values, suitable for metric evaluation (e.g., RMSE, MAPE).
         */
        def inverseTransform(batch: Batch): Batch = batch match
            case Batch.Labeled(x, y) =>
                require(yMin.isDefined && yMax.isDefined, "Cannot unscale targets: Scaler was fit on Unlabeled data.")
                Batch.Labeled(unscale(x, xMin.get, xMax.get), unscale(y, yMin.get, yMax.get))
        
            case Batch.Unlabeled(predictions) =>
                require(yMin.isDefined && yMax.isDefined, "Cannot unscale predictions: Scaler was fit on Unlabeled data.")
                Batch.Unlabeled(unscale(predictions, yMin.get, yMax.get))


    object MinMaxScaler:

        /** Instantiates an empty, untrained scaler wrapped in a ZLayer. */
        def live:ZLayer[Any, Nothing, MinMaxScaler] = 
            ZLayer.fromZIO(ZIO.succeed(MinMaxScaler()))
        
        /**
         * Evaluates a dataset batch to determine the column-wise minimums and maximums, 
         * locking them into state for future transformations.
         * * @param batch The dataset to learn scaling parameters from.
         * @return A fitted MinMaxScaler instance containing the min/max bounds.
         */
        def fit(batch: Batch): MinMaxScaler = batch match
            case Batch.Labeled(x, y) =>
                MinMaxScaler(
                    xMin = Some(getMins(x)), xMax = Some(getMaxes(x)),
                    yMin = Some(getMins(y)), yMax = Some(getMaxes(y))
                )
            case Batch.Unlabeled(x) =>
                MinMaxScaler(xMin = Some(getMins(x)), xMax = Some(getMaxes(x)))
                
        /** Helper function to securely extract the absolute minimum of each matrix column. */
        private def getMins(mat: DenseMatrix[Double]): DenseMatrix[Double] =
            val minVals = DenseMatrix.zeros[Double](1, mat.cols)
            for c <- 0 until mat.cols do minVals(0, c) = min(mat(::, c))
            minVals
            
        /** Helper function to securely extract the absolute maximum of each matrix column. */
        private def getMaxes(mat: DenseMatrix[Double]): DenseMatrix[Double] =
            val maxVals = DenseMatrix.zeros[Double](1, mat.cols)
            for c <- 0 until mat.cols do maxVals(0, c) = max(mat(::, c))
            maxVals