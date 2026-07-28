package rezervoir

import zio.*
import breeze.linalg.DenseMatrix
import zio.stream.*

/**
 * The Optimizer trait defines the contract for training algorithms in the rezervoir ecosystem.
 * Because Echo State Networks (ESNs) keep their input and reservoir weights fixed, the optimizer's 
 * sole responsibility is to fit the weights of the Readout layer.
 *
 * @tparam D The specific type of Dataset (e.g., FullDataset, BatchedDataset) the optimizer requires.
 */
sealed trait Optimizer[D <: Dataset : Tag]:
    
    /**
     * Executes a single training step (or a full closed-form solution, depending on the implementation)
     * using the dataset provided via the ZIO environment.
     */
    protected def step[M <: Model[?, ?, ?, ?]](model: M, refOpt:Option[Ref[Int]]): ZIO[D, Nothing, M]
    
    /**
     * Executes the training loop for a specified number of iterations/epochs
     * using the dataset provided via the ZIO environment.
     */
    def fit[M <: Model[?, ?, ?, ?]](model: M, nIter: Int): ZIO[D, Nothing, M]
    
    /**
     * Convenience method to inject the dataset directly as a parameter rather than relying 
     * on the ZIO environment. Internally constructs the ZLayer and delegates to the environment-based fit.
     */
    def fit[M <: Model[?, ?, ?, ?]](model: M, nIter: Int, dataset: D): ZIO[Any, Nothing, M] = 
        val env = ZLayer.succeed(dataset)
        this.fit(model, nIter).provideLayer(env) 
        
    /**
     * Convenience method to inject the dataset directly for a single training step.
     */
    protected def step[M <: Model[?, ?, ?, ?]](model: M, dataset: D, refOpt:Option[Ref[Int]]): ZIO[Any, Nothing, M] = 
        val env = ZLayer.succeed(dataset)
        this.step(model, refOpt).provideLayer(env)

    
/**
 * Ridge Regression (Tikhonov Regularization) Optimizer.
 * * This is the gold-standard solver for Echo State Networks. Because the Readout layer is purely linear, 
 * Ridge regression can find the mathematically optimal weights in a single, closed-form step using 
 * native C/Fortran LAPACK routines (via Breeze's `\` operator).
 * * Ridge inherently requires the entire state matrix in memory at once, which is why it strictly 
 * bounds its dataset requirement to `FullDataset`.
 * * @param alpha   The L2 regularization penalty. Higher values prevent overfitting by shrinking Readout weights.
 * @param washout The number of initial time steps to discard before computing the loss. This is critical 
 * in Reservoir Computing to allow the chaotic reservoir to "warm up" and forget its 
 * initial state of zeros (Transient Phase).
 */
class Ridge(val alpha: Double, val washout: Int = 100) extends Optimizer[FullDataset]:
    
    override def step[M <: Model[?, ?, ?, ?]](model: M, refOpt:Option[Ref[Int]]): ZIO[FullDataset, Nothing, M] = for {
        dataset  <- ZIO.service[FullDataset]
        cascade  = model.getCascade
        batchOpt <- dataset.next()

        _ <- batchOpt match {
            case Some(batch) =>
                ZIO.succeed {
                    // Extract the final row of the training data to seed generative forecasting
                    batch match
                        case Batch.Labeled(xMat, _) if xMat.rows > 0 =>
                            val lastRow = xMat(xMat.rows - 1, ::).t.toDenseMatrix
                            model.setSeed(lastRow) 
                        case _ => ()

                    cascade.foreach {
                        case Sequential(layerA, readout: Readout[?,?]) =>
                            val z = layerA.forward(batch)
                            z match
                                case Batch.Labeled(z, y) =>
                                    // Safely drop the transient warmup phase (Washout)
                                    val safeWashout = math.min(washout, z.rows)
                                    val zWashed = if safeWashout > 0 then z(safeWashout until z.rows, ::) else z
                                    val yWashed = if safeWashout > 0 then y(safeWashout until y.rows, ::).copy else y
                                    
                                    // Append a bias column of 1.0s to the state matrix
                                    val zWithBias = DenseMatrix.horzcat(DenseMatrix.ones[Double](zWashed.rows, 1), zWashed)
                                    
                                    // Construct the Identity matrix for L2 Regularization, keeping bias unpenalized (I(0,0) = 0)
                                    val I = DenseMatrix.eye[Double](zWithBias.cols) * alpha
                                    I(0, 0) = 0.0
                                    
                                    // Solve the normal equations: W = (Z^T * Z + alpha * I)^-1 * (Z^T * Y)
                                    val A = (zWithBias.t * zWithBias) + I
                                    val b = zWithBias.t * yWashed
                                    val wTranspose = A \ b // LAPACK optimal solver

                                    // Update the readout weights
                                    readout.update(wTranspose.t)
                                case Batch.Unlabeled(_) => ()
                        case _ => ()
                    }
                }
            case _ => ZIO.unit
        }
    } yield model

    /**
     * Because Ridge regression is a closed-form analytical solver, it finds the global optimum 
     * in a single step. Therefore, `fit` simply delegates to `step` and ignores `nIter`.
     */
    override def fit[M <: Model[?, ?, ?, ?]](model: M, nIter: Int = 1): ZIO[FullDataset, Nothing, M] = step(model, None)


/**
 * Gradient Descent Optimizer for streaming or memory-bound data.
 * * Unlike Ridge, Gradient Descent iteratively updates the Readout weights using backpropagation 
 * through time. This is strictly required when the dataset is too large to fit into RAM, requiring 
 * a `BatchedDataset` to stream chunks of data through the network.
 * * @param lr           The learning rate dictating the step size for weight updates.
 * @param alpha        The L2 regularization penalty applied to the gradients.
 * @param beta         The momentum term
 * @param totalWashout The total number of initial time steps to discard across the incoming stream 
 * to bypass the reservoir's transient warmup phase.
 * 
 * Theoretically the learning rate lies somewhere between 0.0 and 1.0. In practice, the learning rate should be set 
 * to a small value, typically less than 0.1. If a large value (~0.4) is specified then if a sufficient number 
 * of epochs is specified, the step updates will keep overshooting the minimum eventually causing the gradients to 
 * explode. 
 * 
 * The step function is a pass through the entire dataset, in machine learning terminology an epoch.
 * 
 */
class GradientDescent(lr: Double, alpha: Double, val beta: Double = 0.9, val totalWashout: Int = 0) extends Optimizer[BatchedDataset]:

    override def step[M <: Model[?, ?, ?, ?]](model: M, dataset: BatchedDataset, refOpt:Option[Ref[Int]]): ZIO[Any, Nothing, M] = for {
        cascade    <- ZIO.succeed(model.getCascade)
        
        // Stateful reference to track how much of the washout requirement has been satisfied across batches
        washoutRef  = refOpt.get
        
        _ <- dataset.stream().runForeach {
            case Batch.Labeled(x, y) => 
                for {
                    currentWashout <- washoutRef.get
                    
                    _ <- ZIO.succeed {
                        // Continuously update the final row seed so the model is ready for generative forecasting
                        if x.rows > 0 then
                            val lastRow = x(x.rows - 1, ::).t.toDenseMatrix
                            model.setSeed(lastRow)
                    }
                    
                    _ <- ZIO.succeed {
                        cascade.foreach {
                            case Sequential(layerA, readout: Readout[?,?]) =>
                                val zStates = layerA.forward(Batch.Labeled(x, y)) 
                                
                                zStates match {
                                    case Batch.Labeled(z, _) =>
                                        val rowsToDrop = math.min(currentWashout, z.rows)
                                        
                                        // Only apply backprop if there is data left after the washout drop
                                        if rowsToDrop < z.rows then
                                            val zWashed = z(rowsToDrop until z.rows, ::)
                                            val yWashed = y(rowsToDrop until y.rows, ::)
                                            
                                            val n = zWashed.rows.toDouble
                                            val zWithBias = DenseMatrix.horzcat(DenseMatrix.ones[Double](zWashed.rows, 1), zWashed)
                                            
                                            // Forward pass: yHat = Z * W^T
                                            val yhat = zWithBias * readout.w.t
                                            val error = yhat - yWashed
                                            
                                            // Compute Base Gradients
                                            val baseGrad = (error.t * zWithBias) * (1.0 / n)
 
                                            // Compute Regularization Gradients (excluding the bias term)
                                            val regGrad = readout.w * alpha
                                            regGrad(::, 0) := 0.0 
                                            
                                            // Apply update rule: W = W - lr * (Grad + RegGrad)
                                            val grad = baseGrad + regGrad

                                            // Gradient clipping
                                            val clippedGrad = grad.map(g => math.max(-1.0, math.min(1.0, g)))



                                            readout.w :-= (clippedGrad * lr)
                                            
                                    case Batch.Unlabeled(_) => ()
                                }
                            case _ => ()
                        }
                    }
                    
                    // Decrement the remaining washout counter
                    _ <- washoutRef.update(w => math.max(0, w - x.rows))
                    
                } yield ()
                
            case Batch.Unlabeled(_) => ZIO.unit
        }
    } yield model
    
    override def step[M <: Model[?, ?, ?, ?]](model: M, refOpt:Option[Ref[Int]]): ZIO[BatchedDataset, Nothing, M] = for {
        dataset      <- ZIO.service[BatchedDataset]
        trainedModel <- step(model, dataset, refOpt)
    } yield trainedModel

    /**
     * Executes the streaming training loop for a specified number of epochs (nIter).
     */
    override def fit[M <: Model[?, ?, ?, ?]](model: M, nIter: Int, dataset: BatchedDataset): ZIO[Any, Nothing, M] = 
        for {
            washoutRef <- Ref.make(totalWashout)
            //velocityRef <- Ref.make[Option[DenseMatrix[Double]]](None)
            model <- ZStream.fromIterable(0 until nIter).runFoldZIO(model){ case (m, _) => step(m, dataset, Some(washoutRef)) }
        } yield model

    /**
     * Executes the streaming training loop using the dataset provided by the ZIO environment.
     */
    override def fit[M <: Model[?, ?, ?, ?]](model: M, nIter: Int): ZIO[BatchedDataset, Nothing, M] = for {
        washoutRef <- Ref.make(totalWashout)
        model <- ZStream.fromIterable(0 until nIter).runFoldZIO(model){ case (m, _) => step(m, Some(washoutRef)) } 
    } yield model
        