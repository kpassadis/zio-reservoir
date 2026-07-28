package rezervoir

import zio.*
import zio.stream.*
import breeze.linalg.DenseMatrix
import scala.compiletime.ops.int.+ as TypeAdd 
import Utils.*
import rezervoir.Layer.reservoirLayer

trait Network[Out <: Int]:

    def fit[D <: Dataset](optimizer:Optimizer[D], dataset:D, nIter:Int): ZIO[Any, Nothing, Unit] 

    def predict(batch:Batch): ZIO[Any, Nothing, Batch] 

trait MultiGraphNetwork[Out <: Int,  M <: Model[?, Out, ?, ?]] extends Network[Out]:

    val graph: Ref[Seq[M]]


trait SingleGraphNetwork[Out <: Int, M <: Model[?, Out, ?, ?]] extends Network[Out]:

    val graph: Ref[M]

    override def fit[D <: Dataset](optimizer:Optimizer[D], dataset:D, nIter:Int): ZIO[Any, Nothing, Unit] = for {
        model <- graph.get
        trainedModel <- optimizer.fit(model, nIter, dataset)
        _ <- graph.update(_ => trainedModel)
    } yield ()

    override def predict(batch: Batch): ZIO[Any, Nothing, Batch] = for {
        model <- graph.get
        preds = model.predict(batch)
    } yield preds


case class ForestESN[In <: Int, Res <: Int, Out <: Int](
    ensemble: Seq[VanillaESN[In, Res, Out]], 
    graph: Ref[Seq[VanillaESN.Graph[In, Res, Out]]], 
    blockSize: Int, 
    seed: Long
) extends MultiGraphNetwork[Out, VanillaESN.Graph[In, Res, Out]]: 

    /**
     * Train the forest using Moving Block Bootstrap. Each model in the ensemble 
     * receives a slightly different sampled timeline to build variance.
     * @param optimizer The generic optimizer instance.
     * @param dataset The single, unbroken training timeline (Full or Batched).
     * @param nIter The number of training epochs.
     */
    override def fit[D <: Dataset](
        optimizer: Optimizer[D], 
        dataset: D, 
        nIter: Int = 1
    ): ZIO[Any, Nothing, Unit] = for {
        
        // 1. Extract the raw mathematical batch from whatever D is
        originalBatch <- ZIO.succeed(dataset.dataset) 
        
        // 2. Perform the Moving Block Bootstrap to get Seq[Batch]
        batches <- ZIO.succeed(originalBatch.movingBlockBootstrap(blockSize, ensemble.length, seed = seed))
        
        trainedModels <- ZIO.foreachPar(ensemble.zipWithIndex) { case (vanilla, index) =>
            for {
                model <- vanilla.graph.get
                
                // 3. Dynamically wrap the raw Batch back into the specific D subtype!
                data <- dataset match {
                    case _: FullDataset => 
                        FullDataset.fromBatch(batches(index))
                        
                    case b: BatchedDataset => 
                        // If they passed a BatchedDataset, intelligently infer their chosen batch size
                        val batchSize = if (b.indexes.nonEmpty) b.indexes.head.length else 1
                        BatchedDataset.fromBatch(batchSize, batches(index))
                        
                    case _ => 
                        ZIO.die(new IllegalArgumentException("Unsupported Dataset type provided to ForestESN"))
                }
                
                // 4. Safe cast to D to satisfy the generic optimizer bounds
                trainedModel <- optimizer.fit(model, nIter, data.asInstanceOf[D])
            } yield trainedModel
        }
        
        _ <- graph.update(_ => trainedModels)
        
    } yield ()
       

    /**
         * A quick note on the implementation: Why do we use attamptBlocking instead of attempt?
         * When predict(batch) is called on the VanillaESN it does not yield execution.
         * It will execute a series of DenseMatrix multiplications, and under the hood the
         * Breeze library will delegate the work to BLAS/LAPACK. These operations are 
         * synchronous and heavy. Once a thread starts a matrix multiplication it 
         * will be blocked i.e. it will not be available for any other computation until the
         * mathematical computation has been completed.
         * By default ZIO runs code on its Compute Thread Pool. This pool is highly optimized
         * for asynchronous, non-blocking code and is stricktly sized to the number of 
         * CPU cores available. Assume that attempt was used unsitead of attemptBlocking.
         * Here is what would happen:
         * 1. ZIO would spawn n (number of ESN models in forest) lightweight fibers.
         * 2. The first fibers would grab all the available CPUs and would start performing matrix math. 
         * 3. The ZIO runtime would freeze. 
         * 
         * Using the attemptBlocking would send everyting off to a second pool used ZIO
         * called the Blocking pool. This is explicitly designed for legacy JDBC I/O
         * queries, heavy file I/O or synchronous native CPU bound tasks (like this case).
         * 
        */
    override def predict(batch: Batch): ZIO[Any, Nothing, Batch] = for {
        models <- graph.get

        // Dispatch matrix math to the blocking pool to prevent thread starvation
        predictions <- ZIO.foreachPar(models) { model => 
            ZIO.attemptBlocking(model.predict(batch, resetState = true)).orDie
        }
        
        predMatrices = predictions.map {
            case Batch.Unlabeled(x) => x
            case Batch.Labeled(x, _) => x 
        }

        distributionMatrix = ZIO.succeed(DenseMatrix.horzcat(predMatrices: _*))
        
        aggregated <- distributionMatrix.map { dist =>
            val timeSteps = dist.rows
            val outDim    = 1 
            
            val meanMat  = DenseMatrix.zeros[Double](timeSteps, outDim)
            
            for t <- 0 until timeSteps do
                val ensembleRow = dist(t, ::).t.toArray 
                
                // Calculate Mean only (Sorting is no longer required!)
                meanMat(t, 0) = ensembleRow.sum / ensembleRow.length
            
            Batch.Unlabeled(meanMat)
        }

    } yield aggregated


object ForestESN:

    type Graph[In <: Int, Res <: Int, Out <: Int] = 
        Seq[VanillaESN.Graph[In, Res, Out]]

    def build[In <: Int, Res <: Int, Out <: Int](size: Int, blockSize: Int, seed: Long)(using ValueOf[In], ValueOf[Res], ValueOf[Out]): ZIO[Any, Nothing, ForestESN[In, Res, Out]] = for {
        
        ensemble <- ZIO.collectAllPar(List.fill(size)(VanillaESN.build[In, Res, Out]))
        initialGraphs <- ZIO.foreach(ensemble)(esn => esn.graph.get)
        graphRef <- Ref.make[Seq[VanillaESN.Graph[In, Res, Out]]](initialGraphs)
        
    } yield ForestESN(ensemble, graphRef, blockSize, seed)


case class VanillaESN[In <: Int, Res <: Int, Out <: Int](
    input: Input[In, Res],
    reservoir: Reservoir[Res],
    readout: Readout[Res, Out],
    graph: Ref[VanillaESN.Graph[In, Res, Out]]
) extends SingleGraphNetwork[Out, VanillaESN.Graph[In, Res, Out]]
    

object VanillaESN:

    type Graph[In <: Int, Res <: Int, Out <: Int] = 
        Sequential[In, Res, Out, Sequential[In, Res, Res, Input[In, Res], Reservoir[Res]], Readout[Res, Out]]
    
    def build[In <: Int, Res <: Int, Out <: Int](using ValueOf[In], ValueOf[Res], ValueOf[Out]): ZIO[Any, Nothing, VanillaESN[In, Res, Out]] = 
        for {
            inLayer  <- Input.typed[In, Res]
            resLayer <- Reservoir.typed[Res]
            outLayer <- Readout.typed[Res, Out]
            graph    <- Ref.make(inLayer >>> resLayer >>> outLayer)
        } yield VanillaESN[In, Res, Out](inLayer, resLayer, outLayer, graph)


case class SkipESN[In <: Int, Res <: Int, Out <: Int](
    input:Input[In, Res],
    reservoir:Reservoir[Res],
    readout:Readout[In TypeAdd Res, Out],
    graph: Ref[SkipESN.Graph[In, Res, Out]]
) extends SingleGraphNetwork[Out, SkipESN.Graph[In, Res, Out]]


object SkipESN:

    type Graph[In <: Int, Res <: Int, Out <: Int] = 
        Sequential[In, In TypeAdd Res, Out, Concat[In, In, Res, Identity[In], Sequential[In, Res, Res, Input[In, Res], Reservoir[Res]]], Readout[In TypeAdd Res, Out]]
        //Sequential[Int, In TypeAdd Res, Out, Concat[Int, In, Res, Identity[In], Sequential[In, Res, Res, Input[In, Res], Reservoir[Res]]], Readout[In TypeAdd Res, Out]]

    def build[In <: Int, Res <: Int, Out <: Int](using ValueOf[In], ValueOf[Res], ValueOf[Out]) = 
        given ValueOf[In TypeAdd Res] = new ValueOf[In TypeAdd Res](
            (valueOf[In] + valueOf[Res]).asInstanceOf[In TypeAdd Res]
        )

        for {
            identity  <- Identity.typed[In]          // Accepts 'In', Outputs 'In'
            input     <- Input.typed[In, Res]        // Accepts 'In', Outputs 'Res'
            reservoir <- Reservoir.typed[Res]
    
            // Right branch: Accepts 'In', Outputs 'Res'
            inputToRes = input >>> reservoir
    
            // Concat block: Accepts 'In', Outputs 'In + Res'
            concat = identity + inputToRes
    
            // Readout perfectly catches the 'In + Res' state!
            readout <- Readout.typed[In TypeAdd Res, Out]
    
            // The compiler proves (In + Res) == (In TypeAdd Res)
            graph <- Ref.make(concat >>> readout)
        } yield SkipESN(input, reservoir, readout, graph)


case class DeepESN[In <: Int, Res <: Int, Out <: Int](
    input: Input[In, Res],
    deepReservoirStack: rezervoir.Layer[Res, Res], 
    readout: Readout[Res, Out],
    graph: Ref[DeepESN.Graph[In, Res, Out]]
) extends SingleGraphNetwork[Out, DeepESN.Graph[In, Res, Out]]
    

object DeepESN:

    type Graph[In <: Int, Res <: Int, Out <: Int] = 
        Sequential[In, Res, Out, Sequential[In, Res, Res, Input[In, Res], rezervoir.Layer[Res, Res]], Readout[Res, Out]]

    def build[In <: Int, Res <: Int, Out <: Int](depth: Int, applyDropout:Boolean=false)(using ValueOf[In], ValueOf[Res], ValueOf[Out]): ZIO[Any, Nothing, DeepESN[In, Res, Out]] = 
        
        for {
            inLayer  <- Input.typed[In, Res]
            outLayer <- Readout.typed[Res, Out]

            tuple <- ZIO.ifZIO(ZIO.succeed(applyDropout))(
                onFalse = for {
                    reservoirs <- ZIO.collectAll(List.fill(depth)(Reservoir.typed[Res]))
                    deepStack = reservoirs.tail.foldLeft[rezervoir.Layer[Res, Res]](reservoirs.head) { (acc, nextRes) =>
                        Sequential[Res, Res, Res, rezervoir.Layer[Res, Res], Reservoir[Res]](acc, nextRes)
                    }
                    graph <- Ref.make(Sequential(Sequential(inLayer, deepStack), outLayer))
                } yield (deepStack, graph),
                onTrue = for {
                    reservoirsWithDropOut <- ZIO.collectAll(List.tabulate(depth)(i => for {
                        reservoir <- Reservoir.typed[Res]
                        dropout <- Dropout.typed[Res](0.5, 42L + i) 
                    } yield reservoir >>> dropout))
                    deepStackWithDropout = reservoirsWithDropOut.tail.foldLeft[rezervoir.Layer[Res, Res]](reservoirsWithDropOut.head) { (acc, nextRes) =>
                        Sequential[Res, Res, Res, rezervoir.Layer[Res, Res], Sequential[Res, Res, Res, Reservoir[Res], Dropout[Res]]](acc, nextRes)
                    }
                    graph <- Ref.make(Sequential(Sequential(inLayer, deepStackWithDropout), outLayer))
                } yield (deepStackWithDropout, graph)   
            )

            (stack, graph) = tuple
            
        } yield DeepESN[In, Res, Out](inLayer, stack, outLayer, graph)

object NetworkPipeline:

    def trainAndDeploy[Out <: Int : Tag, D <: Dataset : Tag]: ZIO[MinMaxScaler & D & Network[Out] & Optimizer[D], Nothing, Network[Out]] = for {
        scaler    <- ZIO.service[MinMaxScaler]
        dataset   <- ZIO.service[D]
        optimizer <- ZIO.service[Optimizer[D]]
        network <- ZIO.service[Network[Out]]
        _         <- ZIO.logInfo(s"Initiating Network Training...")
        _         <- network.fit(optimizer, dataset, nIter = 1)
        _         <- ZIO.logInfo(s"Training Complete. Network ready for inference.")

    } yield network 