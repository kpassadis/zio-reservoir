package rezervoir

import zio.* 
import zio.stream.* 
import breeze.linalg.*
import breeze.numerics.* 
import breeze.linalg.operators.DenseVectorSupportMethods
import breeze.stats.distributions.RandBasis
import breeze.stats.mean
import breeze.stats.DescriptiveStats.percentile
import scala.annotation.implicitNotFound
import scala.compiletime.constValue
import zio.json.*
import scala.compiletime.ops.int.+ as TypeAdd 
import Utils.{denseMatrixDecoder, denseMatrixEncoder}
import rezervoir.Batch.Labeled


/**
 * The Layer trait is the fundamental abstraction used to model the parts of an Echo State Network. 
 * The trait Layer takes two literal type parameters, In and Out, that represent the dimensional 
 * boundaries of the Layer, guaranteeing structural safety at compile time.
 */
sealed trait Layer[In <: Int, Out <: Int]:

    /**
     * The forward method takes an input data batch, propagates it through the model layers and returns a 
     * Batch that contains the predictions
     * @param x an input batch of data
     * @return a batch that contains the predictions of the model
     * 
    */
    def forward(x:Batch):Batch

    /**
     * An ESN model extends the Layer trait. A practical way to be able to distinguish between a composite layer and 
     * a layer which is a fundamental building block is to provide a function that returns a boolean indicating whether
     * a layer is composite or not.
    */
    def isComposite:Boolean = false

    def predict(x: Batch, resetState: Boolean = true): Batch = forward(x)

object Layer:
    
    given inputLayer[In <: Int & Singleton, Out <: Int & Singleton](using 
        evIn: ValueOf[In], 
        evOut: ValueOf[Out], 
        tag: Tag[Input[In, Out]]
    ): ZLayer[Any, Nothing, Input[In, Out]] = ZLayer.fromZIO(Input.typed[In, Out](using evIn, evOut))

    given reservoirLayer[In <: Int & Singleton](using 
        evIn: ValueOf[In], 
        tag: Tag[Reservoir[In]]
    ): ZLayer[Any, Nothing, Reservoir[In]] = ZLayer.fromZIO(Reservoir.typed[In](using evIn))

    given readoutLayer[In <: Int & Singleton, Out <: Int & Singleton](using 
        evIn: ValueOf[In], 
        evOut: ValueOf[Out], 
        tag: Tag[Readout[In, Out]]
    ): ZLayer[Any, Nothing, Readout[In, Out]] = ZLayer.fromZIO(Readout.typed[In, Out](using evIn, evOut))


/**
 * The Dropout layer has exactly the same functionality as in traditional neural networks. Its purpose is to randomly turn off
 * a portion of the units in a Reservoir layer (a Dropoutout layer should only be applied after a Reservoir layer).
 * The Dropout has a single type parameter Dim which is a subtype of Int. Notice that we implicitly bring into scope the ValueOf 
 * trait in order to be able to extract value of the Dim parameter.  
 * @param p the probability of turning off a unit
 * @param seed a long to be used for the random number generator  
 *  
*/
case class Dropout[Dim <: Int : ValueOf](p: Double, seed: Long = 42L) extends Layer[Dim, Dim]:

    require(p > 0.0 && p < 1.0)

    val dim = valueOf[Dim]
    private val gen = new scala.util.Random(seed)
    
    override def forward(batch: Batch): Batch = 
        batch match
            case Batch.Labeled(x, y) => 
                val scale = 1.0 / (1.0 - p) 
                
                val mask = DenseMatrix.tabulate(x.rows, x.cols) { case (i, j) => 
                    val r = gen.nextDouble()
                    if r > p then
                        scale * x(i, j)
                    else
                        0.0
                }
                
                Batch.Labeled(mask, y)
                
            case Batch.Unlabeled(x) => Batch.Unlabeled(x)

object Dropout:

    def typed[Dim <: Int](p:Double, seed:Long)(using ev: ValueOf[Dim]): ZIO[Any, Nothing, Dropout[Dim]] = 
        ZIO.succeed(Dropout[Dim](p, seed))
        

/**
 * An extension to the Layer trait to add support for layers that have learnable (or not) parameters 
 * like Reservoir or Readout.
 * 
*/
sealed trait ParameterizedLayer[In <: Int, Out <: Int] extends Layer[In, Out]:
    val w:DenseMatrix[Double]
    
    def shape(): (Int, Int) = (w.rows, w.cols)


class Identity[Dim <: Int : ValueOf]() extends Layer[Dim, Dim]:
    
    val in = valueOf[Dim]
    
    override def toString(): String = s"[ Identity ]"
    override def forward(x: Batch): Batch = x

object Identity:
    def typed[Dim <: Int](using ev: ValueOf[Dim]): ZIO[Any, Nothing, Identity[Dim]] = 
        ZIO.succeed(Identity[Dim]())

/**
 * The Input layer is a Parameterized layer and as such it has weights. 
 * 
*/
class Input[In <: Int : ValueOf, Out <: Int : ValueOf](override val w:DenseMatrix[Double]) extends ParameterizedLayer[In, Out]:

    val in = valueOf[In]
    val out = valueOf[Out]

    require(
        w.rows == out && w.cols == in + 1, 
        s"Input weight matrix w must have shape ($out, ${in + 1}) to match types [In=$in, Out=$out] with bias, but got (${w.rows}, ${w.cols})"
    )

    override def toString():String = s"[ Input (${shape()})]"
    override def forward(x: Batch): Batch = 

        def run(x:DenseMatrix[Double]):DenseMatrix[Double] =
            val ones = DenseMatrix.ones[Double](x.rows, 1)
        
            // xWithBias shape: [N x (in + 1)]
            val xWithBias = DenseMatrix.horzcat(ones, x)
        
            // Output shape: [N x (in + 1)] x [(in + 1) x out] = [N x out]
            val output = xWithBias * w.t 
        
            output
        
        x match 
            case Batch.Labeled(x, y) => 
                val yhat = run(x)
                Batch.Labeled(yhat, y)
            case Batch.Unlabeled(x) =>
                val yhat = run(x)  
                Batch.Unlabeled(yhat)

class Mean[Dim <: Int]() extends Layer[Dim, Dim]:

    override def forward(x: Batch): Batch = x match
        case Batch.Labeled(x, y) => 
            val rowMeans = mean(x, Axis._1)
            Batch.Labeled(rowMeans.toDenseMatrix, y)
        case Batch.Unlabeled(x) => 
            val rowMeans = mean(x, Axis._1)
            Batch.Unlabeled(rowMeans.toDenseMatrix)

object Mean:
    def typed[Dim <: Int : ValueOf]: ZIO[Any, Nothing, Mean[Dim]] = 
        ZIO.succeed(Mean[Dim]())



object Input:

    //Two different ways to do the same thing but the second is idiomatic to Scala 3, and looks more elegant.
    /*def typed[In <: Int, Out <: Int](using evIn: ValueOf[In], evOut: ValueOf[Out]) = 
        val in  = evIn.value
        val out = evOut.value
    
        val initializer = RandomInitializer(out, in + 1)
        for {
            weights <- initializer.generate()
        } yield Input[In, Out](weights)*/

    def typed[In <: Int : ValueOf, Out <: Int : ValueOf] = 
        val in = valueOf[In]
        val out = valueOf[Out]
        val initializer = RandomInitializer(out, in + 1)
        for {
            weights <- initializer.generate()
        } yield Input[In, Out](weights)

/**
 * The Reservoir layer 
 * 
*/
class Reservoir[Dim <: Int : ValueOf](
    override val w: DenseMatrix[Double],           // Recurrent spectral matrix [Out x Out]
    val activation: Activation, 
    val leak: Double
) extends ParameterizedLayer[Dim, Dim]:

    val dim = valueOf[Dim]

    require(w.rows == dim && w.cols == dim, s"Recurrent matrix w must be square ($dim x $dim)")

    override def toString():String = s"[ Reservoir (${shape()}) ]"
    
    // The persistent state vector of the reservoir layer. The state is a vector of
    // dimension [Dim x 1] and not [1 x Dim]. The reason is that the default layout
    // in Breeze is column major. To speed things up Breeze uses a BLAS implementation  
    // which is built in Fortran and Fortran is column - major (as opposed to other popularized numerical libraries like Numpy that are row major).
    private val state: DenseVector[Double] = DenseVector.zeros[Double](w.cols)

    def reset(): Unit = {
        // Reset back to a clean slate of zeros matching your neuron layout
        this.state := DenseVector.zeros[Double](w.cols)
    }

    override def predict(x: Batch, resetState: Boolean = true): Batch = 
        if resetState then
            this.state := breeze.linalg.DenseVector.zeros[Double](w.cols)
        
        forward(x)
    
    override def forward(x: Batch): Batch = 

        def run(x: DenseMatrix[Double]): DenseMatrix[Double] = 
            val nTimeSteps = x.rows
            val nNeurons = w.cols
            val output = DenseMatrix.zeros[Double](nTimeSteps, nNeurons)

            for t <- 0 until nTimeSteps do
                // Creating a view to a row of a Breeze matrix returns a type of: Transpose[DenseVector[Double]]. This represents a row vector (which still is a view to the underlying storage).
                // We transpose the view to the a column vector by using the t operator. Now all our shapes should match.
                val u_t = x(t, ::).t // Raw input vector at time t: Shape [Dim x 1]

                // Calculate the next state. The u_t is the output of the previous layer, be that an input layer another reservoir or 
                //a redout
                val nextState = (1.0 - leak) * state + leak * activation(u_t + (w * state))
                
                // Set the corresponding output pattern to be equal to the next state
                output(t, ::) := nextState.t

                // Update the state of the reservoir
                state := nextState

            // Return the output
            output

        x match 
            case Batch.Labeled(x, y) => 
                val yhat = run(x)
                Batch.Labeled(yhat, y)
            case Batch.Unlabeled(x) =>
                val yhat = run(x)  
                Batch.Unlabeled(yhat)

object Reservoir:
    
    def typed[Dim <: Int : ValueOf]: ZIO[Any, Nothing, Reservoir[Dim]] = 
        val dim = valueOf[Dim]
        val spectralInit = SpectralInitializer(dim)
        for {
            wRes <- spectralInit.generate()
        } yield Reservoir[Dim](wRes, Tanh(), spectralInit.leak)

    def typed[Dim <: Int : ValueOf](spectralRadius:Double, inputScale:Double, density:Double, leak:Double, nIter:Int=100):ZIO[Any, Nothing, Reservoir[Dim]] = 
        val dim = valueOf[Dim]
        val spectralInit = SpectralInitializer(dim, spectralRadius, inputScale, density, nIter, leak)
        for {
            wRes <- spectralInit.generate()
        } yield Reservoir[Dim](wRes, Tanh(), spectralInit.leak)

    

class Readout[In <: Int : ValueOf, Out <: Int : ValueOf](override val w:DenseMatrix[Double]) extends ParameterizedLayer[In, Out]:

    val in = valueOf[In]
    val out = valueOf[Out]

    require(
        w.rows == out && w.cols == in + 1, 
        s"Readout matrix w must have shape ($out, ${in + 1}) to match types [In=$in, Out=$out] with bias, but got (${w.rows}, ${w.cols})"
    )

    override def toString():String = s"[ Readout (${shape()})]"

    override def forward(x: Batch): Batch = 

        def run(z:DenseMatrix[Double]):DenseMatrix[Double] =
            val zWithBias = DenseMatrix.horzcat(DenseMatrix.ones[Double](z.rows, 1), z)
            zWithBias * w.t
        
        x match 
            case Batch.Labeled(x, y) => 
                val yhat = run(x)
                Batch.Labeled(yhat, y)
            case Batch.Unlabeled(x) =>
                val yhat = run(x)  
                Batch.Unlabeled(yhat)

    def update(w:DenseMatrix[Double]):Unit = this.w := w

object Readout:
    def typed[In <: Int, Out <: Int](using evIn: ValueOf[In], evOut: ValueOf[Out]) = 
        val in  = evIn.value
        val out = evOut.value
    
        // [Rows = out, Cols = in + 1]
        val initializer = RandomInitializer(out, in + 1) 
        for {
            weights <- initializer.generate()
        } yield Readout[In, Out](weights)

/**
 * Now that we have defined all the layers we need to be able to assemble these layers. For this purpose we will define
 * two traits: the SeriralFlow trait which allows us to assemble layers in series i.e. one after the other and in Parallel.
 * 
 * Both traits have two paramaters both being subtypes of Layer. The traits and the implementations are used exclusively for
 * enforcing constraints to how layers can be connected. 
 * 
 * The point is to treat the traits not as data structures but as mathematical proof. If the compiler can locate a valid instance
 * of SerialFlow in implicit scope then it proves that layers are composable.
 * 
*/
@implicitNotFound("Architecture Error: Cannot sequentially pipe ${From} into ${To}.")
sealed trait SerialFlow[From <: Layer[?, ?], To <: Layer[?,?]]

@implicitNotFound("Architecture Error: Cannot parallelize ${A} and ${B}.")
sealed trait ParallelFlow[A <: Layer[?,?], B <: Layer[?,?]]

/**
 * The best way to restrict how the layers should be connected is by providing implicit implementations of the flows. Each of these
 * either implements the serial flow or the parallel flow. They provide the proof, the evidence, required by the compiler to allow us to put stuff together. 
 * 
 * We are enforcing a Logical Topology of a Neural Network. We are teaching the Scala compiler the laws of Reservoir computing. 
 * 
*/
object SerialFlow:

    //Allow serial connection from readout to input
    given readoutToInput[In <: Int, Mid <: Int, Out <: Int]:SerialFlow[Readout[In, Mid], Input[Mid, Out]] with {}
    //Allow serial connection from input to reservoir. Input -> Reservoir
    given inputToReservoir[In <: Int, Out <: Int]: SerialFlow[Input[In, Out], Reservoir[Out]] with {}
    //Allow serial connection from reservoir to reservoir. Reservoir -> Reservoir. This will allow us to construct Deep ESN architectures. 
    given resToRes[In <: Int, Out <: Int]: SerialFlow[Reservoir[In], Reservoir[Out]] with {}
    //Allow serial connection from reservoir to readout. Reservoir -> Readout
    given resToReadout[Out <: Int, OutR <: Int]: SerialFlow[Reservoir[Out], Readout[Out, OutR]] with {}
    //Allow identity layer to any other layer
    given identityToLayer[In <: Int, Out <: Int]: SerialFlow[Identity[In], Layer[In, Out]] with {}   
    //Reservoir (Out=Mid) >>> Dropout (In=Mid, Out=Mid). Allow serial connection from reservoir to dropout.
    given resToDropout[Mid <: Int]: SerialFlow[Reservoir[Mid], Dropout[Mid]] with {}
    //Dropout (In=Mid, Out=Mid) >>> Readout (In=Mid, Out=Final). Allow serial connection from dropout to reservoir.
    given dropoutToReadout[Mid <: Int, Final <: Int]: SerialFlow[Dropout[Mid], Readout[Mid, Final]] with {}
    /*
     Inductive step for Deep Stacks. This allows us to push a new layer to a sequential model (provided that the last layer of the sequential model is compatible with the layer being pushed).
     To define the serial flow we use the Sequential data type that accepts five type parameters and serial flow evidence. 
     Lets break things down: we are telling the compiler that we want to connect a sequential layer to a new layer. Now we need to specify the 
     dimensionality of three layers, the first two that will be used to construct the first layer as being a sequential one and the third layer which 
     connects to the sequential. The using is an inductive step. If it is ommited then any serial connection would be possible.
     However, including it provides an inductive step that rigorously proves the tail of the sequence B is allowed to connect to C.
    */
    
    given seqToNext[In <: Int, Mid <: Int, Out <: Int, A <: Layer[In, Mid], B <: Layer[Mid, Out], C <: Layer[?, ?]](using SerialFlow[B, C]): SerialFlow[Sequential[In, Mid, Out, A, B], C] with {}

    // Skip-Connection Output Flowing into Readout
    given skipToReadout[In <: Int, Mid <: Int, OutB <: Int,InL <: Input[In, Mid], ResL <: Reservoir[Mid],ImpL <: Input[In, OutB]]: SerialFlow[Concat[In, Mid, OutB, Sequential[In, Mid, Mid, InL, ResL], ImpL], Readout[?, ?]] with {}
  
    
    // Allow serial connection between concatenation layer and any other layer
    given concatToNext[
        OutA <: Int, 
        OutB <: Int, 
        A <: Layer[?, OutA], 
        B <: Layer[?, OutB],
        OutNext <: Int,
        Next <: Layer[OutA TypeAdd OutB, OutNext]
    ]: SerialFlow[Concat[Int, OutA, OutB, A, B], Next] with {}

    // Allow any layer to feed into a Concat block, provided its output dimension 
    // exactly matches the input dimension the Concat block expects.
    given layerToConcat[
        In   <: Int,
        Mid  <: Int,
        OutA <: Int,
        OutB <: Int,
        Prev <: Layer[In, Mid],
        A    <: Layer[Mid, OutA],
        B    <: Layer[Mid, OutB]
    ]: SerialFlow[Prev, Concat[Mid, OutA, OutB, A, B]] with {}

    // Allow a Concat block to feed into any subsequent layer (like Readout),
    // provided the next layer's input exactly matches (OutA + OutB).
    given concToNext[
        In   <: Int,
        OutA <: Int,
        OutB <: Int,
        A    <: Layer[In, OutA],
        B    <: Layer[In, OutB],
        C    <: Layer[OutA TypeAdd OutB, ?] // The ultimate dimensional proof!
    ]: SerialFlow[Concat[In, OutA, OutB, A, B], C] with {}

object ParallelFlow:
    // Standard skip connection branching
    given resAndInput[
        In <: Int, Mid <: Int, OutB <: Int,
        InL <: Input[In, Mid], ResL <: Reservoir[Mid],
        ImpL <: Input[In, OutB]
    ]: ParallelFlow[Sequential[In, Mid, Mid, InL, ResL], Input[In, OutB]] with {}

    given parallelLayers[
        In <: Int, 
        OutA <: Int, 
        OutB <: Int, 
        A <: Layer[In, OutA], 
        B <: Layer[In, OutB]
    ]: ParallelFlow[A, B] with {}


    


/**
 * The Model trait has four type parameters. The first two represent the input and output dimensions of the resulting layer
 * and the subsequent two the two layers that will be combined to produce the compisite layer.
 * 
*/
sealed trait Model[In <: Int, Out <: Int, A <: Layer[?, ?], B <: Layer[?, ?]] extends Layer[In, Out]:
    val layerA:A
    val layerB:B

    // A seed for putting the model to work in generative mode. This currently work only for univariate time series, i.e. no exogenous 
    // variables.
    protected var finalTrainingInput: Option[DenseMatrix[Double]] = None

    //Silence the compiler; this is exhaustive 
    override def toString(): String = (this: @unchecked) match
        case Sequential(layerA, layerB) => 
            s"${layerA.toString} ----> ${layerB.toString}" 
        case Concat(layerA, layerB) => 
            s"""[ ${layerA.toString} + ${layerB.toString} ]"""
            
    override def isComposite: Boolean = true

    def setSeed(input: DenseMatrix[Double]): Unit = 
        this.finalTrainingInput = Some(input.copy)

    /**
     * The evaluate method expects a Labelled batch. It will create a batched dataset out of it where 
     * each batch will have a size equal to one. The model predicts a 
     * 
    */
    def evaluate(input: Batch): ZIO[Any, Nothing, Batch] = 
        input match
            case Batch.Labeled(x, y) => 
                ZIO.succeed {
                    val outBatch = this.predict(Batch.Unlabeled(x), resetState = true) 
                    val preds = outBatch.asInstanceOf[Batch.Unlabeled].x
                    // Return an Evaluation Batch: Labeled(True, Predicted)
                    Batch.Labeled(y, preds)
                }
            
            case _ => ZIO.succeed(input)
         
    def forward(x:Batch):Batch
    
    def getReservoirs: Seq[Reservoir[?]] = 
        @annotation.tailrec
        def go(acc: Seq[Reservoir[?]], remaining: Seq[Layer[?, ?]]): Seq[Reservoir[?]] = 
            //Base case, return the accumulated list of Reservoirs
            if remaining.isEmpty then acc
            //Decompose the list into the head and tail
            else
                val current = remaining.head
                val nextTail = remaining.tail
                
                //investigate the head
                current match
                    // It is a Reservoir. Add it to the accumulator and recursively call the function with the tail
                    case r: Reservoir[?] => 
                        go(r +: acc, nextTail)

                    // It is a sequential layer i.e. complex layer. Decompose, add the constituents to the tail and proceed with 
                    // a new recursive call
                    case Sequential(a, b) => 
                        go(acc, a +: b +: nextTail)

                    // Same case as before, composite layer but concat this time 
                    case Concat(a, b) => 
                        go(acc, a +: b +: nextTail)

                    // Add layer
                    //case Add(a, b) => 
                    //    go(acc, a +: b +: nextTail)

                    // The only remaining case if the inbvestiagted layer is not a reservoir or a composite layer is that it is another primitive 
                    // layer, in whcih case we ignore.
                    case _ => 
                        go(acc, nextTail)
                    
        go(Seq.empty, Seq(this))

    def getCascade:Seq[Layer[?,?]] = 

        @annotation.tailrec
        def go(stack:Seq[Layer[?,?]], model:Layer[?,?]):Seq[Layer[?,?]] = 
            model match
                //Why do we expect layerA to be composite? Becuase in the simplest possible scenario 
                //an ESN constists of input -> reservoir -> readout and the combination input -> reservoir make a composite layer
                case Sequential(layerA, layerB:Readout[?,?]) if layerA.isComposite =>
                    go(model +: stack, layerA)

                case Sequential(layerA, layerB) =>
                    go(stack, layerA) 

                //The base case should be a Sequential(Input, Reservoir) or Add(Sequential(Input, Reservoir), Input) if a residual layer is used 
                case Sequential(_:Input[?,?], _:Reservoir[?]) | Concat(Sequential(_:Input[?,?], _:Reservoir[?]), _:Input[?,?]) => stack

                //Invalid architecture return so stack does not blow 
                case _ => stack
            

        go(Seq.empty, this)

    
    def parameters():Seq[DenseMatrix[Double]] = 
        val submodels = getCascade
        submodels.map{ 
            case Sequential(layerA, layerB:Readout[?, ?]) => Some(layerB.w)
            case _ =>  None
        }.filter(_.isDefined).map(_.get) 

    //Generative mode: works only for univariate time series
    def forecast(steps: Int): Batch.Unlabeled =
        val seed = finalTrainingInput.getOrElse(
            throw new IllegalStateException("Cannot forecast: Model has not been trained yet, or no training seed was captured.")
        )

        val predictionMatrix = DenseMatrix.zeros[Double](steps, 1)
        var currentInput = seed.copy

        for t <- 0 until steps do
            // Warm start: only reset on the very first step of forecasting
            val shouldReset = (t == 0)
            
            val outputBatch = this.predict(Batch.Unlabeled(currentInput), resetState = shouldReset)
            
            outputBatch match
                case Batch.Unlabeled(yHat) =>
                    val nextVal = yHat(0, 0)
                    predictionMatrix(t, 0) = nextVal
                    currentInput(0, 0) = nextVal // Feedback loop
                case _ => 
                    throw new RuntimeException("Forecast engine encountered invalid structural topology.")

        Batch.Unlabeled(predictionMatrix)

object Model:

    import Utils.{denseMatrixDecoder, denseMatrixEncoder}

//Look at the beautiful part: A <: Layer[In, Mid], B <: Layer[Mid, Out]
//This will enforce that the output of layer A will be of the same size as the input to layer B.
case class Sequential[In <: Int, Mid <: Int, Out <: Int, A <: Layer[In, Mid], B <: Layer[Mid, Out]](layerA:A, layerB: B) extends Model[In, Out, A, B]:
    override def forward(x: Batch): Batch = 
        // Whether called by optimizer.fit() or model.predict(),
        // if a Labeled training batch passes through here, grab the final row!
        x match
            case Batch.Labeled(xMat, _) if xMat.rows > 0 =>
                val lastRow = xMat(xMat.rows - 1, ::).t.toDenseMatrix
                this.setSeed(lastRow)
            case _ => () 

        // execution cascade
        layerB.forward(layerA.forward(x))

    override def predict(x: Batch, resetState: Boolean = true): Batch = 
        layerB.predict(layerA.predict(x, resetState), resetState)
       

case class Concat[In <: Int, OutA <: Int, OutB <: Int, A <: Layer[?, OutA], B <: Layer[?, OutB]](layerA: A, layerB: B) extends Model[In, OutA TypeAdd OutB, A, B]:
        override def forward(batch: Batch): Batch = batch 
            match
                case Batch.Labeled(x, y) =>
                    val b1 = layerB.forward(Batch.Labeled(x, y))
                    val b2 = layerA.forward(Batch.Labeled(x, y))
                    (b1, b2) match
                        case (Batch.Labeled(z1, y), Batch.Labeled(z2, _)) => 
                            val yhat = DenseMatrix.horzcat(z1, z2)
                            Batch.Labeled(yhat, y)

                        //This should never happen
                        case _ => batch
                case Batch.Unlabeled(x) =>
                    val b1 = layerB.forward(Batch.Unlabeled(x))
                    val b2 = layerA.forward(Batch.Unlabeled(x))
                    (b1, b2) match
                        case (Batch.Unlabeled(z1), Batch.Unlabeled(z2)) =>
                                val yhat = DenseMatrix.horzcat(z1, z2)
                                Batch.Unlabeled(yhat)

                        case _ => batch


extension [In <: Int, Mid <: Int, A <: Layer[In, Mid]](a: A)
    // Enforces that the output dimension of A matches the input dimension of B
    def >>> [Out <: Int, B <: Layer[Mid, Out]](b: B)(using proof: SerialFlow[A, B]): Sequential[In, Mid, Out, A, B] = 
        Sequential(a, b)
    
    def <<< [Out <: Int, B <: Layer[Out, In]](b: B)(using proof: SerialFlow[B, A]): Sequential[Out, In, Mid, B, A] = 
        Sequential(b, a)

extension [In <: Int, OutA <: Int, A <: Layer[In, OutA]](a: A)
    // Enforces that parallel branches receive the same input type dimension
    def + [OutB <: Int, B <: Layer[In, OutB]](layerB: B)(using proof: ParallelFlow[A, B]): Concat[In, OutA, OutB, A, B] =
        Concat(a, layerB)

extension [In <: Int, OutA <: Int, OutB <: Int, A <: Layer[In, OutA], B <: Layer[In, OutB]](c: Concat[In, OutA, OutB, A, B])
    def >>> [OutNext <: Int, Next <: Layer[OutA TypeAdd OutB, OutNext]](next: Next)(using proof: SerialFlow[Concat[In, OutA, OutB, A, B], Next]): Sequential[In, OutA TypeAdd OutB, OutNext, Concat[In, OutA, OutB, A, B], Next] =
        Sequential(c, next)