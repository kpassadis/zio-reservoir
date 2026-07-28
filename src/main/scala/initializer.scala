package rezervoir

import zio.*
import breeze.linalg.{DenseVector, DenseMatrix, norm}

/**
 * The objective of the initializer data type is to provide the means to intialize a weight matrix 
 * for a layer.
 * There are three general categories of layers: an input layer, a reservoir layer and 
 * a readout layer.
 * Concrete implementations are provided for each type of layer. The constructor of each layer
 * is used to provide some essential information required for the layer construction (for instance 
 * the dimensionality of the matrix).
 * 
*/
trait Initializer:
    def generate():ZIO[Any, Nothing, DenseMatrix[Double]]

//A reservoir initializer requires additional information. The information carried by the initializer
//will also be used by the Reservoir layer.
trait ReservoirInitializer extends Initializer:
    val nRes:Int
    val spectralRadius:Double
    val leak:Double

class RandomInitializer(rows:Int, cols:Int, inputScale:Double=1.0, seed:Long=42) extends Initializer:

    private val rng = new scala.util.Random(seed.toLong)
    override def generate(): ZIO[Any, Nothing, DenseMatrix[Double]] = ZIO.succeed {
        val data = Array.fill(cols * rows) {
            (rng.nextDouble() * 2.0 - 1.0) * inputScale
        }
        new DenseMatrix(rows, cols, data)
    }

object RandomInitializer:

    def live(rows:Int, cols:Int, inputScale:Double=1.0, seed:Long=42):ZLayer[Any, Nothing, RandomInitializer] = 
        ZLayer.fromZIO(ZIO.succeed(RandomInitializer(rows, cols, inputScale, seed)))

/**
 * The initialization method of the reservoir as described in the paper.
 * 
*/
class SpectralInitializer(override val nRes:Int, override val spectralRadius:Double = 0.9, inputScale:Double=0.1, density:Double=0.1, nIter:Int=100, override val leak:Double=0.2, seed:Long = 42L) extends ReservoirInitializer:

     private val rng = new scala.util.Random(seed.toLong)

     private def powerIter(A: DenseMatrix[Double], nIter: Int = 50): Double =
        val v = DenseVector.ones[Double](A.cols)
        val t = Seq.unfold((v / norm(v), 0.0, 0)) { case (v, lam, i) =>
        val Av = A * v
        val nrm = norm(Av)
        if nrm == 0.0 || i == nIter then None
        else
            val lamNew = v.t * Av
            Some(((Av / nrm, lamNew, i + 1), (Av / nrm, lamNew, i + 1)))
        }
        math.abs(t.last._2)

     override def generate(): ZIO[Any, Nothing, DenseMatrix[Double]] = ZIO.succeed{

         // raw weights in [-1, 1]
        val rawData = Array.fill(nRes * nRes) {
            rng.nextDouble() * 2.0 - 1.0
        }
        val mat = new DenseMatrix(nRes, nRes, rawData)

        // mask with given density
        val maskData = Array.fill(nRes * nRes) {
            val p = rng.nextDouble()
            if (p < density) 1.0 else 0.0
        }
        val mask = new DenseMatrix(nRes, nRes, maskData)

        val masked = mask *:* mat
        val rhoEst = math.max(powerIter(masked), 1e-12)
        if (rhoEst > 0) masked * (spectralRadius / rhoEst)
        else masked
     }

object SpectralInitializer:

    def live(nRes:Int, spectralRadius:Double, inputScale:Double, density:Double, nIter:Int, leak:Double, seed:Long = 42):ZLayer[Any, Nothing, SpectralInitializer] = 
        ZLayer.fromZIO(ZIO.succeed(SpectralInitializer(nRes, spectralRadius, inputScale, density, nIter, leak, seed)))

class SimpleCycleInitializer(override val nRes: Int, override val spectralRadius: Double, override val leak:Double) extends ReservoirInitializer:
  override def generate(): ZIO[Any, Nothing, DenseMatrix[Double]] = ZIO.succeed {
    val mat = DenseMatrix.zeros[Double](nRes, nRes)
    
    // Connect each node to the next in a ring. Set the weight to be equal to the spectral radius.
    for (i <- 0 until nRes - 1) do
      mat(i + 1, i) = spectralRadius
      
    // Close the cycle
    mat(0, nRes - 1) = spectralRadius
    
    mat
  }

object SimpleCycleInitializer:
    def live(nRes:Int, spectralRadius:Double, leak:Double): ZIO[Any, Nothing, SimpleCycleInitializer] = ZIO.succeed{
        SimpleCycleInitializer(nRes, spectralRadius, leak)
    }