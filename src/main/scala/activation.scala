package rezervoir

import breeze.linalg.*
import breeze.numerics.* 
import zio.* 

sealed trait Activation:

    def apply(x:DenseMatrix[Double]):DenseMatrix[Double] 
    
    def apply(x:DenseVector[Double]):DenseVector[Double]
    

case class Tanh() extends Activation:
    override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = tanh(x)

    override def apply(x: DenseVector[Double]): DenseVector[Double] = tanh(x)

case class Sigmoid() extends Activation:

    override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = sigmoid(x)

    override def apply(x: DenseVector[Double]): DenseVector[Double] = sigmoid(x)

object Activation:
    
    given tanhActivation:ZIO[Any, Nothing, Tanh] = ZIO.succeed(Tanh())

    def live[L <: Activation : Tag](using activation:ZIO[Any, Nothing, L]):ZLayer[Any, Nothing, L] = ZLayer.fromZIO(activation)