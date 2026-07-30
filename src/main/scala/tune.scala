package rezervoir

import zio.*
import zio.stream.*


/**
 * 
 * 
*/
enum Config:
    case ReservoirConfig(spectralRadius:Double, density:Double, leak:Double, scale:Double)



case class OptimalConfig(
    spectralRadius:Double, 
    inputScale:Double, 
    density:Double, 
    leak:Double,
    alpha:Double
)

/**
 * We will implement an automatic tuner based on the Metropolis-Hastings algorithm. The idea is the following:
 * The hyperaparameter values are sampled. Once we have obtained a full sample we train the model and evaluate it using some 
 * loss function. If we trace down the values of the hyperperameters over time we can imagine as taking a tour around 
 * the hyperparameter space, we are exploring it. The question is the following: how de we explore the space? Common sense
 * says that we should focus around areas of the space where the loss values are low. 
 *  
 * 
*/
trait Tuner:
    def tune(leakStep:Double=0.1, densityStep:Double=0.05, scaleStep:Double=0.05):ZIO[Model[?, ?, ?, ?] & Dataset, Nothing, Seq[OptimalConfig]] = 

        val leakCandidates: Seq[Double] = Seq.unfold(0.1) { leak =>
            if leak <= 0.9 then Some((leak, leak + leakStep))
            else None
        }

        val densityCandidates: Seq[Double] = Seq.unfold(0.05) { density =>
            if density <= 0.3 then Some((density, density + densityStep))
            else None
        }

        val scaleCandidates:Seq[Double] = Seq.unfold(0.1) {scale =>
            if scale <= 0.7 then Some((scale, scale + scaleStep))
            else None
            }

        
        ???