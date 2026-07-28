package rezervoir

import zio.*
import zio.stream.*

case class OptimalConfig(
    spectralRadius:Double, 
    inputScale:Double, 
    density:Double, 
    leak:Double,
    alpha:Double
)

//TODO
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