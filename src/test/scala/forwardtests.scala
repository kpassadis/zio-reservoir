package tests

import zio.*
import rezervoir.*
import rezervoir.Utils.*
import rezervoir.Layer.reservoirLayer

class ForwardPassTests extends munit.FunSuite:

    val x = Dataset.mackeyGlass(1000)
    val y = x.shift(1)
    val dataLayer = FullDataset.live(x, Some(y))

    println(s"${x.rows}, ${x.cols}, ${y.rows}, ${y.cols}")

    test("Manual Vanilla ESN forward pass") {

        val program = (for {
            input <- Input.typed[1, 1000]
            reservoir <- Reservoir.typed[1000]
            readout <- Readout.typed[1000, 1]
            model = input >>> reservoir >>> readout
            dataset <- ZIO.service[FullDataset]
            batch <- dataset.next().map(_.get)
            output = model.forward(batch)
        } yield output).provide(dataLayer)

        val outputBatch = program.unsafeRun()

        assertEquals(outputBatch.rows(), 1000)
    
    }


    test("Predefined architectures forward pass") {
        val program = (for {
            dataset <- ZIO.service[FullDataset]
            batch <- dataset.next().map(_.get)
            input <- Input.typed[1, 1000]
            vanilla <- VanillaESN.build[1, 1000, 1]
            deep <- DeepESN.build[1, 1000, 1](3)
            skip <- SkipESN.build[1, 1000, 1]
        } yield (vanilla.predict(batch), deep.predict(batch), skip.predict(batch))).provide(dataLayer)

        val tuple = program.unsafeRun()

    }