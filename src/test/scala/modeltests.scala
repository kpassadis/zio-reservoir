package tests

import zio.*
import rezervoir.* 
import Utils.*

class ModelTests extends munit.FunSuite:

    test("ForestESN on passengers") {
        val program = for {
            dataset <- FullDataset.fromBatch(Dataset.airpassengers().toBatch(1))
            tuple <- dataset.split(0.9)
            (train, test) = tuple
            model <- ForestESN.build[1, 200, 1](200, 20, 42L)
            optimizer = Ridge(1.0, washout = 20)
            _ <- model.fit(optimizer, train, 1)
            preds <- model.predict(test.dataset)
        } yield preds

        val batch = program.unsafeRun()
        println(batch)
    }

    test("SkipESN on air passengers") {
        val program = for {
            dataset <- BatchedDataset.fromBatch(10, Dataset.airpassengers().toBatch(1))
            tuple <- dataset.split(0.9)
            (train, test) = tuple
            model <- SkipESN.build[1, 200, 1]
            descent = GradientDescent(0.001, 0.1, 0.9, 20)
            _ <- model.fit(descent, train, 200)
            preds <- model.predict(test.dataset)
        } yield preds

        val batch = program.unsafeRun()
        println(batch)
    }

    test("Customly created Wide ESN on air passengers") {
        val program = for {
            // Map the 1D signal into a 50D space
            input <- Input.typed[1, 50]

            // Both reservoirs expect 50D in, and output 50D
            res1 <- Reservoir.typed[50]
            res2 <- Reservoir.typed[50]

            // The Concat block expects 50D in, and outputs 100D (50 + 50)
            concat = res1 + res2

            // The Readout MUST catch the 100D state and map it back to 1D
            readout <- Readout.typed[100, 1]

            // The Graph is now perfectly mathematically proven
            // (1->50) >>> (50->100) >>> (100->1)   
            model = input >>> concat >>> readout

        } yield ()
    }

    test("Custom model for air passengers") {

        val program = for {
            input     <- Input.typed[1, 100]
            reservoir <- Reservoir.typed[100] 
            readout   <- Readout.typed[100, 1]
            input2 <- Input.typed[1, 10]
            reservoir2 <- Reservoir.typed[10] 
            readout2 <- Readout.typed[10,1]
            model = input >>> reservoir >>> readout >>> input2 >>> reservoir2 >>> readout2
            dataset <-  FullDataset.fromBatch(Dataset.airpassengers().toBatch(1))
            tuple <- dataset.split(0.9)
            (train, test) = tuple
            optimizer = Ridge(1.0, washout = 40)
            trainedModel <- optimizer.fit(model, 1, train)
            preds = trainedModel.predict(test.dataset)
        } yield preds

        println(program.unsafeRun())
    }

    test("Vanilla model on air passengers") {
        
        val program = for {
            dataset <-  FullDataset.fromBatch(Dataset.airpassengers().toBatch(1))
            tuple <- dataset.split(0.9)
            (train, test) = tuple
            net <- VanillaESN.build[1, 100, 1]
            optimizer = Ridge(1.0, washout = 40)
            _ <- net.fit(optimizer, train, 1)
            preds <- net.predict(test.dataset)
        } yield preds

        println(program.unsafeRun())
    }