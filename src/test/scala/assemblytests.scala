package tests

import zio.*
import rezervoir.*
import rezervoir.Utils.*
import rezervoir.Layer.reservoirLayer

class ModelAssemblyTests extends munit.FunSuite:
    test("Vanilla ESN construction yields a composite model") {
        val program = for {
            input <- Input.typed[1, 1000]
            reservoir <- Reservoir.typed[1000]
            readout <- Readout.typed[1000, 1]
        } yield input >>> reservoir >>> readout
        
        val model = program.unsafeRun()
        assert(model.isComposite)
    }

    test("Dynamic input layer construction") {
        val in = 10
        val out = 1000
        val program = Input.typed[in.type, out.type]
        val input = program.unsafeRun()
        assert(input.shape() == (10, 1000))
    }

   
    test("Deep ESN with multiple reservoir layers") {
        for {
            input <- Input.typed[1, 1000]
            res1 <- Reservoir.typed[1000]
            res2 <- Reservoir.typed[1000]
            readout <- Readout.typed[1000, 1]
            model = input >>> res1 >>> res2 >>> readout
        } yield model
    }