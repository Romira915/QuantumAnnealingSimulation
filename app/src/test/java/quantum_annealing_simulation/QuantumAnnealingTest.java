package quantum_annealing_simulation;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;

import static org.junit.Assert.*;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

public class QuantumAnnealingTest {
    @Test
    public void testAmp2prob() {
        INDArray array = Nd4j.create(new double[] { -2.4, 5.2, 0.0, -0.5, 0.2 });
        array = QuantumAnnealing.amp2prob(array);

        assertThat(array, is(Nd4j.create(new double[] { 2.4 * 2.4, 5.2 * 5.2, 0.0 * 0.0, 0.5 * 0.5, 0.2 * 0.2 })));
    }
}
