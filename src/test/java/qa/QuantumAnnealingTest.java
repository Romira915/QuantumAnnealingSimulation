package qa;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.PointValuePair;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

public class QuantumAnnealingTest {
    @Test
    public void testAmp2prob() {
        RealVector array = MatrixUtils.createRealVector(new double[] { -2.4, 5.2, 0.0, -0.5, 0.2 });
        array = QuantumAnnealing.amp2prob(array);

        assertThat(array, is(
                MatrixUtils.createRealVector(new double[] { 2.4 * 2.4, 5.2 * 5.2, 0.0 * 0.0, 0.5 * 0.5, 0.2 * 0.2 })));
    }

    @Test
    public void testINDArraySum() {
        INDArray array = Nd4j.create(new double[] { -2.0, 4, 8 });

        assertThat(array.sumNumber().doubleValue(), is(10.0));
    }

    @Test
    public void testReverseSpin() {
        RealMatrix qubo = new Array2DRowRealMatrix(2, 2);
        QuantumAnnealing quantumAnnealingQUBO = new QuantumAnnealing(qubo, true, 2, 0, 0, 0, 0, 0, false);
        QuantumAnnealing quantumAnnealingIsing = new QuantumAnnealing(qubo, false, 2, 0, 0, 0, 0, 0, false);

        assertThat(quantumAnnealingQUBO.reverseSpin(0), is(1));
        assertThat(quantumAnnealingQUBO.reverseSpin(1), is(0));
        assertThat(quantumAnnealingIsing.reverseSpin(-1), is(1));
        assertThat(quantumAnnealingIsing.reverseSpin(1), is(-1));
    }
}
