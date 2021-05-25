package qa;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;

public class QuantumAnnealing {
    private final int Tau = 1;
    private final int N;
    private INDArray E;

    private static NDMath ndMath = new NDMath();

    public int getTau() {
        return this.Tau;
    }

    public int getN() {
        return this.N;
    }

    public QuantumAnnealing(int N, INDArray E) {
        this.N = N;
        this.E = E;
    }

    private double scheduleE(double time) {
        return time / this.Tau;
    }

    private double scheduleG(double time) {
        return (Tau - time) / this.Tau;
    }

    public INDArray create_tfim(double time, INDArray hamiltonian) {
        // Set diagonal
        double v = this.scheduleE(time);

        if (hamiltonian == null) {
            hamiltonian = Nd4j.zeros(DataType.DOUBLE, new long[] { E.length(), E.length() });
            new Diag(this.E.mul(v), hamiltonian);
        } else {
            for (int i = 0; i < E.length(); i++) {
                hamiltonian.put(i, i, v * this.E.getDouble(i));
            }
        }
        // End

        // Set off-diagonal
        double g = -1 * this.scheduleG(time);
        for (int i = 0; i < this.E.length(); i++) {
            for (int n = 0; n < this.N; n++) {
                int j = i ^ (1 << n);
                hamiltonian.put(i, j, g);
            }
        }
        // End

        return hamiltonian;
    }

    public double genrate_diffeq(int index, INDArray offdiag_indices, int time, INDArray vec) {
        double v = this.E.getDouble(index) * this.scheduleE(time);
        double g = -1 * this.scheduleG(time);

        return v * vec.getDouble(index) + g;
    }

    public void simdiffeq_rhs(int t, INDArray vec) {

    }

    public static INDArray amp2prob(INDArray vec) {
        INDArray array = QuantumAnnealing.ndMath.abs(vec.dup());
        array = QuantumAnnealing.ndMath.pow(array, 2);

        return array;
    }
}
