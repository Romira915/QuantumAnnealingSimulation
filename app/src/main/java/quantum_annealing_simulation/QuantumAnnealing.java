package quantum_annealing_simulation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.factory.Nd4j;

public class QuantumAnnealing {
    final int Tau = 1;
    final int N;
    INDArray E;

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
}
