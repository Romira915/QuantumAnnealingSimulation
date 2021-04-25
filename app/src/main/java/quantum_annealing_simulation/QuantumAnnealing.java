package quantum_annealing_simulation;

import org.la4j.Matrix;
import org.la4j.Vector;

public class QuantumAnnealing {
    final int Tau = 1;
    final int N;
    Vector E;

    public int getTau() {
        return this.Tau;
    }

    public int getN() {
        return this.N;
    }

    public QuantumAnnealing(int N, Vector E) {
        this.N = N;
        this.E = E;
    }

    private double scheduleE(double time) {
        return time / this.Tau;
    }

    private double scheduleG(double time) {
        return (Tau - time) / this.Tau;
    }

    public Matrix create_tfim(double time, Matrix hamiltonian) {
        // Set diagonal
        double v = this.scheduleE(time);

        if (hamiltonian == null) {
            hamiltonian = E.multiply(v).toDiagonalMatrix();
        } else {
            for (int i = 0; i < E.length(); i++) {
                hamiltonian.set(i, i, v * this.E.get(i));
            }
        }
        // End

        // Set off-diagonal
        double g = -1 * this.scheduleG(time);
        for (int i = 0; i < this.E.length(); i++) {
            for (int n = 0; n < this.N; n++) {
                int j = i ^ (1 << n);
                hamiltonian.set(i, j, g);
            }
        }
        // End

        return hamiltonian;
    }
}
