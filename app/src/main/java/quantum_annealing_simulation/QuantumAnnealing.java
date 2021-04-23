package quantum_annealing_simulation;

import org.la4j.Matrix;

public class QuantumAnnealing {
    final int Tau = 1;

    public double scheduleE(double time) {
        return time / this.Tau;
    }

    public double scheduleG(double time) {
        return (Tau - time) / this.Tau;
    }

    public double create_tfim(double time, Matrix hamiltonian) {
        double v = scheduleE(time);

        if (hamiltonian == null) {
        }

        return 0;
    }
}
