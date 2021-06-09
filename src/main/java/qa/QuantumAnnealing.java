package qa;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.complex.ComplexField;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayFieldVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.FieldVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.Nd4jCpu.static_bidirectional_rnn;

import checkers.units.quals.g;

public class QuantumAnnealing {
    private final int Tau = (int) Math.pow(2, 5);
    private final int N;
    private final double hBar = 1;
    private int trotterN;
    private RealVector E;
    private RealMatrix isingModel;
    private RealVector state[];
    private BiFunction<Double, FieldVector<Complex>, Complex>[] diffeqArray;
    private boolean isQUBO;

    public BiFunction<Double, FieldVector<Complex>, FieldVector<Complex>> simdiffeq_rhs = (t, vec) -> {
        FieldVector<Complex> dydt = new ArrayFieldVector<>(ComplexField.getInstance(), vec.getDimension());

        for (int i = 0; i < this.diffeqArray.length; i++) {
            dydt.setEntry(i, QuantumAnnealing.j.multiply(this.diffeqArray[i].apply(t, vec)).multiply(-1 / hBar));
        }

        return dydt;
    };

    private static NDMath ndMath = new NDMath();
    public final static Complex j = new Complex(0, 1);

    public int getTau() {
        return this.Tau;
    }

    public int getN() {
        return this.N;
    }

    public QuantumAnnealing(int N, RealVector E) {
        this.N = N;
        this.E = E;
    }

    public QuantumAnnealing(RealMatrix ising, boolean isQUBO) {
        this.N = ising.getRowDimension();
        this.isingModel = ising;
        this.isQUBO = isQUBO;

    }

    private double scheduleE(double time) {
        return time / this.Tau;
    }

    private double scheduleG(double time) {
        return (Tau - time) / this.Tau;
    }

    private int spin(double value) {
        int v = (int) (value);

        return this.spin(v);
    }

    private int spin(int value) {
        if (this.isQUBO) {
            return 2 * value - 1;
        } else {
            return value;
        }
    }

    private double classicalTrotterEnergy(int trotter) {
        double E = 0;

        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
                E += -this.isingModel.getEntry(i, j) * this.spin(this.state[trotter].getEntry(i))
                        * this.spin(this.state[trotter].getEntry(j));
            }
        }

        return E;
    }

    private double classicalEnergy() {
        double E = 0;

        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
                for (int k = 0; k < this.trotterN; k++) {
                    E += -(this.isingModel.getEntry(i, j) / this.trotterN) * this.spin(this.state[k].getEntry(i))
                            * this.spin(this.state[k].getEntry(j));
                }
            }
        }

        return E;
    }

    private double quantumEnergy(double T, double gamma) {
        double E = 0;
        double beta = 1 / T;

        for (int i = 0; i < this.N; i++) {
            for (int k = 0; k < this.trotterN; k++) {
                E += this.spin(this.state[k].getEntry(i)) * this.spin(this.state[(k + 1) % this.trotterN].getEntry(i));
            }
        }

        double bias = (beta * gamma) / this.trotterN;
        E *= -1 / (2 * beta) * Math.log(Math.cosh(bias) / Math.sinh(bias));

        return E;
    }

    private double energy(int t) {
        double E = 0;

        E = this.scheduleE(t) * this.classicalEnergy() + this.quantumEnergy(t, this.scheduleG(t));

        return E;
    }

    public RealMatrix create_tfim(double time, RealMatrix hamiltonian) {
        // Set diagonal
        double v = this.scheduleE(time);

        if (hamiltonian == null) {
            hamiltonian = MatrixUtils.createRealDiagonalMatrix(this.E.mapMultiply(v).toArray());
        } else {
            for (int i = 0; i < this.E.getDimension(); i++) {
                hamiltonian.setEntry(i, i, v * this.E.getEntry(i));
            }
        }
        // End

        // Set off-diagonal
        double g = -1 * this.scheduleG(time);
        for (int i = 0; i < this.E.getDimension(); i++) {
            for (int n = 0; n < this.N; n++) {
                int j = i ^ (1 << n);
                hamiltonian.setEntry(i, j, g);
            }
        }
        // End

        return hamiltonian;
    }

    private Complex genrate_diffeq(int index, int offdiag_indices[], double time, FieldVector<Complex> vec) {
        double v = this.E.getEntry(index) * this.scheduleE(time);
        double g = -1 * this.scheduleG(time);

        return vec.getEntry(index).multiply(v).add(
                QuantumAnnealing.sumComplex(QuantumAnnealing.getSubVector(vec, offdiag_indices).toArray()).multiply(g));
    }

    public BiFunction<Double, FieldVector<Complex>, Complex>[] generateDiffeqArray() {
        BiFunction<Double, FieldVector<Complex>, Complex> diffeqArray[] = new BiFunction[(int) Math.pow(2, this.N)];

        for (int i = 0; i < diffeqArray.length; i++) {
            final int finalI = i;
            diffeqArray[i] = (t, vec) -> {
                int indices[] = new int[this.N];
                for (int j = 0; j < indices.length; j++) {
                    indices[j] = finalI ^ (1 << j);
                }

                return this.genrate_diffeq(finalI, indices, t, vec);
            };
        }
        this.diffeqArray = diffeqArray;
        return diffeqArray;
    }

    public static FieldVector<Complex> getSubVector(FieldVector<Complex> vector, int indices[]) {
        FieldVector<Complex> subVector = new ArrayFieldVector<>(ComplexField.getInstance(), indices.length);

        for (int i = 0; i < indices.length; i++) {
            subVector.setEntry(i, vector.getEntry(indices[i]));
        }

        return subVector;
    }

    public static Complex sumComplex(Complex values[]) {
        Complex sumValue = Complex.ZERO;
        for (Complex complex : values) {
            sumValue = sumValue.add(complex);
        }

        return sumValue;
    }

    public static RealVector amp2prob(RealVector vec) {
        INDArray array = QuantumAnnealing.ndMath.abs(QuantumAnnealing.apacheVectorToINDArray(vec));
        array = QuantumAnnealing.ndMath.pow(array, 2);

        return QuantumAnnealing.iNDArrayToApacheVector(array);
    }

    public static RealVector amp2prob(FieldVector<Complex> vec) {
        RealVector result = new ArrayRealVector(vec.getDimension(), 0);
        for (int i = 0; i < result.getDimension(); i++) {
            result.setEntry(i, Math.pow(vec.getEntry(i).abs(), 2));
        }

        return result;
    }

    public static RealMatrix iNDArrayToApacheMatrix(INDArray array) {
        return new Array2DRowRealMatrix(array.toDoubleMatrix());
    }

    public static RealVector iNDArrayToApacheVector(INDArray array) {
        return new ArrayRealVector(array.toDoubleVector());
    }

    public static INDArray apacheMatrixToINDArray(RealMatrix matrix) {
        return Nd4j.create(matrix.getData());
    }

    public static INDArray apacheVectorToINDArray(RealVector matrix) {
        return Nd4j.create(matrix.toArray());
    }

    public static RealVector[] initIsingState(int size, int trotterN) {
        return null;
    }
}
