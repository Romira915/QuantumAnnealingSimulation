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
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.Nd4jCpu.static_bidirectional_rnn;
import org.nd4j.shade.protobuf.Extension;

import checkers.units.quals.g;

public class QuantumAnnealing {
    private final int N;
    private final double hBar = 1;
    private final int reverseSpinN = 1;
    private int reverseTrotterN;
    private int Tau = (int) Math.pow(2, 5);
    private int trotterN;
    private RealVector E;
    private RealMatrix isingModel;
    private RealMatrix state;
    private BiFunction<Double, FieldVector<Complex>, Complex>[] diffeqArray;
    private boolean isQUBO;
    private int monteCarloStep;
    private int annealingStep;
    private java.util.Random jRandom;
    private org.nd4j.linalg.api.rng.Random nRandom;

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

    public RealVector getE() {
        return this.E;
    }

    public QuantumAnnealing(int N, RealVector E) {
        this.N = N;
        this.E = E;
    }

    public QuantumAnnealing(RealMatrix ising, boolean isQUBO, int trotterN, int Tau, int mStep, int aStep, int seed) {
        this.N = ising.getRowDimension();
        this.isingModel = ising;
        this.isQUBO = isQUBO;
        this.trotterN = trotterN;
        this.jRandom = new java.util.Random(seed);
        this.nRandom = Nd4j.getRandom();
        this.nRandom.setSeed(seed);
        this.state = isQUBO ? QuantumAnnealing.initQUBOState(this.N, trotterN, this.nRandom)
                : QuantumAnnealing.initIsingState(this.N, trotterN, this.nRandom);
        this.Tau = Tau;
        this.monteCarloStep = mStep;
        this.annealingStep = aStep;
        this.reverseTrotterN = trotterN / 2;
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
                E += -this.isingModel.getEntry(i, j) * this.spin(this.state.getEntry(trotter, i))
                        * this.spin(this.state.getEntry(trotter, j));
            }
        }

        return E;
    }

    private double classicalEnergy(RealMatrix state) {
        double E = 0;

        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
                for (int k = 0; k < this.trotterN; k++) {
                    E += -(this.isingModel.getEntry(i, j) / this.trotterN) * this.spin(state.getEntry(k, i))
                            * this.spin(state.getEntry(k, j));
                }
            }
        }

        return E;
    }

    private double quantumEnergy(RealMatrix state, double T, double gamma) {
        double E = 0;
        double beta = 1 / T;

        for (int i = 0; i < this.N; i++) {
            for (int k = 0; k < this.trotterN; k++) {
                E += this.spin(state.getEntry(k, i)) * this.spin(state.getEntry((k + 1) % this.trotterN, i));
            }
        }

        double bias = (beta * gamma) / this.trotterN;
        E *= -1 / (2 * beta) * Math.log(Math.cosh(bias) / Math.sinh(bias));

        return E;
    }

    private double energy(RealMatrix state, int t) {
        double E = 0;

        E = this.scheduleE(t) * this.classicalEnergy(state) + this.quantumEnergy(state, t, this.scheduleG(t));

        return E;
    }

    private double diffEnergy(RealMatrix before, RealMatrix after, double t) {
        double deltaE = 0;

        deltaE = this.scheduleE(t) * this.classicalEnergy(after) + this.quantumEnergy(after, t, this.scheduleG(t));
        deltaE -= this.scheduleE(t) * this.classicalEnergy(before) + this.quantumEnergy(before, t, this.scheduleG(t));

        return deltaE;
    }

    private static RealMatrix initState(int size, int trotterN, INDArray source,
            org.nd4j.linalg.api.rng.Random random) {
        RealMatrix state;
        INDArray probs = Nd4j.valueArrayOf(source.length(), 1.0 / source.length());

        INDArray initArray = Nd4j.create(source.dataType(), trotterN, size);
        initArray = Nd4j.choice(source, probs, initArray, random);
        state = QuantumAnnealing.iNDArrayToApacheMatrix(initArray);

        return state;
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

    public RealMatrix randomSpinReverse() {
        if (this.trotterN < this.reverseTrotterN || this.N < this.reverseSpinN) {
            throw new ArithmeticException();
        }
        RealMatrix nextState = this.state.copy();

        INDArray rndTrotterIndex = Nd4j.linspace(0, this.trotterN, this.trotterN, DataType.INT16);
        Nd4j.shuffle(rndTrotterIndex, this.jRandom, 0);
        INDArray rndSpinIndex = Nd4j.linspace(0, this.N, this.N, DataType.INT16);
        Nd4j.shuffle(rndSpinIndex, this.jRandom, 0);

        for (int i = 0; i < this.reverseTrotterN; i++) {
            for (int j = 0; j < this.reverseSpinN; j++) {
                int spin = (int) this.state.getEntry(i, j);
                nextState.setEntry(i, j, this.reverseSpin(spin));
            }
        }

        return nextState;
    }

    public int reverseSpin(int spin) {
        if (spin == 1) {
            spin = this.isQUBO ? 0 : -1;
        } else {
            spin = 1;
        }

        return spin;
    }

    public void execQMC() {
        for (double t = 0; t <= this.Tau; t += (double) this.Tau / this.annealingStep) {
            for (int j = 0; j < this.monteCarloStep; j++) {
                RealMatrix nextState = this.randomSpinReverse();

                double deltaE = this.diffEnergy(this.state, nextState, t);
                double p = Math.min(1, Math.exp(-deltaE / t));
                if (QuantumAnnealing.randomBoolean(p, this.jRandom)) {
                    this.state = nextState;
                }
            }
        }
    }

    public RealVector getMinEnergyState() {
        RealVector stateEnergy = new ArrayRealVector(this.state.getRowDimension());
        for (int i = 0; i < this.state.getRowDimension(); i++) {
            stateEnergy.setEntry(i, this.classicalTrotterEnergy(i));
        }

        return this.state.getRowVector(stateEnergy.getMinIndex());
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

    public static INDArray apacheVectorToINDArray(RealVector vector) {
        return Nd4j.create(vector.toArray());
    }

    public static RealMatrix initIsingState(int size, int trotterN, org.nd4j.linalg.api.rng.Random random) {
        INDArray source = Nd4j.createFromArray(new float[] { 1, -1 });
        return QuantumAnnealing.initState(size, trotterN, source, random);
    }

    public static RealMatrix initQUBOState(int size, int trotterN, org.nd4j.linalg.api.rng.Random random) {
        INDArray source = Nd4j.createFromArray(new float[] { 1, 0 });
        return QuantumAnnealing.initState(size, trotterN, source, random);
    }

    public static boolean randomBoolean(double p, java.util.Random random) {
        return p > random.nextDouble();
    }
}
