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
import org.apache.commons.math3.util.Pair;
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
import qa.SchedulerQA.HyperParameter;

public class QuantumAnnealing {
    private final int N;
    private final double hBar = 1;
    private final int reverseSpinN = 1;
    private int reverseTrotterN;
    private int Tau = (int) Math.pow(2, 5);
    private int time = 0;
    private double initBeta;
    private int initGamma = 1;
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
    private boolean needDebugLog;

    public BiFunction<Double, FieldVector<Complex>, FieldVector<Complex>> simdiffeq_rhs = (t, vec) -> {
        FieldVector<Complex> dydt = new ArrayFieldVector<>(ComplexField.getInstance(), vec.getDimension());

        for (int i = 0; i < this.diffeqArray.length; i++) {
            dydt.setEntry(i, QuantumAnnealing.j.multiply(this.diffeqArray[i].apply(t, vec)).multiply(-1 / hBar));
        }

        return dydt;
    };

    private static NDMath ndMath = Nd4j.math();
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

    public QuantumAnnealing(RealMatrix ising, boolean isQUBO, int trotterN, double initBeta, int initGamma, int mStep,
            int aStep, int seed, boolean needDebugLog) {
        this.jRandom = new java.util.Random(seed);
        this.nRandom = Nd4j.getRandom();
        this.nRandom.setSeed(seed);

        this.N = ising.getRowDimension();
        this.isingModel = ising.copy();
        this.isQUBO = isQUBO;
        this.trotterN = trotterN;
        this.state = isQUBO ? QuantumAnnealing.initQUBOState(this.N, trotterN, this.nRandom)
                : QuantumAnnealing.initIsingState(this.N, trotterN, this.nRandom);
        this.initBeta = initBeta;
        this.initGamma = initGamma;
        this.monteCarloStep = mStep;
        this.annealingStep = aStep;
        this.reverseTrotterN = 1;
        this.needDebugLog = needDebugLog;
    }

    public QuantumAnnealing(RealMatrix ising, boolean isQUBO, HyperParameter hyperParameter, int seed,
            boolean needDebugLog) {
        this(ising, isQUBO, hyperParameter.trotterN, hyperParameter.initBeta, hyperParameter.initGamma,
                hyperParameter.monteCarloStep, hyperParameter.annealingStep, seed, needDebugLog);
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
                if (i <= j) {
                    E += -this.isingModel.getEntry(i, j) * this.spin(this.state.getEntry(trotter, i))
                            * this.spin(this.state.getEntry(trotter, j));
                }
            }
        }

        return E;
    }

    private double classicalEnergy(RealMatrix state) {
        double E = 0;

        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
                for (int k = 0; k < this.trotterN; k++) {
                    if (i <= j) {
                        E += -this.isingModel.getEntry(i, j) * this.spin(state.getEntry(k, i))
                                * this.spin(state.getEntry(k, j));
                    }
                }
            }
        }

        E /= this.trotterN;

        return E;
    }

    private double quantumEnergy(RealMatrix state, double beta, double gamma) {
        double E = 0;

        for (int i = 0; i < this.N; i++) {
            for (int k = 0; k < this.trotterN; k++) {
                E += this.spin(state.getEntry(k, i)) * this.spin(state.getEntry((k + 1) % this.trotterN, i));
            }
        }

        double bias = (beta * gamma) / this.trotterN;
        bias = -1 / (2 * beta) * Math.log(1f / Math.tanh(bias));
        E *= Double.isNaN(bias) ? 0 : bias;

        return E;
    }

    private double energy(RealMatrix state, double beta, double gamma) {
        double E = 0;

        // E = this.scheduleE(t) * this.classicalEnergy(state) +
        // this.quantumEnergy(state, t, this.scheduleG(t));
        double c = this.classicalEnergy(state);
        double q = this.quantumEnergy(state, beta, gamma);
        E = c + q;

        return E;
    }

    private double diffEnergy(RealMatrix before, RealMatrix after, double beta, double gamma) {
        double deltaE = 0;

        double a = this.energy(after, beta, gamma);
        double b = this.energy(before, beta, gamma);
        deltaE = a - b;

        return deltaE;
    }

    private double diffEnergy(int trotter, int spinIndex) {
        double deltaE = 0;

        for (int i = 0; i < this.N; i++) {
            deltaE += this.isingModel.getEntry(spinIndex, i) * this.state.getEntry(trotter, i);
        }
        deltaE *= 2 * this.state.getEntry(trotter, spinIndex);

        return deltaE;
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

        if (this.reverseTrotterN == 1 && this.reverseSpinN == 1) {
            int trotterIndex = this.jRandom.nextInt(this.trotterN);
            int spinIndex = this.jRandom.nextInt(this.N);
            nextState.setEntry(trotterIndex, spinIndex,
                    this.reverseSpin((int) nextState.getEntry(trotterIndex, spinIndex)));
            return nextState;
        }

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

    public Pair<Integer, Integer> randomSpinReversePair() {
        int trotterIndex = this.jRandom.nextInt(this.trotterN);
        int spinIndex = this.jRandom.nextInt(this.N);

        return new Pair<Integer, Integer>(trotterIndex, spinIndex);
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
        double gamma = this.initGamma;
        double deltaGamma = (double) this.initGamma / (this.annealingStep);
        this.Tau = this.annealingStep;
        double decrease = 0;
        double increase = 0;

        // for (double beta = this.initBeta; beta > 0; beta -= (double) this.initBeta
        // / this.annealingStep, this.time += 1) {
        // for (int g = 0; g < this.monteCarloStep; g++) {
        // RealMatrix nextState = this.randomSpinReverse();

        // double deltaE = this.diffEnergy(this.state, nextState, beta, gamma);
        // double p = Math.min(1, Math.exp(-deltaE / beta));
        // if (QuantumAnnealing.randomBoolean(p, this.jRandom)) {
        // this.state = nextState;
        // if (this.needDebugLog) {
        // System.out.println("deltaE: " + deltaE + " p: " + p);
        // }
        // if (deltaE > 0) {
        // increase += deltaE;
        // } else {
        // decrease += deltaE;
        // }
        // }

        // gamma -= deltaGamma;
        // }
        // }

        double beta = this.initBeta;
        for (int i = 0; i < this.annealingStep; i++) {
            for (int g = 0; g < this.monteCarloStep; g++) {
                RealMatrix nextState = this.randomSpinReverse();

                double deltaE = this.diffEnergy(this.state, nextState, beta, gamma);
                double p = Math.min(1, Math.exp(-deltaE * beta));
                if (QuantumAnnealing.randomBoolean(p, this.jRandom)) {
                    this.state = nextState;
                    if (this.needDebugLog) {
                        System.out.println("deltaE: " + deltaE + " p: " + p);
                    }
                    if (deltaE > 0) {
                        increase += deltaE;
                    } else {
                        decrease += deltaE;
                    }
                }

            }
            gamma -= deltaGamma;
        }

        if (this.needDebugLog) {
            System.out.println("increase:" + increase);
            System.out.println("decrease:" + decrease);
        }
    }

    public RealMatrix getState() {
        return this.state.copy();
    }

    public Pair<RealVector, Double> getMinEnergyState(boolean getQUBO) {
        RealVector stateEnergy = new ArrayRealVector(this.state.getRowDimension());
        for (int i = 0; i < this.state.getRowDimension(); i++) {
            stateEnergy.setEntry(i, this.classicalTrotterEnergy(i));
            if (this.needDebugLog) {
                System.out.println("Energy[" + i + "]: " + stateEnergy.getEntry(i));
            }
        }

        RealVector minState = this.state.getRowVector(stateEnergy.getMinIndex());
        if (getQUBO && !this.isQUBO) {
            minState = minState.map((v) -> (v + 1) / 2);
        } else if (!getQUBO && this.isQUBO) {
            minState = minState.map((v) -> 2 * v - 1);
        }

        return new Pair<RealVector, Double>(minState, stateEnergy.getMinValue());
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

    private static RealMatrix initState(int size, int trotterN, INDArray source,
            org.nd4j.linalg.api.rng.Random random) {
        RealMatrix state;
        INDArray probs = Nd4j.valueArrayOf(source.length(), 1.0 / source.length());

        INDArray initArray = Nd4j.create(source.dataType(), trotterN, size);
        initArray = Nd4j.choice(source, probs, initArray, random);
        state = QuantumAnnealing.iNDArrayToApacheMatrix(initArray);

        return state;
    }

    public static boolean randomBoolean(double p, java.util.Random random) {
        return p > random.nextDouble();
    }
}
