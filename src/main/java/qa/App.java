/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package qa;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.BiFunction;

import org.apache.commons.lang3.CharUtils;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayFieldVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.FieldVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.random.UncorrelatedRandomVectorGenerator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.Range;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.rng.NativeRandom;

import checkers.igj.quals.I;
import checkers.units.quals.A;
import checkers.units.quals.m;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.eigen.Eigen;

public class App {

    private static XYDataset createDataset(INDArray f) {
        final XYSeries data = new XYSeries("normal");
        for (int i = 0; i < f.length(); i++) {
            data.add(i, f.getDouble(i));
        }

        final XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(data);
        return dataset;
    }

    private static void timeIndependent(QuantumAnnealing quantumAnnealing, int N) {
        try {
            Files.createDirectories(Paths.get("./amp"));
        } catch (Exception e) {
            // TODO: handle exception
        }

        RealMatrix H = null;
        double step = 0.01;
        double time_steps[] = Nd4j.arange(0, quantumAnnealing.getTau() + step, step).toDoubleVector();
        // RealVector eigenValues = Nd4j.zeros((int) Math.pow(2, N), time_steps.length);
        RealMatrix eigenValues = new Array2DRowRealMatrix((int) Math.pow(2, N), time_steps.length);
        int eigenValuesIndex = 0;
        for (double t : time_steps) {
            H = quantumAnnealing.create_tfim(t, H);
            EigenDecomposition eigenDecomposition = new EigenDecomposition(H);
            RealMatrix eigenVectors = eigenDecomposition.getV(); // deepcopy
            eigenValues.setColumn(eigenValuesIndex, eigenDecomposition.getRealEigenvalues());

            eigenValuesIndex += 1;

            RealVector amp = QuantumAnnealing.amp2prob(eigenVectors.getColumnVector(0));
            PlotChart probabilityDensityChart = new PlotChart("probabilityDensity", "x", "y", PlotChart.createXYDataset(
                    QuantumAnnealing.iNDArrayToApacheVector(Nd4j.arange(0, amp.getDimension())), amp, "baseState"));
            probabilityDensityChart.setYRange(0, 1);
            probabilityDensityChart.saveChartAsJPEG("./amp/amp" + String.format("%.2f", t) + ".jpg", 600, 400);

            if (t == quantumAnnealing.getTau()) {
                XYDataset dataset = PlotChart.createXYDataset(
                        QuantumAnnealing.iNDArrayToApacheVector(Nd4j.arange(0, amp.getDimension())),
                        QuantumAnnealing.iNDArrayToApacheMatrix(
                                Nd4j.create(new double[][] { amp.toArray(), quantumAnnealing.getE().toArray() })),
                        new String[] { "probability", "E" }, true);
                PlotChart tauChart = new PlotChart("", "x", "y", dataset);
                tauChart.saveChartAsJPEG("tau_gaussian.jpg", 1000, 800);
            }
        }

        String eigenChartKeys[] = Arrays.stream(Nd4j.arange(0, eigenValues.getRowDimension()).toIntVector())
                .mapToObj(String::valueOf).toArray(String[]::new);

        System.out.println("time: " + time_steps.length + " rowsValues: " + eigenValues.getRowDimension()
                + " columnValues: " + eigenValues.getColumnDimension() + " Keys: " + eigenChartKeys.length);
        PlotChart eigenValueChart = new PlotChart("eigenValue", "t/τ", "E",
                PlotChart.createXYDataset(new ArrayRealVector(time_steps), eigenValues, eigenChartKeys, true));
        eigenValueChart.saveChartAsJPEG("eigenValue.jpg", 800, 600);
        // eigenValueChart.showChart();
    }

    public static void execQA() {
        final int N = 5;

        Random rand = Nd4j.getRandom();
        rand.setSeed(1042);
        double rndVec[] = Nd4j.randn(0.0, N / 2.0, new long[] { (long) Math.pow(2, N) }, rand).toDoubleVector();
        RealVector E = new ArrayRealVector(rndVec);

        PlotChart gaussianChart = new PlotChart("Gaussian", "x", "y", PlotChart
                .createXYDataset(new ArrayRealVector(Nd4j.arange(0, (int) Math.pow(2, N)).toDoubleVector()), E, "E"));
        gaussianChart.saveChartAsJPEG("gaussian.jpg", 800, 600);
        // gaussianChart.showChart();

        QuantumAnnealing quantumAnnealing = new QuantumAnnealing(N, E);

        FieldVector<Complex> y0 = new ArrayFieldVector<Complex>((int) Math.pow(2, quantumAnnealing.getN()),
                new Complex(Math.pow(2, -quantumAnnealing.getN() / 2)));
        quantumAnnealing.generateDiffeqArray();
        ComplexOde complexOde = new ComplexOde(quantumAnnealing.simdiffeq_rhs);
        complexOde.setInitValue(y0);

        FieldVector<Complex> result = complexOde.integrateToTau(quantumAnnealing.getTau());
        RealVector psi = QuantumAnnealing.amp2prob(result);
        XYDataset resultDataset = PlotChart.createXYDataset(
                QuantumAnnealing.iNDArrayToApacheVector(Nd4j.arange((int) Math.pow(2, N))), psi, "result");
        PlotChart resultPlotChart = new PlotChart("result", "|ψ|^2", "p", resultDataset);
        resultPlotChart.showChart();
    }

    public static void main(String[] args) {
        int N = 10;
        int seed = (int) System.currentTimeMillis();
        Random random = Nd4j.getRandom();
        random.setSeed(seed);

        INDArray numbers = Nd4j.rand(0, 1, random, N);
        double sum = numbers.sumNumber().doubleValue();
        double m = sum * 0.5;
        INDArray qubo = Nd4j.zeros(N, N);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double value = 4 * numbers.getDouble(i) * numbers.getDouble(j);
                qubo.putScalar(i, j, value);
            }
        }

        for (int i = 0; i < N; i++) {
            double value = qubo.getDouble(i) - 4 * sum * numbers.getDouble(i);

            qubo.putScalar(i, i, value);
        }

        QuantumAnnealing quantumAnnealing = new QuantumAnnealing(QuantumAnnealing.iNDArrayToApacheMatrix(qubo), true,
                16, 1, 1000000, 1000, 1000, seed, true);

        quantumAnnealing.execQMC();

        RealVector resultState = quantumAnnealing.getMinEnergyState();
        RealVector apacheNumbers = QuantumAnnealing.iNDArrayToApacheVector(numbers);
        // resultState = resultState.map((v) -> (v + 1) / 2);
        System.out.println(numbers);
        System.out.println("sum:" + sum);
        System.out.println("m:" + m);
        System.out.println("state" + resultState);
        double a = resultState.dotProduct(apacheNumbers);
        System.out.println("a: " + a);
        System.out.println("b: " + (sum - a));
    }
}