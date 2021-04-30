/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package quantum_annealing_simulation;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.lang3.CharUtils;
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
import org.nd4j.rng.NativeRandom;

import checkers.igj.quals.I;

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

    public static void main(String[] args) {
        final int N = 5;

        try {
            Files.createDirectories(Paths.get("./amp"));
        } catch (Exception e) {
            // TODO: handle exception
        }

        Random rand = Nd4j.getRandom();
        // rand.setSeed(915);
        INDArray E = Nd4j.randn(0.0, N / 2.0, new long[] { (long) Math.pow(2, N) }, rand);

        PlotChart gaussianChart = new PlotChart("Gaussian", "x", "y",
                PlotChart.createXYDataset(Nd4j.arange(0, (int) Math.pow(2, N)), E, new String[] { "E" }, true));
        // gaussianChart.showChart();

        // ChartFrame chartFrame = new ChartFrame("ガウス分布", jFreeChart);
        // chartFrame.setSize(1200, 1200);
        // chartFrame.setVisible(true);
        // try {
        // ChartUtils.saveChartAsJPEG(new File("gaussian.jpg"), jFreeChart, 600, 400);
        // } catch (Exception e) {
        // // TODO: handle exception
        // }

        QuantumAnnealing quantumAnnealing = new QuantumAnnealing(N, E);

        INDArray H = null;
        double step = 0.01;
        INDArray time_steps = Nd4j.arange(0, 1 + step, step);
        INDArray eigenValues = Nd4j.zeros((int) Math.pow(2, N), time_steps.length());
        int eigenValuesIndex = 0;
        for (double t : time_steps.toDoubleVector()) {
            H = quantumAnnealing.create_tfim(t, H);
            INDArray eigenVectors = H.dup(); // deepcopy
            eigenValues.putColumn(eigenValuesIndex, Eigen.symmetricGeneralizedEigenvalues(eigenVectors));
            eigenValuesIndex += 1;
            // INDArray eigenValues = Eigen.symmetricGeneralizedEigenvalues(eigenVectors);

            // XYSeriesCollection eigenVaCollection = new XYSeriesCollection();
            // for (int i = 0; i < eigenValues.length(); i++) {
            // if (eigenSeries[i] == null) {
            // eigenSeries[i] = new XYSeries(i);
            // }
            // eigenSeries[i].add(t, eigenValues.getDouble(i));
            // }
            // for (XYSeries xySeries : eigenSeries) {
            // eigenVaCollection.addSeries(xySeries);
            // }
            // JFreeChart eigenValChart = ChartFactory.createXYLineChart("eigenValue.jpg",
            // "t/τ", "E", eigenVaCollection);
            // try {
            // ChartUtils.saveChartAsJPEG(new File("./eigenValue.jpg"), eigenValChart, 800,
            // 600);
            // } catch (Exception e) {
            // // TODO: handle exception
            // }

            INDArray amp = QuantumAnnealing.amp2prob(eigenVectors.getColumn(0, false));
            PlotChart probabilityDensityChart = new PlotChart("probabilityDensity", "x", "y",
                    PlotChart.createXYDataset(Nd4j.arange(0, amp.length()), amp, new String[] { "baseState" }, false));
            probabilityDensityChart.saveChartAsJPEG("./amp/amp" + (int) (t * 100) + ".jpg", 600, 400);

            // JFreeChart chart = ChartFactory.createXYLineChart("amp", "x", "y",
            // App.createDataset(amp));
            // try {
            // ChartUtils.saveChartAsJPEG(new File("./amp/amp" + (int) (t * 100) + ".jpg"),
            // chart, 600, 400);
            // } catch (Exception e) {
            // // TODO: handle exception
            // }
        }

        String eigenChartKeys[] = Arrays.asList(Nd4j.arange(0, eigenValues.columns()).toIntVector()).stream().map(i -> {
            System.out.println(i);
            return String.valueOf(i);
        }).toArray(String[]::new);
        PlotChart eigenValueChart = new PlotChart("eigenValue", "t/τ", "E",
                PlotChart.createXYDataset(time_steps, eigenValues, eigenChartKeys, false));
        eigenValueChart.saveChartAsJPEG("eigenValue.png", 800, 600);
    }
}