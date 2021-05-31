package qa;

import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.io.File;
import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartMouseEvent;
import org.jfree.chart.ChartMouseListener;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.panel.CrosshairOverlay;
import org.jfree.chart.plot.Crosshair;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.netbeans.api.annotations.common.NullAllowed;

import checkers.units.quals.min;

public class PlotChart {
    private JFreeChart chart;
    private String title = "";

    public PlotChart(@NullAllowed String title, @NullAllowed String xLabel, @NullAllowed String yLabel,
            XYDataset xyDataset) {

        this.chart = ChartFactory.createXYLineChart(title, xLabel, yLabel, xyDataset);
        this.title = title;
    }

    public void saveChartAsJPEG(String file, int width, int height) {
        try {
            ChartUtils.saveChartAsJPEG(new File(file), 0.85f, this.chart, width, height);
        } catch (Exception e) {
            // TODO: handle exception
        }
    }

    public void showChart() {
        ChartFrame chartFrame = new ChartFrame(this.title, this.chart);
        chartFrame.setSize(800, 600);
        chartFrame.setVisible(true);
    }

    public void setXRange(double min, double max) {
        this.chart.getXYPlot().getDomainAxis().setRange(min, max);
    }

    public void setYRange(double min, double max) {
        this.chart.getXYPlot().getRangeAxis().setRange(min, max);
    }

    /**
     * Data Shape in the case of isRowVectors is true, xData.length ==
     * yData.columnLength (if isRowVectors is false, yData.rowLength) keys.length ==
     * yData.rowLength (if isRowVectors is false, yData.columnLength)
     */
    public static XYDataset createXYDataset(RealVector xData, RealMatrix yData, String[] keys, boolean isRowVectors) {
        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();

        int dataNum = 0;
        if (isRowVectors) {
            dataNum = yData.getRowDimension();
        } else {
            dataNum = yData.getColumnDimension();
        }

        // Data shape check
        // if (num != keys.length) {
        // System.err.println("Not match yDataSize:" + num + " and keysSize:" +
        // keys.length);
        // return null;
        // }

        for (int i = 0; i < dataNum; i++) {
            XYSeries xySeries = new XYSeries(keys[i]);
            double yValues[] = isRowVectors ? yData.getRow(i) : yData.getColumn(i);
            for (int j = 0; j < yValues.length; j++) {
                xySeries.add(xData.getEntry(j), yValues[j]);
            }
            xySeriesCollection.addSeries(xySeries);
        }

        return xySeriesCollection;
    }

    public static XYDataset createXYDataset(RealVector xData, RealVector yData, String key) {
        return PlotChart.createXYDataset(xData, MatrixUtils.createRowRealMatrix(yData.toArray()), new String[] { key },
                true);
    }
}
