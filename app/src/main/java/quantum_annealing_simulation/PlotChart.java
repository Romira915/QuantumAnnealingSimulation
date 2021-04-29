package quantum_annealing_simulation;

import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.io.File;
import java.util.Arrays;

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
            ChartUtils.saveChartAsJPEG(new File(file), 85, this.chart, width, height);
        } catch (Exception e) {
            // TODO: handle exception
        }
    }

    public void showChart() {
        ChartFrame chartFrame = new ChartFrame(this.title, this.chart);
        chartFrame.setSize(800, 600);
        chartFrame.setVisible(true);
    }

    public static XYDataset createXYDataset(INDArray xData, INDArray yData, String[] keys, boolean isRowVectors) {
        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();

        if (yData.isVector()) {
            yData = yData.reshape(1, yData.length());
            isRowVectors = true;
        }

        int num = 0;
        if (isRowVectors) {
            num = yData.rows();
        } else {
            num = yData.columns();
        }

        for (int i = 0; i < num; i++) {
            XYSeries xySeries = new XYSeries(keys[i]);
            INDArray rowVec = isRowVectors ? yData.getRow(i) : yData.getColumn(i);
            for (int j = 0; j < rowVec.length(); j++) {
                xySeries.add(xData.getNumber(j), rowVec.getNumber(j));
            }
            xySeriesCollection.addSeries(xySeries);
        }

        return xySeriesCollection;
    }
}
