package qa;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.apache.commons.math3.linear.RealVector;

public class SchedulerQA {
    private QuantumAnnealing[] quantumAnnealings;
    private ExecQA execQA[];

    public static class HyperParameter {
        public int trotterN;
        public double initBeta;
        public int initGamma;
        public int monteCarloStep;
        public int annealingStep;

        public HyperParameter(int trotterN, double initBeta, int initGamma, int mStep, int aStep) {
            this.trotterN = trotterN;
            this.initBeta = initBeta;
            this.initGamma = initGamma;
            this.monteCarloStep = mStep;
            this.annealingStep = aStep;
        }

        @Override
        public String toString() {
            return "trotterN: " + this.trotterN + " initBeta: " + this.initBeta + " initGamma: " + this.initGamma
                    + " mStep: " + this.monteCarloStep + "astep: " + this.annealingStep;
        }
    }

    private class ExecQA extends Thread {
        private QuantumAnnealing threadQA;

        public ExecQA(QuantumAnnealing quantumAnnealing) {
            this.threadQA = quantumAnnealing;
        }

        @Override
        public void run() {
            this.threadQA.execQMC();
        }
    }

    public SchedulerQA(RealMatrix ising, boolean isQUBO, HyperParameter[] hyperParameters, int seed,
            boolean needDebugLog) {
        this.quantumAnnealings = new QuantumAnnealing[hyperParameters.length];
        this.execQA = new ExecQA[hyperParameters.length];

        for (int i = 0; i < hyperParameters.length; i++) {
            this.quantumAnnealings[i] = new QuantumAnnealing(ising, isQUBO, hyperParameters[i], seed, needDebugLog);
            this.execQA[i] = new ExecQA(this.quantumAnnealings[i]);
        }
    }

    public void run() {
        for (int i = 0; i < execQA.length; i++) {
            this.execQA[i].start();
        }

        for (int i = 0; i < execQA.length; i++) {
            try {
                this.execQA[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (Exception e) {
                // TODO: handle exception
            }
        }
    }

    public Pair<RealVector, Double>[] getResult(boolean getQUBO) {
        Pair<RealVector, Double>[] results = new Pair[this.quantumAnnealings.length];

        for (int i = 0; i < results.length; i++) {
            results[i] = this.quantumAnnealings[i].getMinEnergyState(getQUBO);
        }

        return results;
    }
}
