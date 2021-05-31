package qa;

import java.util.function.BiFunction;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.FieldMatrix;
import org.apache.commons.math3.linear.FieldVector;

public class ComplexOde {
    private BiFunction<Double, FieldVector<Complex>, FieldVector<Complex>> diffeq;
    private double t = 0.0;
    private FieldVector<Complex> y;
    private double h = 0.001;

    public ComplexOde(BiFunction<Double, FieldVector<Complex>, FieldVector<Complex>> diffeq) {
        this.diffeq = diffeq;
    }

    public void setInitValue(double t0, FieldVector<Complex> y0) {
        this.t = t0;
        this.setInitValue(y0);
    }

    public void setInitValue(FieldVector<Complex> y0) {
        this.y = y0;
    }

    public void setDeltaValue(double h) {
        this.h = h;
    }

    public FieldVector<Complex> integrate() {
        FieldVector<Complex> nextY;

        // for (int i = 0; i < this.diffeq.length; i++) {
        // Complex k1 = this.diffeq[i].apply(Double.valueOf(this.t),
        // this.y).multiply(this.h);
        // Complex k2 = this.diffeq[i].apply(this.t + this.h * 0.5,
        // this.y.mapAdd(k1.multiply(0.5))).multiply(this.h);
        // Complex k3 = this.diffeq[i].apply(this.t + this.h * 0.5,
        // this.y.mapAdd(k2.multiply(0.5))).multiply(this.h);
        // Complex k4 = this.diffeq[i].apply(this.t + this.h,
        // this.y.mapAdd(k3)).multiply(this.h);
        // Complex kSum = k1.add(k2.multiply(2.0)).add(k3.multiply(2.0)).add(k4);

        // nextY.setEntry(i, y.getEntry(i).add(kSum.multiply(1.0 / 6.0)));
        // }

        FieldVector<Complex> k1 = this.diffeq.apply(this.t, this.y).mapMultiply(new Complex(this.h));
        FieldVector<Complex> k2 = this.diffeq.apply(this.t + this.h * 0.5, this.y.add(k1.mapMultiply(new Complex(0.5))))
                .mapMultiply(new Complex(this.h));
        FieldVector<Complex> k3 = this.diffeq.apply(this.t + this.h * 0.5, this.y.add(k2.mapMultiply(new Complex(0.5))))
                .mapMultiply(new Complex(this.h));
        FieldVector<Complex> k4 = this.diffeq.apply(this.t + this.h, this.y.add(k3)).mapMultiply(new Complex(this.h));
        FieldVector<Complex> kSum = k1.add(k2.mapMultiply(new Complex(2.0))).add(k3.mapMultiply(new Complex(2.0)))
                .add(k4).mapMultiply(new Complex(1.0 / 6.0));

        nextY = this.y.add(kSum);
        this.y = nextY;
        this.t += this.h;

        return nextY;
    }

    public FieldVector<Complex> integrateToTau(double Tau) {
        FieldVector<Complex> y = this.y;

        while (this.t <= Tau) {
            y = this.integrate();
        }

        return y.copy();
    }
}
