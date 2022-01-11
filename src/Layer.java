import java.io.Serializable;

public class Layer implements Serializable {

    private static final long serialVersionUID = 9216696740948170978L;

    public int size;
    public double[] neurons;
    public double[] biases;
    public double[][] weights;

    public Layer(int size, int nextSize) {
        this.size = size;
        neurons = new double[size];
        biases = new double[size];
        weights = new double[size][nextSize];
    }

}