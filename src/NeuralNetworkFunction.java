import java.util.function.UnaryOperator;

public class NeuralNetworkFunction {


    public static UnaryOperator<Double> getFunction(NeuralNetworkFunctionName functionName) {

        UnaryOperator<Double> result;

        if (functionName == NeuralNetworkFunctionName.SIGMOID) {
            result = x -> 1 / (1 + Math.exp(-x));
        } else if (functionName == NeuralNetworkFunctionName.D_SIGMOID) {
            result = y -> y * (1 - y);
        } else {
            result = x -> 1 / (1 + Math.exp(-x));
        }
        return result;
    }
}
