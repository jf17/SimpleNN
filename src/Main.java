import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.function.UnaryOperator;

public class Main {
    public static final double LEARNING_RATE = 0.001;
    public static final NeuralNetworkFunctionName ACTIVATION = NeuralNetworkFunctionName.SIGMOID;
    public static final NeuralNetworkFunctionName DERIVATIVE = NeuralNetworkFunctionName.D_SIGMOID;
    public static final String TRAIN_PATH = "./dataset/digits/train";
    public static final String NEURAL_NETWORK_PATH = "./NeuralNetwork";
    public static final int EPOCHS = 1000;
    public static final int BATCH_SIZE = 100;


    public static void main(String[] args) throws IOException, ClassNotFoundException {
        dots();
//        digitsTrainNewNeuralNetwork();
//        digitsLoadFromFileNeuralNetwork();
    }

    private static void dots() {
        FormDots f = new FormDots();
        new Thread(f).start();
    }

    private static void digitsTrainNewNeuralNetwork() throws IOException {
        NeuralNetwork nn = new NeuralNetwork(LEARNING_RATE, ACTIVATION, DERIVATIVE, 784, 512, 128, 32, 10);

        int samples = 60000;
        BufferedImage[] images = new BufferedImage[samples];
        int[] digits = new int[samples];
        File[] imagesFiles = new File(TRAIN_PATH).listFiles();
        for (int i = 0; i < samples; i++) {
            images[i] = ImageIO.read(imagesFiles[i]);
            digits[i] = Integer.parseInt(imagesFiles[i].getName().charAt(10) + "");
        }

        double[][] inputs = new double[samples][784];
        for (int i = 0; i < samples; i++) {
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    inputs[i][x + y * 28] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                }
            }
        }

        int epochs = EPOCHS;
        for (int i = 1; i < epochs; i++) {
            int right = 0;
            double errorSum = 0;
            int batchSize = BATCH_SIZE;
            for (int j = 0; j < batchSize; j++) {
                int imgIndex = (int)(Math.random() * samples);
                double[] targets = new double[10];
                int digit = digits[imgIndex];
                targets[digit] = 1;

                double[] outputs = nn.feedForward(inputs[imgIndex]);
                int maxDigit = 0;
                double maxDigitWeight = -1;
                for (int k = 0; k < 10; k++) {
                    if(outputs[k] > maxDigitWeight) {
                        maxDigitWeight = outputs[k];
                        maxDigit = k;
                    }
                }
                if (digit == maxDigit) right++;
                for (int k = 0; k < 10; k++) {
                    errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                }
                nn.backpropagation(targets);
            }
            System.out.println("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
        }

        FileOutputStream outputStream = new FileOutputStream(NEURAL_NETWORK_PATH + "/digit-epochs-" + EPOCHS + ".dat");
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);

        objectOutputStream.writeObject(nn);
        objectOutputStream.close();

        FormDigits f = new FormDigits(nn);
        new Thread(f).start();
    }

    private static void digitsLoadFromFileNeuralNetwork() throws IOException, ClassNotFoundException {

        FileInputStream fileInputStream = new FileInputStream(NEURAL_NETWORK_PATH + "/digit-epochs-" + EPOCHS + ".dat");
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);

        NeuralNetwork neuralNetwork = (NeuralNetwork) objectInputStream.readObject();

        FormDigits f = new FormDigits(neuralNetwork);
        new Thread(f).start();
    }

}