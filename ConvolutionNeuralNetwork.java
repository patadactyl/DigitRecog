/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package digitrecog;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import sun.rmi.server.Activation;

/**
 *
 * @author patrickcharlton
 */
public class ConvolutionNeuralNetwork {
    
    private static final String OUT_DIR = "resources/cnnCurrentTrainingModels";
    private static final String TRAINED_MODEL_FILE = "resources/cnnTrainedModels/bestModel.bin";

    private static final Logger LOG = LoggerFactory.getLogger(ConvolutionalNeuralNetwork.class);
    private MultiLayerNetwork preTrainedModel;

    public void init() throws IOException {
        preTrainedModel = ModelSerializer.restoreMultiLayerNetwork(new File(TRAINED_MODEL_FILE));
    }

    public int predict(LabeledImage labeledImage) {
        double[] pixels = labeledImage.getPixels();
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = pixels[i] / 255d;
        }
        int[] predict = preTrainedModel.predict(Nd4j.create(pixels));

        return predict[0];
    }

    public void train(Integer trainDataSize, Integer testDataSize) throws IOException {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 20; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, trainDataSize, false, true, true, 12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(800)
                        .nOut(128).build())
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(128)
                        .nOut(64).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true).pretrain(false).build();

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(75, TimeUnit.MINUTES))
                .scoreCalculator(new AccuracyCalculator(
                        new MnistDataSetIterator(testDataSize, testDataSize, false, false, true, 12345)))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(OUT_DIR))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, mnistTrain);

        EarlyStoppingResult result = trainer.fit();

        LOG.info("Termination reason: " + result.getTerminationReason());
        LOG.info("Termination details: " + result.getTerminationDetails());
        LOG.info("Total epochs: " + result.getTotalEpochs());
        LOG.info("Best epoch number: " + result.getBestModelEpoch());
        LOG.info("Score at best epoch: " + result.getBestModelScore());
    }

    public static void main(String[] args) throws Exception {
        new ConvolutionalNeuralNetwork().train(60000, 1000);
    }
}
