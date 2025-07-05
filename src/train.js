import tf from "@tensorflow/tfjs-node";
import fs from "node:fs";
import path from "node:path";
import {getFlattenedInputShape, getInputShape, loadProcessedData,} from "./preprocess.js";
import {createModel} from "./createModel.js";
import {createTripletLoss} from "./createTripletLoss.js";
import {generateTriplets} from "./generateTriplets.js";

const config = {
    modelDir: "./models/icon-similarity",
    epochs: 100,
    batchSize: 24,
    learningRate: 0.001, // Increased learning rate
    validationSplit: 0.2,
    embeddingDim: 128,
    patience: 15, // Increased patience
    minDelta: 0.001,
    margin: 0.5, // Increased margin
};

/**
 * Custom callback to monitor training progress
 */
class TripletLossCallback extends tf.Callback {
    constructor() {
        super();
        this.losses = [];
        this.valLosses = [];
    }

    setCurrentEpoch(epoch) {
        this.currentEpoch = epoch;
    }

    async onEpochEnd(epoch, logs) {
        const loss =
            typeof logs.loss === "number" ? logs.loss : logs.loss.dataSync()[0];
        const valLoss =
            typeof logs.val_loss === "number"
                ? logs.val_loss
                : logs.val_loss.dataSync()[0];

        this.losses.push(loss);
        this.valLosses.push(valLoss);

        // Check for NaN
        if (isNaN(loss) || isNaN(valLoss)) {
            console.error("NaN loss detected! Training may be unstable.");
        }

        // Log meaningful metrics
        console.log(
            `Epoch ${this.currentEpoch + 1}: loss=${loss.toFixed(6)}, val_loss=${valLoss.toFixed(6)}`,
        );
    }
}

/**
 * Train the model
 */
export async function trainModel() {
    console.log("Loading processed data...");
    const {metadata, features} = loadProcessedData();

    console.log(`Training on ${metadata.totalIcons} icons`);
    console.log(`Feature dimensions: ${metadata.featureLength}`);

    const inputShape = getInputShape();
    console.log(`Input shape: ${inputShape}`);

    // Create model
    console.log("Creating model...");
    const model = createModel(inputShape, config.embeddingDim);

    // Use a learning rate scheduler
    let currentLearningRate = config.learningRate;

    // Compile model
    model.compile({
        optimizer: tf.train.adam(currentLearningRate),
        loss: createTripletLoss(config.margin),
    });

    // Print model summary
    model.summary();

    // Calculate number of triplets per epoch
    const tripletsPerEpoch = Math.min(features.length, 2000);
    const validationTriplets = Math.floor(tripletsPerEpoch * 0.2);
    const trainingTriplets = tripletsPerEpoch - validationTriplets;

    // Generate validation data once
    console.log("Generating validation data...");
    const {x: xVal, y: yVal} = generateTriplets(
        features,
        validationTriplets,
        config.embeddingDim
    );

    // Ensure model directory exists
    const modelDir = path.dirname(config.modelDir);
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, {recursive: true});
    }

    const tripletCallback = new TripletLossCallback();

    console.log("Starting training...");
    let bestValLoss = Infinity;
    let bestWeights = null;
    let epochsSinceImprovement = 0;
    const history = {loss: [], val_loss: []};

    for (let epoch = 0; epoch < config.epochs; epoch++) {
        tripletCallback.setCurrentEpoch(epoch);

        // Learning rate decay
        if (epoch > 0 && epoch % 20 === 0) {
            currentLearningRate *= 0.8;
            console.log(`Reducing learning rate to ${currentLearningRate}`);
            model.compile({
                optimizer: tf.train.adam(currentLearningRate),
                loss: createTripletLoss(config.margin),
            });
        }

        // Generate new training data each epoch
        const {x: xTrain, y: yTrain} = generateTriplets(
            features,
            trainingTriplets,
            config.embeddingDim
        );

        // Train for one epoch
        const epochHistory = await model.fit(xTrain, yTrain, {
            epochs: 1,
            batchSize: config.batchSize,
            validationData: [xVal, yVal],
            callbacks: [tripletCallback],
            verbose: 0,
        });

        // Record history
        const trainLoss = epochHistory.history.loss[0];
        const valLoss = epochHistory.history.val_loss[0];
        history.loss.push(trainLoss);
        history.val_loss.push(valLoss);

        // Check for improvement
        if (valLoss < bestValLoss - config.minDelta) {
            bestValLoss = valLoss;
            epochsSinceImprovement = 0;

            // Save best weights
            if (bestWeights) {
                bestWeights.forEach((w) => w.dispose());
            }
            bestWeights = model.getWeights().map((w) => w.clone());

            // Save best model
            await model.save(`file://${config.modelDir}-best`);
            console.log(
                `  -> New best model saved (val_loss: ${valLoss.toFixed(6)})`,
            );
        } else {
            epochsSinceImprovement++;
        }

        // Clean up training data
        xTrain.dispose();
        yTrain.dispose();

        // Check if training should stop
        if (isNaN(valLoss) || isNaN(trainLoss)) {
            console.error("NaN detected, stopping training.");
            break;
        }

        // Manual early stopping
        if (epochsSinceImprovement >= config.patience) {
            console.log(`Early stopping triggered at epoch ${epoch + 1}`);
            if (bestWeights) {
                model.setWeights(bestWeights);
            }
            break;
        }
    }

    // Clean up
    if (bestWeights) {
        bestWeights.forEach((w) => w.dispose());
    }
    xVal.dispose();
    yVal.dispose();

    // Save final model
    console.log("Saving final model...");
    await model.save(`file://${config.modelDir}`);

    // Save training metrics
    const metrics = {
        loss: history.loss,
        valLoss: history.val_loss,
        epochs: history.loss.length,
        bestEpoch: history.val_loss.indexOf(Math.min(...history.val_loss)),
        trainedAt: new Date().toISOString(),
        config: config,
    };

    const modelMetadata = {
        inputShape,
        flattenedInputShape: getFlattenedInputShape(),
        embeddingDim: config.embeddingDim,
        totalIcons: metadata.totalIcons,
        version: metadata.version,
        trainedAt: new Date().toISOString(),
    };

    fs.writeFileSync(
        path.join(modelDir, "training-metrics.json"),
        JSON.stringify(metrics, null, 2),
    );

    fs.writeFileSync(
        path.join(modelDir, "model-metadata.json"),
        JSON.stringify(modelMetadata, null, 2),
    );

    console.log("Training completed!");
    console.log(`Best epoch: ${metrics.bestEpoch + 1}`);
    console.log(
        `Best validation loss: ${Math.min(...metrics.valLoss).toFixed(6)}`,
    );
    console.log(
        `Final validation loss: ${metrics.valLoss[metrics.valLoss.length - 1].toFixed(6)}`,
    );

    // Cleanup
    model.dispose();
}
