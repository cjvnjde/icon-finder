import tf from "@tensorflow/tfjs-node";
import fs from "node:fs";
import path from "node:path";
import {getFlattenedInputShape, getInputShape, loadProcessedData,} from "./preprocess.js";
import {createModel} from "./createModel.js";
import {createTripletLoss} from "./createTripletLoss.js";
import {generateTriplets} from "./generateTriplets.js";

const config = {
    modelDir: "./models/icon-similarity",
    epochs: 200,
    batchSize: 32,
    // A slightly lower, more stable learning rate is often better for triplet loss.
    learningRate: 0.0001,
    validationSplit: 0.2,
    embeddingDim: 128,
    // Increased patience to allow the model more time to converge.
    patience: 40,
    minDelta: 0.0001,
    margin: 0.2,
};


/**
 * Custom callback to monitor training progress
 */
class TripletLossCallback extends tf.Callback {
    constructor() {
        super();
        this.losses = [];
        this.valLosses = [];
        this.bestValLoss = Infinity;
    }

    setCurrentEpoch(epoch) {
        this.currentEpoch = epoch;
    }

    async onEpochEnd(epoch, logs) {
        const loss = typeof logs.loss === "number" ? logs.loss : logs.loss.dataSync()[0];
        const valLoss = typeof logs.val_loss === "number" ? logs.val_loss : logs.val_loss.dataSync()[0];

        this.losses.push(loss);
        this.valLosses.push(valLoss);

        // Check for NaN
        if (isNaN(loss) || isNaN(valLoss)) {
            console.error("NaN loss detected! Training may be unstable.");
            return;
        }

        // Check for improvement
        const improvement = this.bestValLoss - valLoss;
        if (valLoss < this.bestValLoss) {
            this.bestValLoss = valLoss;
        }

        // Log meaningful metrics with improvement indicator
        const improvementStr = improvement > 0 ? `↓${improvement.toFixed(6)}` : "";
        console.log(
            `Epoch ${this.currentEpoch + 1}: loss=${loss.toFixed(6)}, val_loss=${valLoss.toFixed(6)} ${improvementStr}`,
        );
    }
}

/**
 * Train the model with improved strategy
 */
export async function trainModel() {
    console.log("Loading processed data...");
    const {metadata, features} = loadProcessedData();

    console.log(`Training on ${metadata.totalIcons} icons`);
    console.log(`Feature dimensions: ${metadata.featureLength}`);

    // Validate data
    if (!features || features.length === 0) {
        throw new Error("No features found in processed data");
    }

    const inputShape = getInputShape();
    console.log(`Input shape: ${inputShape}`);

    // Create model
    console.log("Creating model...");
    const model = createModel(inputShape, config.embeddingDim);

    // Create initial optimizer
    let optimizer = tf.train.adam(config.learningRate); // Start with warmup LR

    // Compile model
    model.compile({
        optimizer: optimizer,
        loss: createTripletLoss(config.margin),
    });

    // Print model summary
    model.summary();

    // Calculate number of triplets per epoch
    const tripletsPerEpoch = Math.min(features.length, 4000); // Adjusted for clarity
    const validationTriplets = Math.floor(tripletsPerEpoch * config.validationSplit);
    const trainingTriplets = tripletsPerEpoch - validationTriplets;

    console.log(`Triplets per epoch: ${tripletsPerEpoch} (${trainingTriplets} training, ${validationTriplets} validation)`);

    // Generate validation data with hard negative mining
    console.log("Generating validation data...");
    const {anchors: anchorsVal, positives: positivesVal, negatives: negativesVal} = generateTriplets(
        features,
        validationTriplets,
        {hardNegativeMining: true, isValidation: true}
    );

    // Concatenate validation triplets
    const xVal = tf.concat([anchorsVal, positivesVal, negativesVal], 0);
    const yVal = tf.zeros([validationTriplets * 3, config.embeddingDim]);

    // Ensure model directory exists
    const modelDir = path.dirname(config.modelDir);
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, {recursive: true});
    }

    const tripletCallback = new TripletLossCallback();

    console.log("Starting training with improved strategy...");
    let bestValLoss = Infinity;
    let epochsSinceImprovement = 0;
    const history = {loss: [], val_loss: []};

    for (let epoch = 0; epoch < config.epochs; epoch++) {
        tripletCallback.setCurrentEpoch(epoch);

        // Generate new training data each epoch with progressive difficulty
        const {anchors: anchorsTrain, positives: positivesTrain, negatives: negativesTrain} = generateTriplets(
            features,
            trainingTriplets,
            {hardNegativeMining: true}
        );

        // Concatenate training triplets
        const xTrain = tf.concat([anchorsTrain, positivesTrain, negativesTrain], 0);
        const yTrain = tf.zeros([trainingTriplets * 3, config.embeddingDim]);

        // Train for one epoch
        const epochHistory = await model.fit(xTrain, yTrain, {
            epochs: 1,
            batchSize: config.batchSize,
            validationData: [xVal, yVal],
            callbacks: [tripletCallback],
            verbose: 0,
            shuffle: true,
        });

        // Record history
        tf.dispose([xTrain, yTrain, anchorsTrain, positivesTrain, negativesTrain]);
        const trainLoss = epochHistory.history.loss[0];
        const valLoss = epochHistory.history.val_loss[0];
        history.loss.push(trainLoss);
        history.val_loss.push(valLoss);

        // Check for improvement
        if (valLoss < bestValLoss - config.minDelta) {
            bestValLoss = valLoss;
            epochsSinceImprovement = 0;
            await model.save(`file://${config.modelDir}-best`);
            console.log(`  -> New best model saved (val_loss: ${valLoss.toFixed(6)})`);
        } else {
            epochsSinceImprovement++;
        }
        if (epochsSinceImprovement >= config.patience) {
            console.log(`Early stopping triggered at epoch ${epoch + 1}`);
            break;
        }


        // Check if training should stop
        if (isNaN(valLoss) || isNaN(trainLoss)) {
            console.error("NaN detected, stopping training.");
            break;
        }

        // Early stopping
        if (epochsSinceImprovement >= config.patience) {
            console.log(`Early stopping triggered at epoch ${epoch + 1}`);
            break;
        }

        // Log progress
        if ((epoch + 1) % 10 === 0) {
            const avgLoss = history.loss.slice(-10).reduce((a, b) => a + b) / 10;
            const avgValLoss = history.val_loss.slice(-10).reduce((a, b) => a + b) / 10;
            console.log(`Progress: ${epoch + 1}/${config.epochs} epochs | Avg loss: ${avgLoss.toFixed(6)} | Avg val_loss: ${avgValLoss.toFixed(6)}`);
        }
    }

    xVal.dispose();
    yVal.dispose();
    anchorsVal.dispose();
    positivesVal.dispose();
    negativesVal.dispose();

    // Save final model
    console.log("Saving final model...");
    await model.save(`file://${config.modelDir}`);

    // Save comprehensive metrics
    const metrics = {
        loss: history.loss,
        valLoss: history.val_loss,
        epochs: history.loss.length,
        bestEpoch: history.val_loss.indexOf(Math.min(...history.val_loss)),
        bestValLoss: Math.min(...history.val_loss),
        finalValLoss: history.val_loss[history.val_loss.length - 1],
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

    console.log("\n=== Training Summary ===");
    console.log(`Total epochs: ${metrics.epochs}`);
    console.log(`Best epoch: ${metrics.bestEpoch + 1}`);
    console.log(`Best validation loss: ${metrics.bestValLoss.toFixed(6)}`);
    console.log(`Final validation loss: ${metrics.finalValLoss.toFixed(6)}`);
    console.log(`Improvement: ${((1 - metrics.bestValLoss) * 100).toFixed(2)}%`);

    // Cleanup
    model.dispose();
    optimizer.dispose();
}
