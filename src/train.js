import tf from "@tensorflow/tfjs-node";
import fs from "node:fs";
import path from "node:path";
import {
  loadProcessedData,
  getInputShape,
  getFlattenedInputShape,
} from "./preprocess.js";
import { createModel } from "./createModel.js";

const config = {
  modelDir: "./models/icon-similarity",
  epochs: 100,
  batchSize: 24, // Reduced batch size
  learningRate: 0.0001, // Much lower learning rate
  validationSplit: 0.2,
  embeddingDim: 128,
  patience: 10,
  minDelta: 0.001,
  margin: 0.2, // Reduced margin for stability
};

/**
 * Create a stable triplet loss function
 * @param {number} margin - Margin for triplet loss
 * @returns {Function} Loss function
 */
function createTripletLoss(margin = 0.5) {
  return (yTrue, yPred) => {
    return tf.tidy(() => {
      // Get batch dimensions
      const batchSize = yPred.shape[0];
      const embeddingDim = yPred.shape[1];

      // Ensure batch size is divisible by 3
      const numTriplets = Math.floor(batchSize / 3);
      if (numTriplets === 0) {
        return tf.scalar(0);
      }

      // Split into anchor, positive, negative
      const anchors = yPred.slice([0, 0], [numTriplets, embeddingDim]);
      const positives = yPred.slice(
        [numTriplets, 0],
        [numTriplets, embeddingDim],
      );
      const negatives = yPred.slice(
        [numTriplets * 2, 0],
        [numTriplets, embeddingDim],
      );

      // L2 normalize embeddings with better numerical stability
      const epsilon = tf.scalar(1e-6);

      // Normalize each set separately
      const anchorNorm = tf.sqrt(
        tf.maximum(tf.sum(tf.square(anchors), 1, true), epsilon),
      );
      const positiveNorm = tf.sqrt(
        tf.maximum(tf.sum(tf.square(positives), 1, true), epsilon),
      );
      const negativeNorm = tf.sqrt(
        tf.maximum(tf.sum(tf.square(negatives), 1, true), epsilon),
      );

      const normalizedAnchors = tf.div(anchors, anchorNorm);
      const normalizedPositives = tf.div(positives, positiveNorm);
      const normalizedNegatives = tf.div(negatives, negativeNorm);

      // Calculate squared Euclidean distances
      // d(a,b) = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>
      // Since we normalized, ||a||^2 = ||b||^2 = 1
      // So d(a,b) = 2 - 2*<a,b>

      const anchorPosDot = tf.sum(
        tf.mul(normalizedAnchors, normalizedPositives),
        1,
      );
      const anchorNegDot = tf.sum(
        tf.mul(normalizedAnchors, normalizedNegatives),
        1,
      );

      // Clamp dot products to [-1, 1] to avoid numerical issues
      const clampedPosDot = tf.clipByValue(anchorPosDot, -1, 1);
      const clampedNegDot = tf.clipByValue(anchorNegDot, -1, 1);

      // Convert to distances
      const posDistance = tf.mul(
        tf.scalar(2),
        tf.sub(tf.scalar(1), clampedPosDot),
      );
      const negDistance = tf.mul(
        tf.scalar(2),
        tf.sub(tf.scalar(1), clampedNegDot),
      );

      // Triplet loss with margin
      const losses = tf.maximum(
        tf.scalar(0),
        tf.add(tf.sub(posDistance, negDistance), tf.scalar(margin)),
      );

      // Return mean loss with a small epsilon to avoid exact zero
      const meanLoss = tf.mean(losses);
      return tf.add(meanLoss, tf.scalar(1e-7));
    });
  };
}

/**
 * Augment an image tensor
 * @param {tf.Tensor3D} tensor - Image tensor
 * @returns {tf.Tensor3D} Augmented image tensor
 */
function augmentImageTensor(tensor) {
  return tf.tidy(() => {
    let augmented = tensor;

    // Random horizontal flip
    if (Math.random() > 0.5) {
      augmented = tf.reverse(augmented, 1);
    }

    // Random small rotation (-15 to +15 degrees)
    if (Math.random() > 0.7) {
      const angle = (Math.random() - 0.5) * 30 * (Math.PI / 180);
      // Simple rotation using affine transformation would be ideal here
      // but tfjs doesn't have built-in image rotation, so we'll skip this
    }

    // Random brightness adjustment
    if (Math.random() > 0.5) {
      const brightness = 1 + (Math.random() - 0.5) * 0.2;
      augmented = tf.mul(augmented, tf.scalar(brightness));
      augmented = tf.clipByValue(augmented, 0, 1);
    }

    // Random contrast adjustment
    if (Math.random() > 0.5) {
      const contrast = 1 + (Math.random() - 0.5) * 0.2;
      const mean = tf.mean(augmented);
      augmented = tf.add(
        tf.mul(tf.sub(augmented, mean), tf.scalar(contrast)),
        mean,
      );
      augmented = tf.clipByValue(augmented, 0, 1);
    }

    // Small amount of noise
    if (Math.random() > 0.7) {
      const noise = tf.randomNormal(augmented.shape, 0, 0.02);
      augmented = tf.add(augmented, noise);
      augmented = tf.clipByValue(augmented, 0, 1);
    }

    return augmented;
  });
}

/**
 * Generate triplets for training
 * @param {Array} features - Array of feature arrays
 * @param {number} numTriplets - Number of triplets to generate
 * @returns {{x: tf.Tensor4D, y: tf.Tensor2D}} Training data
 */
function generateTriplets(features, numTriplets) {
  const inputShape = getInputShape();

  console.log(`Generating ${numTriplets} triplets...`);

  // Create tensors for all triplets at once
  const anchorIndices = [];
  const positiveIndices = [];
  const negativeIndices = [];

  for (let i = 0; i < numTriplets; i++) {
    // Random anchor
    const anchorIdx = Math.floor(Math.random() * features.length);
    anchorIndices.push(anchorIdx);

    // For positive, we'll use the same image with augmentation
    positiveIndices.push(anchorIdx);

    // Random negative (different from anchor)
    let negativeIdx = Math.floor(Math.random() * features.length);
    while (negativeIdx === anchorIdx) {
      negativeIdx = Math.floor(Math.random() * features.length);
    }
    negativeIndices.push(negativeIdx);
  }

  // Create tensors
  const anchors = [];
  const positives = [];
  const negatives = [];

  for (let i = 0; i < numTriplets; i++) {
    // Get anchor tensor
    const anchorData = features[anchorIndices[i]];
    const anchorTensor = tf.tensor3d(anchorData, inputShape);
    anchors.push(anchorTensor);

    // Create positive by augmenting anchor
    const positiveTensor = augmentImageTensor(anchorTensor);
    positives.push(positiveTensor);

    // Get negative tensor
    const negativeData = features[negativeIndices[i]];
    const negativeTensor = tf.tensor3d(negativeData, inputShape);
    negatives.push(negativeTensor);
  }

  // Stack all tensors
  const anchorStack = tf.stack(anchors);
  const positiveStack = tf.stack(positives);
  const negativeStack = tf.stack(negatives);

  // Concatenate into single tensor [anchors, positives, negatives]
  const x = tf.concat([anchorStack, positiveStack, negativeStack], 0);

  // Create dummy labels (not used but required by Keras)
  const y = tf.zeros([numTriplets * 3, config.embeddingDim]);

  // Clean up individual tensors
  anchors.forEach((t) => t.dispose());
  positives.forEach((t) => t.dispose());
  negatives.forEach((t) => t.dispose());
  anchorStack.dispose();
  positiveStack.dispose();
  negativeStack.dispose();

  return { x, y };
}

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
    let loss = logs.loss;
    let valLoss = logs.val_loss;
    // Handle different types of loss values
    if (loss && loss.dataSync) {
      // It's a tensor
      const data = loss.dataSync();
      loss = data[0];
    } else if (loss && loss.data) {
      // It's a promise or has data method
      const data = await loss.data();
      loss = Array.isArray(data) ? data[0] : data;
    } else if (Array.isArray(loss)) {
      loss = loss[0];
    }

    if (valLoss !== undefined && valLoss !== null) {
      if (valLoss && valLoss.dataSync) {
        const data = valLoss.dataSync();
        valLoss = data[0];
      } else if (valLoss && valLoss.data) {
        const data = await valLoss.data();
        valLoss = Array.isArray(data) ? data[0] : data;
      } else if (Array.isArray(valLoss)) {
        valLoss = valLoss[0];
      }
    }

    // Ensure we have numbers
    loss = Number(loss);
    if (valLoss !== undefined && valLoss !== null) {
      valLoss = Number(valLoss);
    }

    this.losses.push(loss);
    if (valLoss !== undefined && valLoss !== null) {
      this.valLosses.push(valLoss);
    }

    // Check for NaN
    if (isNaN(loss)) {
      console.error("NaN loss detected! Training may be unstable.");
    }

    // Log meaningful metrics
    console.log(
      `Epoch ${epoch + 1} | ${this.currentEpoch}: loss=${loss.toFixed(4)}, val_loss=${valLoss.toFixed(4)}`,
    );
  }
}

/**
 * Train the model
 */
export async function trainModel() {
  console.log("Loading processed data...");
  const { metadata, features } = loadProcessedData();

  console.log(`Training on ${metadata.totalIcons} icons`);
  console.log(`Feature dimensions: ${metadata.featureLength}`);

  const inputShape = getInputShape();
  console.log(`Input shape: ${inputShape}`);

  // Create model
  console.log("Creating model...");
  const model = createModel(inputShape, config.embeddingDim);

  // Compile model
  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: createTripletLoss(config.margin),
  });

  // Print model summary
  model.summary();

  // Calculate number of triplets per epoch
  const tripletsPerEpoch = Math.min(features.length * 2, 3000);
  const validationTriplets = Math.floor(tripletsPerEpoch * 0.2);
  const trainingTriplets = tripletsPerEpoch - validationTriplets;

  // Generate validation data once
  console.log("Generating validation data...");
  const { x: xVal, y: yVal } = generateTriplets(features, validationTriplets);

  // Ensure model directory exists
  const modelDir = path.dirname(config.modelDir);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }

  const tripletCallback = new TripletLossCallback();

  console.log("Starting training...");
  let bestValLoss = Infinity;
  let bestWeights = null;
  let epochsSinceImprovement = 0;
  const history = { loss: [], val_loss: [] };

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    tripletCallback.setCurrentEpoch(epoch);
    // Generate new training data each epoch
    const { x: xTrain, y: yTrain } = generateTriplets(
      features,
      trainingTriplets,
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
    history.loss.push(epochHistory.history.loss[0]);
    history.val_loss.push(epochHistory.history.val_loss[0]);

    // Check for early stopping manually
    const currentValLoss = epochHistory.history.val_loss[0];

    if (currentValLoss < bestValLoss - config.minDelta) {
      bestValLoss = currentValLoss;
      epochsSinceImprovement = 0;

      // Save best weights
      if (bestWeights) {
        bestWeights.forEach((w) => w.dispose());
      }
      bestWeights = model.getWeights().map((w) => w.clone());

      // Save best model
      await model.save(`file://${config.modelDir}-best`);
    } else {
      epochsSinceImprovement++;
    }

    // Clean up training data
    xTrain.dispose();
    yTrain.dispose();

    // Check if training should stop
    if (isNaN(currentValLoss) || isNaN(epochHistory.history.loss[0])) {
      console.error("NaN detected, stopping training.");
      break;
    }

    // Manual early stopping
    if (epochsSinceImprovement >= config.patience) {
      console.log(`Early stopping triggered at epoch ${epoch + 1}`);
      // Restore best weights
      if (bestWeights) {
        model.setWeights(bestWeights);
      }
      break;
    }
  }

  // Clean up best weights if any
  if (bestWeights) {
    bestWeights.forEach((w) => w.dispose());
  }

  // Clean up validation data
  xVal.dispose();
  yVal.dispose();

  // Save final model
  console.log("Saving model...");
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
    `Final validation loss: ${metrics.valLoss[metrics.valLoss.length - 1]}`,
  );

  // Cleanup
  model.dispose();
}

/**
 * Load the trained model
 * @returns {Promise<tf.LayersModel>} The loaded model
 */
export async function loadTrainedModel() {
  const modelPath = `file://${config.modelDir}`;
  return await tf.loadLayersModel(modelPath);
}

/**
 * Load model metadata
 * @returns {Object} Model metadata
 */
export function loadModelMetadata() {
  const modelDir = path.dirname(config.modelDir);
  const metadataPath = path.join(modelDir, "model-metadata.json");

  if (!fs.existsSync(metadataPath)) {
    throw new Error(`Model metadata not found at ${metadataPath}`);
  }

  return JSON.parse(fs.readFileSync(metadataPath, "utf8"));
}

/**
 * Load training metrics
 * @returns {Object} Training metrics
 */
export function loadTrainingMetrics() {
  const modelDir = path.dirname(config.modelDir);
  const metricsPath = path.join(modelDir, "training-metrics.json");

  if (!fs.existsSync(metricsPath)) {
    throw new Error(`Training metrics not found at ${metricsPath}`);
  }

  return JSON.parse(fs.readFileSync(metricsPath, "utf8"));
}
