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
  batchSize: 24,
  learningRate: 0.001, // Increased learning rate
  validationSplit: 0.2,
  embeddingDim: 128,
  patience: 15, // Increased patience
  minDelta: 0.001,
  margin: 0.5, // Increased margin
};

/**
 * Create a stable triplet loss function
 * @param {number} margin - Margin for triplet loss
 * @returns {Function} Loss function
 */
function createTripletLoss(margin = 0.5) {
  return (yTrue, yPred) => {
    return tf.tidy(() => {
      const batchSize = yPred.shape[0];
      const embeddingDim = yPred.shape[1];
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

      // L2 normalize embeddings
      const epsilon = 1e-8;
      const anchorNorm = tf.norm(anchors, 2, 1, true).add(epsilon);
      const positiveNorm = tf.norm(positives, 2, 1, true).add(epsilon);
      const negativeNorm = tf.norm(negatives, 2, 1, true).add(epsilon);

      const normalizedAnchors = anchors.div(anchorNorm);
      const normalizedPositives = positives.div(positiveNorm);
      const normalizedNegatives = negatives.div(negativeNorm);

      // Calculate Euclidean distances
      const posDistance = tf.norm(
        normalizedAnchors.sub(normalizedPositives),
        2,
        1,
      );
      const negDistance = tf.norm(
        normalizedAnchors.sub(normalizedNegatives),
        2,
        1,
      );

      // Triplet loss: max(0, pos_dist - neg_dist + margin)
      const losses = tf.maximum(
        tf.scalar(0),
        posDistance.sub(negDistance).add(margin),
      );

      return tf.mean(losses);
    });
  };
}

/**
 * Augment an image tensor with more aggressive augmentation
 * @param {tf.Tensor3D} tensor - Image tensor
 * @returns {tf.Tensor3D} Augmented image tensor
 */
function augmentImageTensor(tensor) {
  return tf.tidy(() => {
    let augmented = tensor.clone(); // Start with a clone

    // Random horizontal flip
    if (Math.random() > 0.5) {
      const flipped = tf.reverse(augmented, 1);
      augmented.dispose();
      augmented = flipped;
    }

    // Random brightness adjustment (more aggressive)
    if (Math.random() > 0.3) {
      const brightness = 0.7 + Math.random() * 0.6; // 0.7 to 1.3
      const brightened = tf.mul(augmented, brightness);
      const clipped = tf.clipByValue(brightened, 0, 1);
      augmented.dispose();
      brightened.dispose();
      augmented = clipped;
    }

    // Random contrast adjustment (more aggressive)
    if (Math.random() > 0.3) {
      const contrast = 0.7 + Math.random() * 0.6; // 0.7 to 1.3
      const mean = tf.mean(augmented);
      const centered = tf.sub(augmented, mean);
      const contrasted = tf.mul(centered, contrast);
      const final = tf.add(contrasted, mean);
      const clipped = tf.clipByValue(final, 0, 1);

      augmented.dispose();
      mean.dispose();
      centered.dispose();
      contrasted.dispose();
      final.dispose();
      augmented = clipped;
    }

    // Add noise
    if (Math.random() > 0.5) {
      const noise = tf.randomNormal(augmented.shape, 0, 0.05);
      const noisy = tf.add(augmented, noise);
      const clipped = tf.clipByValue(noisy, 0, 1);

      augmented.dispose();
      noise.dispose();
      noisy.dispose();
      augmented = clipped;
    }

    // Random small translation/shift (simplified to avoid complex operations)
    if (Math.random() > 0.8) {
      // Reduced frequency to avoid complexity
      const [height, width, channels] = augmented.shape;
      if (height > 4 && width > 4) {
        // Only if image is large enough
        const cropSize = Math.min(height - 2, width - 2);
        const startY = Math.floor(Math.random() * (height - cropSize));
        const startX = Math.floor(Math.random() * (width - cropSize));

        const cropped = tf.slice(
          augmented,
          [startY, startX, 0],
          [cropSize, cropSize, channels],
        );
        const resized = tf.image.resizeBilinear(cropped, [height, width]);

        augmented.dispose();
        cropped.dispose();
        augmented = resized;
      }
    }

    return augmented;
  });
}

/**
 * Create similarity groups based on icon categories or features
 * This is a simplified version - in practice you'd want better grouping
 * @param {Array} features - Array of feature arrays
 * @param {Array} metadata - Array of metadata for each icon
 * @returns {Map} Map of group ID to array of indices
 */
function createSimilarityGroups(features, metadata) {
  const groups = new Map();

  // For now, create simple groups based on basic image properties
  // In practice, you'd want to use semantic categories or manual labeling
  for (let i = 0; i < features.length; i++) {
    const feature = features[i];

    // Simple grouping based on average brightness and variance
    const avgBrightness =
      feature.reduce((sum, val) => sum + val, 0) / feature.length;
    const variance =
      feature.reduce((sum, val) => sum + Math.pow(val - avgBrightness, 2), 0) /
      feature.length;

    // Create group key based on quantized brightness and variance
    const brightnessGroup = Math.floor(avgBrightness * 4); // 0-3
    const varianceGroup = Math.floor(variance * 10); // 0-9
    const groupKey = `${brightnessGroup}_${varianceGroup}`;

    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(i);
  }

  // Filter out groups with only one item
  const filteredGroups = new Map();
  for (const [key, indices] of groups) {
    if (indices.length > 1) {
      filteredGroups.set(key, indices);
    }
  }

  return filteredGroups;
}

/**
 * Generate better triplets for training
 * @param {Array} features - Array of feature arrays
 * @param {Array} metadata - Array of metadata for each icon
 * @param {number} numTriplets - Number of triplets to generate
 * @returns {{x: tf.Tensor4D, y: tf.Tensor2D}} Training data
 */
function generateTriplets(features, metadata, numTriplets) {
  return tf.tidy(() => {
    const inputShape = getInputShape();
    const similarityGroups = createSimilarityGroups(features, metadata);
    const groupKeys = Array.from(similarityGroups.keys());

    console.log(`Generated ${similarityGroups.size} similarity groups`);
    console.log(`Generating ${numTriplets} triplets...`);

    // Pre-allocate arrays for better performance
    const anchorData = [];
    const positiveData = [];
    const negativeData = [];

    for (let i = 0; i < numTriplets; i++) {
      let anchorIdx, positiveIdx, negativeIdx;

      // 70% of the time, use similarity groups for positive pairs
      if (Math.random() < 0.7 && groupKeys.length > 0) {
        // Pick a random group
        const groupKey =
          groupKeys[Math.floor(Math.random() * groupKeys.length)];
        const groupIndices = similarityGroups.get(groupKey);

        if (groupIndices.length >= 2) {
          // Pick anchor and positive from the same group
          anchorIdx =
            groupIndices[Math.floor(Math.random() * groupIndices.length)];
          do {
            positiveIdx =
              groupIndices[Math.floor(Math.random() * groupIndices.length)];
          } while (positiveIdx === anchorIdx);
        } else {
          // Fallback to random selection
          anchorIdx = Math.floor(Math.random() * features.length);
          positiveIdx = Math.floor(Math.random() * features.length);
          while (positiveIdx === anchorIdx) {
            positiveIdx = Math.floor(Math.random() * features.length);
          }
        }
      } else {
        // 30% of the time, use random positive pairs (harder training)
        anchorIdx = Math.floor(Math.random() * features.length);
        positiveIdx = Math.floor(Math.random() * features.length);
        while (positiveIdx === anchorIdx) {
          positiveIdx = Math.floor(Math.random() * features.length);
        }
      }

      // Always pick a random negative
      negativeIdx = Math.floor(Math.random() * features.length);
      while (negativeIdx === anchorIdx || negativeIdx === positiveIdx) {
        negativeIdx = Math.floor(Math.random() * features.length);
      }

      // Store the data (not tensors yet)
      anchorData.push(features[anchorIdx]);
      positiveData.push(features[positiveIdx]);
      negativeData.push(features[negativeIdx]);
    }

    // Now create all tensors at once and apply augmentation
    const anchorTensors = anchorData.map((data) => {
      const tensor = tf.tensor3d(data, inputShape);
      return augmentImageTensor(tensor);
    });

    const positiveTensors = positiveData.map((data) => {
      const tensor = tf.tensor3d(data, inputShape);
      return augmentImageTensor(tensor);
    });

    const negativeTensors = negativeData.map((data) => {
      const tensor = tf.tensor3d(data, inputShape);
      return augmentImageTensor(tensor);
    });

    // Stack all tensors
    const anchorStack = tf.stack(anchorTensors);
    const positiveStack = tf.stack(positiveTensors);
    const negativeStack = tf.stack(negativeTensors);

    // Concatenate into single tensor [anchors, positives, negatives]
    const x = tf.concat([anchorStack, positiveStack, negativeStack], 0);

    // Create dummy labels (not used but required by Keras)
    const y = tf.zeros([numTriplets * 3, config.embeddingDim]);

    return { x, y };
  });
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
  const { metadata, features } = loadProcessedData();

  console.log(`Training on ${metadata.totalIcons} icons`);
  console.log(`Feature dimensions: ${metadata.featureLength}`);

  const inputShape = getInputShape();
  console.log(`Input shape: ${inputShape}`);

  // Create model
  console.log("Creating model...");
  const model = createModel(inputShape, config.embeddingDim);

  // Use a learning rate scheduler
  const initialLearningRate = config.learningRate;
  let currentLearningRate = initialLearningRate;

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
  const { x: xVal, y: yVal } = generateTriplets(
    features,
    metadata,
    validationTriplets,
  );

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
    const { x: xTrain, y: yTrain } = generateTriplets(
      features,
      metadata,
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
