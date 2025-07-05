import tf from "@tensorflow/tfjs-node";

/**
 * Create the embedding model
 * @param {number[]} inputShape - Shape of input images [height, width, channels]
 * @param {number} embeddingDim - Dimension of the embedding space
 * @returns {tf.Sequential} The model
 */
export function createModel(inputShape, embeddingDim) {
  const model = tf.sequential();

  model.add(
    tf.layers.inputLayer({
      inputShape,
    }),
  );

  // First conv block
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  // Second conv block
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  // Third conv block
  model.add(
    tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "heNormal",
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  // Flatten and dense layers
  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 256,
      activation: "relu",
      kernelInitializer: "heNormal",
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(
    tf.layers.dropout({
      rate: 0.3, // Reduced dropout rate
    }),
  );

  // Embedding layer (no activation, will be L2 normalized)
  model.add(
    tf.layers.dense({
      units: embeddingDim,
      activation: "linear",
      name: "embedding",
      kernelInitializer: "glorotUniform",
    }),
  );

  return model;
}
